"""Text deduplication using MinHash/LSH for large-scale corpus processing.

Handles both exact and near-duplicate detection for corpus curation at scale,
as well as training/evaluation corpus overlap filtering with grapheme-safe
Unicode NFC normalization.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from adhan_slm.core import get_logger

logger = get_logger(__name__)


def extract_graphemes(text: str) -> List[str]:
    """Segment text into grapheme clusters (aksharas) in a grapheme-safe manner.

    Preserves Tamil aksharas (base consonants + vowel signs/pulli) and
    Unicode combining marks as atomic clusters so shingles never split them.

    Args:
        text: Text to segment.

    Returns:
        List[str]: Grapheme clusters.
    """
    text = unicodedata.normalize("NFC", text)
    clusters: List[str] = []
    for ch in text:
        cp = ord(ch)
        # Tamil matras (0x0BBE-0x0BCC), virama/pulli (0x0BCD), or standard combining marks
        if clusters and ((0x0BBE <= cp <= 0x0BCD) or unicodedata.combining(ch) > 0):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def _extract_text(item: Any) -> str:
    """Extract string text from a string or document dictionary."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("text", "content", "body", "sentence", "tamil"):
            val = item.get(k)
            if isinstance(val, str):
                return val
    return str(item)


class TextDeduplicator:
    """MinHash/LSH-based deduplication for 300M+ token corpus.

    Provides efficient deduplication through:
    1. Exact match detection (SHA-256 hashing on NFC-normalized text)
    2. Near-duplicate detection (MinHash + LSH with configurable similarity threshold)
    3. Per-source deduplication
    4. Train/eval overlap filtering

    This is production-grade for large-scale corpus curation.
    """

    @staticmethod
    def _derive_lsh_params(num_perm: int, threshold: float) -> Tuple[int, int, float]:
        """Derive (bands, rows, catch_probability) favoring recall.

        Finds all (bands, rows) factor pairs where bands * rows == num_perm.
        Requires the probability of catching a pair at the threshold:
            P(catch) = 1.0 - (1.0 - threshold ** rows) ** bands
        to be >= 0.99. Among splits satisfying this, chooses the one with the
        largest rows (fewest candidate collisions).
        """
        factors: List[Tuple[int, int]] = []
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                factors.append((b, r))

        valid: List[Tuple[int, int, float]] = []
        all_splits: List[Tuple[int, int, float]] = []
        for b, r in factors:
            p_catch = 1.0 - (1.0 - (threshold**r)) ** b
            all_splits.append((b, r, p_catch))
            if p_catch >= 0.99:
                valid.append((b, r, p_catch))

        if valid:
            best = max(valid, key=lambda x: (x[1], x[2]))
        else:
            best = max(all_splits, key=lambda x: x[2])
        return best[0], best[1], best[2]

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        shingle_size: int = 5,
        seed: int = 42,
        use_lsh: bool = True,
        bands: Optional[int] = None,
        rows: Optional[int] = None,
    ):
        """Initialize TextDeduplicator.

        Args:
            num_perm: Number of hash permutations for MinHash (default: 128).
                     Higher = more accurate but slower. 128 is standard.
            threshold: Jaccard similarity threshold for near-duplicates (0-1).
                     0.85 = 85% similarity required. Default: 0.85.
            shingle_size: Size of character shingles for Jaccard calculation.
                         Larger = more specific matching. Default: 5.
            seed: Random seed for reproducibility. Default: 42.
            use_lsh: If True, use LSH banding for candidate lookup (default: True).
            bands: Optional override for number of LSH bands.
            rows: Optional override for rows per LSH band.
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.seed = seed
        self.use_lsh = use_lsh
        self.logger = logger

        # Validate / configure LSH parameters
        if bands is not None and rows is not None:
            if bands <= 0 or rows <= 0 or bands * rows != self.num_perm:
                raise ValueError(
                    f"Invalid LSH parameters: bands ({bands}) * rows ({rows}) != num_perm ({self.num_perm})"
                )
            self.bands = bands
            self.rows = rows
            self.catch_probability = 1.0 - (1.0 - (self.threshold**self.rows)) ** self.bands
        elif bands is not None:
            if bands <= 0 or self.num_perm % bands != 0:
                raise ValueError(
                    f"Invalid LSH bands: num_perm ({self.num_perm}) is not divisible by bands ({bands})"
                )
            self.bands = bands
            self.rows = self.num_perm // bands
            self.catch_probability = 1.0 - (1.0 - (self.threshold**self.rows)) ** self.bands
        elif rows is not None:
            if rows <= 0 or self.num_perm % rows != 0:
                raise ValueError(
                    f"Invalid LSH rows: num_perm ({self.num_perm}) is not divisible by rows ({rows})"
                )
            self.rows = rows
            self.bands = self.num_perm // rows
            self.catch_probability = 1.0 - (1.0 - (self.threshold**self.rows)) ** self.bands
        else:
            self.bands, self.rows, self.catch_probability = self._derive_lsh_params(
                self.num_perm, self.threshold
            )

        # Track hashes for exact dedup
        self.exact_hashes: Set[str] = set()

        # Track MinHash signatures for near-duplicate detection
        self.minhash_signatures: Dict[str, List[int]] = {}
        self.seen_docs: Dict[str, Dict[str, Any]] = {}

        # LSH buckets: one dictionary per band mapping band tuple to doc_id list
        self.buckets: List[Dict[Tuple[int, ...], List[str]]] = [{} for _ in range(self.bands)]

        self.logger.info(
            f"Deduplicator initialized: num_perm={num_perm}, "
            f"threshold={threshold}, shingle_size={shingle_size}, "
            f"use_lsh={use_lsh}, bands={self.bands}, rows={self.rows}, "
            f"catch_probability={self.catch_probability:.4f}"
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for consistent hashing.

        Applies Unicode NFC normalization, lowercases, and collapses whitespace.
        Grapheme-safe for Tamil and multilingual text.

        Args:
            text: Raw text to normalize.

        Returns:
            str: Normalized text (NFC, lowercase, whitespace collapsed).
        """
        # NFC normalization ensures composed and decomposed characters match
        text = unicodedata.normalize("NFC", text)
        text = text.lower().strip()
        text = " ".join(text.split())
        return unicodedata.normalize("NFC", text)

    def _compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text.

        Args:
            text: Text to hash.

        Returns:
            str: Hex-encoded SHA-256 hash.
        """
        normalized = self.normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _get_shingles(self, text: str) -> Set[str]:
        """Get grapheme-safe character shingles from text.

        Preserves Tamil aksharas and combining marks without splitting clusters.

        Args:
            text: Text to extract shingles from.

        Returns:
            Set[str]: Character/grapheme shingles of specified size.
        """
        normalized = self.normalize_text(text)
        if not normalized:
            return set()

        clusters = extract_graphemes(normalized)
        if len(clusters) < self.shingle_size:
            return {"".join(clusters)}

        shingles: Set[str] = set()
        for i in range(len(clusters) - self.shingle_size + 1):
            shingles.add("".join(clusters[i : i + self.shingle_size]))
        return shingles

    def _minhash(self, text: str) -> List[int]:
        """Compute MinHash signature for text.

        Uses permutation-based hashing on character shingles.

        Args:
            text: Text to compute MinHash for.

        Returns:
            List[int]: MinHash signature (num_perm integers).
        """
        shingles = self._get_shingles(text)

        if not shingles:
            return [0] * self.num_perm

        signature: List[int] = []
        for i in range(self.num_perm):
            min_hash: int = 1 << 128
            for shingle in shingles:
                hash_val = int(
                    hashlib.md5(
                        f"{self.seed + i}{shingle}".encode(), usedforsecurity=False
                    ).hexdigest(),
                    16,
                )
                min_hash = min(min_hash, hash_val)

            signature.append(int(min_hash % (2**32)))

        return signature

    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures.

        Args:
            sig1: First MinHash signature.
            sig2: Second MinHash signature.

        Returns:
            float: Estimated Jaccard similarity (0-1).
        """
        if len(sig1) != len(sig2):
            return 0.0

        matches = sum(1 for s1, s2 in zip(sig1, sig2) if s1 == s2)
        return matches / len(sig1)

    def is_duplicate(
        self, text: str, doc_id: str, check_near: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Check if text is a duplicate of previously seen text.

        Returns both exact matches and near-duplicates based on threshold.

        Args:
            text: Text to check.
            doc_id: Unique document ID for tracking.
            check_near: If True, check near-duplicates via MinHash. If False, exact only.

        Returns:
            Tuple[bool, Optional[str]]: (is_duplicate, duplicate_of_doc_id)
        """
        text_hash = self._compute_hash(text)

        # Check exact duplicates
        if text_hash in self.exact_hashes:
            for existing_id, existing_doc in self.seen_docs.items():
                if existing_doc["hash"] == text_hash:
                    return True, existing_id
            return True, None

        signature: Optional[List[int]] = None
        if check_near:
            # Compute MinHash for near-duplicate detection
            signature = self._minhash(text)

            if self.use_lsh:
                # Look up candidates only from buckets that share a band with the new signature
                candidates: Set[str] = set()
                for band_idx in range(self.bands):
                    start = band_idx * self.rows
                    end = start + self.rows
                    band_key = tuple(signature[start:end])
                    bucket = self.buckets[band_idx].get(band_key)
                    if bucket is not None:
                        candidates.update(bucket)

                # VERIFY every candidate with existing Jaccard estimate against threshold
                for cand_id in candidates:
                    cand_sig = self.minhash_signatures[cand_id]
                    similarity = self._jaccard_similarity(signature, cand_sig)
                    if similarity >= self.threshold:
                        return True, cand_id
            else:
                # Check against all previously seen signatures (brute-force)
                for existing_id, existing_sig in self.minhash_signatures.items():
                    similarity = self._jaccard_similarity(signature, existing_sig)
                    if similarity >= self.threshold:
                        return True, existing_id

        # Not a duplicate - remember this text
        self.exact_hashes.add(text_hash)
        if signature is not None:
            self.minhash_signatures[doc_id] = signature
            if self.use_lsh:
                for band_idx in range(self.bands):
                    start = band_idx * self.rows
                    end = start + self.rows
                    band_key = tuple(signature[start:end])
                    if band_key not in self.buckets[band_idx]:
                        self.buckets[band_idx][band_key] = []
                    self.buckets[band_idx][band_key].append(doc_id)
        self.seen_docs[doc_id] = {
            "hash": text_hash,
            "text": text[:200],
        }

        return False, None

    def deduplicate(
        self,
        documents: Iterable[Dict[str, Any]],
        track_sources: bool = True,
        check_near: bool = True,
    ) -> Tuple[Generator[Dict[str, Any], None, None], Dict[str, Any]]:
        """Remove exact and near-duplicates from document stream.

        Yields documents that are NOT duplicates. Returns statistics on removal.

        Args:
            documents: Iterable yielding document dicts with 'text' and 'id' fields.
            track_sources: If True, track which sources had duplicates.
            check_near: If True, check near-duplicates via MinHash. If False, exact only.

        Yields:
            Dict[str, Any]: Non-duplicate documents.

        Returns:
            Dict[str, Any]: Deduplication statistics (counts, rates, per-source breakdown).
        """
        stats: Dict[str, Any] = {
            "total_seen": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "kept": 0,
            "removed": 0,
            "removal_rate": 0.0,
            "per_source": defaultdict(lambda: {"seen": 0, "duplicates": 0}),
        }

        def deduplicate_generator() -> Generator[Dict[str, Any], None, None]:
            """Inner generator for deduplication."""
            for doc in documents:
                stats["total_seen"] = int(stats["total_seen"]) + 1
                text = doc.get("text", "")
                doc_id = doc.get("id", f"doc-{stats['total_seen']}")
                source = doc.get("source", "unknown")

                per_source_dict: Dict[str, Dict[str, int]] = stats["per_source"]
                if track_sources:
                    per_source_dict[source]["seen"] += 1

                # Check for duplicates
                is_dup, dup_of = self.is_duplicate(text, doc_id, check_near=check_near)

                if is_dup:
                    # Determine if exact or near-duplicate
                    text_hash = self._compute_hash(text)
                    if text_hash in self.exact_hashes:
                        stats["exact_duplicates"] = int(stats["exact_duplicates"]) + 1
                        if track_sources:
                            per_source_dict[source]["duplicates"] += 1
                        self.logger.debug(f"Exact duplicate: {doc_id} (matches {dup_of})")
                    else:
                        stats["near_duplicates"] = int(stats["near_duplicates"]) + 1
                        if track_sources:
                            per_source_dict[source]["duplicates"] += 1
                        self.logger.debug(
                            f"Near-duplicate: {doc_id} (matches {dup_of} "
                            f"at {self.threshold:.1%} similarity)"
                        )
                else:
                    # Keep this document
                    stats["kept"] = int(stats["kept"]) + 1
                    yield doc

            # Finalize the shared stats dict in place once the stream is exhausted.
            removed = int(stats["exact_duplicates"]) + int(stats["near_duplicates"])
            stats["removed"] = removed
            stats["removal_rate"] = (
                removed / int(stats["total_seen"]) if int(stats["total_seen"]) > 0 else 0.0
            )
            stats["per_source"] = dict(stats["per_source"])

            self.logger.info(
                f"Deduplication complete: {stats['kept']}/{stats['total_seen']} kept, "
                f"{float(stats['removal_rate']):.1%} removal rate"
            )

        return deduplicate_generator(), stats

    def per_source_dedup(
        self, documents: Iterable[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Remove duplicates within each source separately.

        Useful when sources might have significant overlap (e.g., duplicate
        Reddit posts across different subreddits).

        Args:
            documents: Iterable yielding document dicts.

        Yields:
            Dict[str, Any]: Documents without per-source duplicates.
        """
        source_hashes: Dict[str, Set[str]] = defaultdict(set)

        for doc in documents:
            text = str(doc.get("text", ""))
            source = str(doc.get("source", "unknown"))

            text_hash = self._compute_hash(text)

            if text_hash not in source_hashes[source]:
                # New document for this source
                source_hashes[source].add(text_hash)
                yield doc
            else:
                # Duplicate within source
                self.logger.debug(f"Per-source duplicate in {source}: {doc.get('id')}")

    def write_stats(self, stats: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write deduplication statistics to JSON file.

        Args:
            stats: Statistics dictionary from deduplicate().
            output_path: Path to write JSON report.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Deduplication report written to {output_path}")

    @staticmethod
    def filter_train_eval_overlap(
        train_data: Iterable[Any], eval_data: Iterable[Any]
    ) -> Tuple[List[Any], int]:
        """Filter out training items that appear in the eval corpus."""
        return filter_train_eval_overlap(train_data, eval_data)

    @staticmethod
    def remove_train_eval_overlap(train_data: List[Any], eval_data: Iterable[Any]) -> int:
        """Remove training items in-place that appear in the eval corpus."""
        return remove_train_eval_overlap(train_data, eval_data)

    @staticmethod
    def compute_train_eval_overlap(train_data: Iterable[Any], eval_data: Iterable[Any]) -> int:
        """Count how many training items appear in the evaluation corpus."""
        return compute_train_eval_overlap(train_data, eval_data)


def filter_train_eval_overlap(
    train_data: Iterable[Any],
    eval_data: Iterable[Any],
) -> Tuple[List[Any], int]:
    """Filter out training lines or documents that appear in the eval corpus.

    Matches lines based on NFC-normalized, whitespace-collapsed text to ensure
    safe comparison of Tamil Unicode representations (composed vs decomposed).

    Args:
        train_data: Iterable of training lines (strings) or document dicts.
        eval_data: Iterable of evaluation lines (strings) or document dicts.

    Returns:
        Tuple[List[Any], int]: (filtered_train_items, removed_count)
    """
    eval_set: Set[str] = set()
    for item in eval_data:
        text = _extract_text(item)
        norm = TextDeduplicator.normalize_text(text)
        if norm:
            eval_set.add(norm)

    kept: List[Any] = []
    removed_count = 0
    for item in train_data:
        text = _extract_text(item)
        norm = TextDeduplicator.normalize_text(text)
        if norm and norm in eval_set:
            removed_count += 1
        else:
            kept.append(item)

    return kept, removed_count


def remove_train_eval_overlap(
    train_data: List[Any],
    eval_data: Iterable[Any],
) -> int:
    """Remove any training line in-place that also appears in the eval corpus.

    Matching is performed on NFC-normalized text.

    Args:
        train_data: List of training lines or document dicts (modified in-place).
        eval_data: Iterable of evaluation lines or document dicts.

    Returns:
        int: Number of training items removed.
    """
    kept, removed = filter_train_eval_overlap(train_data, eval_data)
    train_data[:] = kept
    return removed


def compute_train_eval_overlap(
    train_data: Iterable[Any],
    eval_data: Iterable[Any],
) -> int:
    """Count how many training lines appear in the evaluation corpus.

    Matching is performed on NFC-normalized text.

    Args:
        train_data: Iterable of training lines or document dicts.
        eval_data: Iterable of evaluation lines or document dicts.

    Returns:
        int: Number of overlapping lines.
    """
    eval_set = {
        TextDeduplicator.normalize_text(_extract_text(item))
        for item in eval_data
        if TextDeduplicator.normalize_text(_extract_text(item))
    }
    overlap = 0
    for item in train_data:
        norm = TextDeduplicator.normalize_text(_extract_text(item))
        if norm and norm in eval_set:
            overlap += 1
    return overlap
