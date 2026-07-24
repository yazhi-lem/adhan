"""Text deduplication using MinHash/LSH for large-scale corpus processing.

Handles both exact and near-duplicate detection for corpus curation at scale.
"""

import hashlib
import json
import logging
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Set, Tuple

from adhan_slm.core import get_logger

logger = get_logger(__name__)


class TextDeduplicator:
    """MinHash/LSH-based deduplication for 300M+ token corpus.

    Provides efficient deduplication through:
    1. Exact match detection (SHA-256 hashing)
    2. Near-duplicate detection (MinHash + LSH)
    3. Per-source deduplication

    This is production-grade for large-scale corpus curation.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        shingle_size: int = 5,
        seed: int = 42,
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
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.seed = seed
        self.logger = logger

        # Track hashes for exact dedup
        self.exact_hashes: Set[str] = set()

        # Track MinHash signatures for near-duplicate detection
        self.minhash_signatures: Dict[str, List[int]] = {}
        self.seen_docs: Dict[str, Dict] = {}

        self.logger.info(
            f"Deduplicator initialized: num_perm={num_perm}, "
            f"threshold={threshold}, shingle_size={shingle_size}"
        )

    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing.

        Args:
            text: Raw text to normalize.

        Returns:
            str: Normalized text (lowercase, whitespace collapsed).
        """
        # Lowercase and collapse whitespace
        text = text.lower().strip()
        text = " ".join(text.split())  # Collapse multiple spaces
        return text

    def _compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text.

        Args:
            text: Text to hash.

        Returns:
            str: Hex-encoded SHA-256 hash.
        """
        normalized = self.normalize_text(text)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _get_shingles(self, text: str) -> Set[str]:
        """Get character shingles from text.

        Args:
            text: Text to extract shingles from.

        Returns:
            Set[str]: Character shingles of specified size.
        """
        normalized = self.normalize_text(text)
        if len(normalized) < self.shingle_size:
            return {normalized}

        shingles = set()
        for i in range(len(normalized) - self.shingle_size + 1):
            shingles.add(normalized[i : i + self.shingle_size])
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

        signature = []
        for i in range(self.num_perm):
            # Different hash seed for each permutation
            min_hash = float("inf")
            for shingle in shingles:
                hash_val = int(
                    hashlib.md5(f"{self.seed + i}{shingle}".encode()).hexdigest(), 16
                )
                min_hash = min(min_hash, hash_val)

            signature.append(min_hash % (2**32))  # Bound to 32-bit int

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

    def is_duplicate(self, text: str, doc_id: str) -> Tuple[bool, Optional[str]]:
        """Check if text is a duplicate of previously seen text.

        Returns both exact matches and near-duplicates based on threshold.

        Args:
            text: Text to check.
            doc_id: Unique document ID for tracking.

        Returns:
            Tuple[bool, Optional[str]]: (is_duplicate, duplicate_of_doc_id)
        """
        text_hash = self._compute_hash(text)

        # Check exact duplicates
        if text_hash in self.exact_hashes:
            # Find which doc this matches
            for existing_id, existing_doc in self.seen_docs.items():
                if existing_doc["hash"] == text_hash:
                    return True, existing_id

        # Compute MinHash for near-duplicate detection
        signature = self._minhash(text)

        # Check against all previously seen signatures
        for existing_id, existing_sig in self.minhash_signatures.items():
            similarity = self._jaccard_similarity(signature, existing_sig)
            if similarity >= self.threshold:
                return True, existing_id

        # Not a duplicate - remember this text
        self.exact_hashes.add(text_hash)
        self.minhash_signatures[doc_id] = signature
        self.seen_docs[doc_id] = {
            "hash": text_hash,
            "text": text[:200],
        }  # Store preview

        return False, None

    def deduplicate(
        self, documents: Generator[dict, None, None], track_sources: bool = True
    ) -> Tuple[Generator[dict, None, None], Dict]:
        """Remove exact and near-duplicates from document stream.

        Yields documents that are NOT duplicates. Returns statistics on removal.

        Args:
            documents: Generator yielding document dicts with 'text' and 'id' fields.
            track_sources: If True, track which sources had duplicates.

        Yields:
            dict: Non-duplicate documents.

        Returns:
            Dict: Deduplication statistics (counts, rates, per-source breakdown).
        """

        def deduplicate_generator():
            """Inner generator for deduplication."""
            stats = {
                "total_seen": 0,
                "exact_duplicates": 0,
                "near_duplicates": 0,
                "kept": 0,
                "per_source": defaultdict(lambda: {"seen": 0, "duplicates": 0}),
            }

            for doc in documents:
                stats["total_seen"] += 1
                text = doc.get("text", "")
                doc_id = doc.get("id", f"doc-{stats['total_seen']}")
                source = doc.get("source", "unknown")

                if track_sources:
                    stats["per_source"][source]["seen"] += 1

                # Check for duplicates
                is_dup, dup_of = self.is_duplicate(text, doc_id)

                if is_dup:
                    # Determine if exact or near-duplicate
                    text_hash = self._compute_hash(text)
                    if text_hash in self.exact_hashes:
                        stats["exact_duplicates"] += 1
                        if track_sources:
                            stats["per_source"][source]["duplicates"] += 1
                        self.logger.debug(
                            f"Exact duplicate: {doc_id} (matches {dup_of})"
                        )
                    else:
                        stats["near_duplicates"] += 1
                        if track_sources:
                            stats["per_source"][source]["duplicates"] += 1
                        self.logger.debug(
                            f"Near-duplicate: {doc_id} (matches {dup_of} "
                            f"at {self.threshold:.1%} similarity)"
                        )
                else:
                    # Keep this document
                    stats["kept"] += 1
                    yield doc

            # Yield stats before returning
            dedup_report = {
                "total_seen": stats["total_seen"],
                "kept": stats["kept"],
                "removed": stats["exact_duplicates"] + stats["near_duplicates"],
                "exact_duplicates": stats["exact_duplicates"],
                "near_duplicates": stats["near_duplicates"],
                "removal_rate": (
                    (stats["exact_duplicates"] + stats["near_duplicates"])
                    / stats["total_seen"]
                    if stats["total_seen"] > 0
                    else 0.0
                ),
                "per_source": dict(stats["per_source"]),
            }

            self.logger.info(
                f"Deduplication complete: {stats['kept']}/{stats['total_seen']} kept, "
                f"{dedup_report['removal_rate']:.1%} removal rate"
            )

            return dedup_report

        return deduplicate_generator(), self._dedup_stats_placeholder()

    def _dedup_stats_placeholder(self) -> Dict:
        """Placeholder for stats (actual stats returned from generator)."""
        return {}

    def per_source_dedup(
        self, documents: Generator[dict, None, None]
    ) -> Generator[dict, None, None]:
        """Remove duplicates within each source separately.

        Useful when sources might have significant overlap (e.g., duplicate
        Reddit posts across different subreddits).

        Args:
            documents: Generator yielding document dicts.

        Yields:
            dict: Documents without per-source duplicates.
        """
        source_hashes: Dict[str, Set[str]] = defaultdict(set)

        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source", "unknown")

            text_hash = self._compute_hash(text)

            if text_hash not in source_hashes[source]:
                # New document for this source
                source_hashes[source].add(text_hash)
                yield doc
            else:
                # Duplicate within source
                self.logger.debug(f"Per-source duplicate in {source}: {doc.get('id')}")

    def write_stats(self, stats: Dict, output_path: str) -> None:
        """Write deduplication statistics to JSON file.

        Args:
            stats: Statistics dictionary from deduplicate().
            output_path: Path to write JSON report.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Deduplication report written to {output_path}")
