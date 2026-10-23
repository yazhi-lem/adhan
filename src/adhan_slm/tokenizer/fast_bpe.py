"""Rust-backed akshara BPE for the Swaram / Aksharam tokenizer family.

Why this exists
---------------
``swaram_tokenizer.SwaramTokenizer`` is a reference implementation: segmentation
and merges are pure Python. Encoding measured ~7 kchar/s, i.e. **43 hours** for
the 1.06G-character Tamil archive. Tokenizer quality is not the problem; the
merge loop is.

``tokenizers`` (Rust) cannot express "BPE over grapheme clusters" directly: its
BPE starts from single characters, so a merge table over multi-codepoint
aksharas never fires. The workaround used here — and the reason this file is
subtle — is to prepend *composition merges* that build every akshara from its
codepoints at the highest priority (ranks 0..K), followed by the learned
akshara-level merges. The Rust encoder therefore forms aksharas first, then
morphology, which is exactly the two-layer contract of the Python tokenizer.

Training is done in a private-use "PUA space" where each akshara is a single
codepoint, so ``BpeTrainer`` learns merges over aksharas rather than over
codepoints; the learned vocab/merges are then translated back to real strings.

Byte fallback
-------------
``byte_fallback=True`` adds the 256 ``<0xNN>`` tokens, so *any* character the
vocabulary has never seen — any Indic script, Latin, digits, emoji — round-trips
through its UTF-8 bytes instead of collapsing to ``<unk>``. This is what makes
the tokenizer lossless on real scraped text (the 12k reference vocab had no byte
tokens and lost 8.5% of web documents to ``<unk>``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from adhan_slm.core.logging import get_logger
from adhan_slm.tokenizer import indic_akshara, morphology
from adhan_slm.tokenizer.swaram_tokenizer import (
    BYTE_TOKENS,
    SPECIAL_TOKENS,
    WORD_MARK,
    SwaramTokenizer,
)

logger = get_logger(__name__)

# Private Use Area base for the "one akshara == one codepoint" training view.
PUA_BASE = 0xE000
_MAX_PUA = 0xF8FF - PUA_BASE + 1

_BYTE_VALUES = {t: i for i, t in enumerate(BYTE_TOKENS)}


def _composition_merges(inventory: Sequence[str]) -> List[Tuple[str, str]]:
    """Merges that assemble each akshara from its codepoints, left to right.

    ``கொ`` (ka + o-matra) becomes ``("க","ொ") -> கொ``; a Devanagari conjunct such
    as ``क्ष`` chains through ``("क","्") -> क्`` then ``("क्","ष") -> क्ष``.
    Every intermediate symbol is itself an inventory akshara, so this adds no
    vocabulary — only merge ranks.
    """
    out: List[Tuple[str, str]] = []
    seen = set()
    for aks in inventory:
        prefix = aks[0]
        for ch in aks[1:]:
            pair = (prefix, ch)
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
            prefix += ch
    return out


class AksharaBPE:
    """Akshara-native BPE that encodes in Rust and stays byte-lossless."""

    def __init__(
        self,
        tokenizer,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        inventory: Sequence[str],
        script: indic_akshara.Script,
        byte_fallback: bool = True,
    ) -> None:
        self._tok = tokenizer
        self.vocab = vocab
        self.merges = merges
        self.inventory = list(inventory)
        self.script = script
        self.byte_fallback = byte_fallback
        self._inv_vocab = {i: t for t, i in vocab.items()}
        self._unk = vocab.get("<unk>", 0)

    # -- introspection ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def unk_id(self) -> int:
        return self._unk

    def tokenize(self, text: str) -> List[str]:
        return self._tok.encode(text, add_special_tokens=False).tokens

    def tokenize_spans(self, text: str) -> List[morphology.TokenSpan]:
        """Tokens with the akshara range each covers (see ``morphology.TokenSpan``).

        Layer A is enforced by the composition merges, so every emitted token is
        a whole number of aksharas and the ranges are exact.
        """
        aksharas = [WORD_MARK if a == " " else a for a in self._segment(text)]
        return morphology.spans_from_pieces(self.tokenize(text), aksharas)

    def morphology(self, words: Sequence[str]) -> morphology.BoundaryReport:
        """Dependency-free stem/suffix boundary agreement (see morphology module)."""
        return morphology.boundary_agreement(self, words)

    def sandhi_points(self, text: str) -> List[int]:
        """Akshara indices in ``text`` that look like an internal sandhi join."""
        return morphology.sandhi_points(text, lambda t: self._segment(t))

    # -- encode / decode -------------------------------------------------------
    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_special:
            bos = self.vocab.get("<bos>", self._unk)
            eos = self.vocab.get("<eos>", self._unk)
            ids = [bos, *ids, eos]
        return ids

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Iterable[int], skip_special: bool = True, hide_unk: bool = False) -> str:
        """Mirror of ``SwaramTokenizer.decode`` (byte runs, specials, word mark)."""
        structural = set(SPECIAL_TOKENS) - {"<unk>"}
        toks: List[str] = []
        pending = bytearray()
        for i in ids:
            t = self._inv_vocab.get(i, "<unk>")
            b = _BYTE_VALUES.get(t)
            if b is not None:
                pending.append(b)
                continue
            if pending:
                toks.append(pending.decode("utf-8", errors="replace"))
                pending.clear()
            if skip_special and t in structural:
                continue
            if t == "<unk>" and hide_unk:
                continue
            toks.append(t)
        if pending:
            toks.append(pending.decode("utf-8", errors="replace"))
        return "".join(toks).replace(WORD_MARK, " ")

    def fertility(self, text: str) -> float:
        n_aks = sum(1 for a in self._segment(text) if a.strip())
        if n_aks == 0:
            return 0.0
        n_tok = sum(1 for t in self.tokenize(text) if t != WORD_MARK)
        return n_tok / n_aks

    def _segment(self, text: str) -> List[str]:
        return indic_akshara.segment(text, [self.script])

    # -- training --------------------------------------------------------------
    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        inventory: Sequence[str],
        vocab_size: int = 32000,
        script: indic_akshara.Script = indic_akshara.TAMIL,
        byte_fallback: bool = True,
        min_frequency: int = 2,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        show_progress: bool = False,
    ) -> "AksharaBPE":
        """Learn akshara merges in PUA space, then rebuild a real-string tokenizer."""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        inv = list(inventory)
        if len(inv) > _MAX_PUA:
            raise ValueError(f"inventory too large for PUA mapping ({len(inv)} > {_MAX_PUA})")
        pua = {aks: chr(PUA_BASE + i) for i, aks in enumerate(inv)}
        pua_unmap = {v: k for k, v in pua.items()}

        # Longest-first alternation so "க்" wins over "க" at the same position.
        pattern = re.compile("|".join(re.escape(a) for a in sorted(inv, key=len, reverse=True)))

        def to_pua(text: str) -> str:
            text = unicodedata.normalize("NFC", text).replace(" ", WORD_MARK)
            return pattern.sub(lambda m: pua.get(m.group(), m.group()), text)

        def iterator() -> Iterator[str]:
            for line in corpus:
                if line:
                    yield to_pua(line)

        alphabet = list(pua.values()) + [WORD_MARK] + indic_akshara.script_alphabet(script)
        specs = list(special_tokens) + (list(BYTE_TOKENS) if byte_fallback else [])

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Split(pattern=WORD_MARK, behavior="isolated")
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=specs,
            initial_alphabet=alphabet,
            show_progress=show_progress,
        )
        tok.train_from_iterator(iterator(), trainer)

        raw_vocab = tok.get_vocab()
        raw_merges = [tuple(m) for m in json.loads(tok.to_str())["model"]["merges"]]

        # Translate PUA space back to real strings, dropping any collision.
        real_vocab: Dict[str, int] = {}
        collisions = 0
        for token, tid in sorted(raw_vocab.items(), key=lambda kv: kv[1]):
            real = "".join(pua_unmap.get(ch, ch) for ch in token)
            if real in real_vocab:
                collisions += 1
                continue
            real_vocab[real] = len(real_vocab)
        if collisions:
            logger.warning("dropped %d colliding tokens while leaving PUA space", collisions)

        learned: List[Tuple[str, str]] = []
        seen = set()
        for a, b in raw_merges:
            ra = "".join(pua_unmap.get(ch, ch) for ch in a)
            rb = "".join(pua_unmap.get(ch, ch) for ch in b)
            if ra not in real_vocab or rb not in real_vocab:
                continue
            if (ra, rb) in seen:
                continue
            seen.add((ra, rb))
            learned.append((ra, rb))

        return cls._build(
            real_vocab,
            learned,
            inv,
            script=script,
            byte_fallback=byte_fallback,
            special_tokens=special_tokens,
        )

    def prune_morphology(self) -> "AksharaBPE":
        """Return a copy whose merge table does not cut through known suffixes.

        Compression and morphology pull against each other: a bigger vocabulary
        merges *through* stem/suffix breaks more often. This drops exactly the
        merges that do that (see ``morphology.boundary_crossing_merges``), which
        costs a little fertility and buys boundary agreement.
        """
        pruned = morphology.prune_boundary_crossing_merges(self.merges)
        comp = set(_composition_merges(self.inventory))
        learned = [m for m in pruned if m not in comp]
        return self._build(
            dict(self.vocab),
            learned,
            self.inventory,
            script=self.script,
            byte_fallback=self.byte_fallback,
        )

    @classmethod
    def from_python(
        cls,
        py_tok: SwaramTokenizer,
        inventory: Sequence[str],
        script: indic_akshara.Script = indic_akshara.TAMIL,
        byte_fallback: bool | None = None,
    ) -> "AksharaBPE":
        """Wrap an already-trained Python tokenizer's vocab/merges in the Rust encoder."""
        if byte_fallback is None:
            byte_fallback = all(t in py_tok.vocab for t in BYTE_TOKENS)
        merges = [tuple(m) for m in py_tok.merges if m[0] and m[1]]
        return cls._build(
            dict(py_tok.vocab),
            merges,
            inventory,
            script=script,
            byte_fallback=byte_fallback,
        )

    @classmethod
    def _build(
        cls,
        vocab: Dict[str, int],
        learned_merges: List[Tuple[str, str]],
        inventory: Sequence[str],
        script: indic_akshara.Script,
        byte_fallback: bool,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> "AksharaBPE":
        """Assemble the runtime tokenizer: composition merges, then learned merges."""
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

        vocab = dict(vocab)
        # Every codepoint of the inventory must exist as its own token or the
        # composition merges cannot fire.
        for aks in inventory:
            for ch in aks:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
        for t in list(special_tokens) + ([WORD_MARK] if WORD_MARK not in vocab else []):
            if t not in vocab:
                vocab[t] = len(vocab)
        if byte_fallback:
            for t in BYTE_TOKENS:
                if t not in vocab:
                    vocab[t] = len(vocab)

        comp = _composition_merges(inventory)
        merges = comp + [m for m in learned_merges if m not in set(comp)]
        # Drop merges whose members are not in the (possibly re-ided) vocab.
        merges = [(a, b) for a, b in merges if a in vocab and b in vocab]

        tok = Tokenizer(
            models.BPE(
                vocab=vocab,
                merges=merges,
                unk_token="<unk>",
                byte_fallback=byte_fallback,
            )
        )
        tok.normalizer = normalizers.Replace(" ", WORD_MARK)
        tok.pre_tokenizer = pre_tokenizers.Split(pattern=WORD_MARK, behavior="isolated")
        tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Replace(WORD_MARK, " ")])
        return cls(tok, vocab, merges, inventory, script, byte_fallback)

    # -- persistence -----------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        """Write tokenizer.json plus vocab.json / merges.txt for the Python path."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(out / "tokenizer.json"))
        (out / "vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "merges.txt").write_text(
            "\n".join(f"{a}\t{b}" for a, b in self.merges if a and b), encoding="utf-8"
        )
        return out / "tokenizer.json"

    @classmethod
    def load(
        cls,
        tokenizer_dir: str | Path,
        inventory: Sequence[str] | None = None,
        script: indic_akshara.Script = indic_akshara.TAMIL,
    ) -> "AksharaBPE":
        from tokenizers import Tokenizer

        d = Path(tokenizer_dir)
        path = d / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"no tokenizer.json in {d}")
        tok = Tokenizer.from_file(str(path))
        vocab = tok.get_vocab()
        merges = [tuple(m) for m in json.loads(tok.to_str())["model"]["merges"]]
        if inventory is None:
            inventory = [a for a in vocab if a and a != WORD_MARK and not a.startswith("<")]
        byte_fallback = all(t in vocab for t in BYTE_TOKENS)
        return cls(tok, vocab, merges, inventory, script, byte_fallback)

    def to_python_tokenizer(self) -> SwaramTokenizer:
        """Reference (pure-Python) tokenizer sharing this vocab and merge table.

        Used to prove the Rust path reproduces the Python one; the Python object
        keeps only the *learned* merges (composition is implicit there).
        """
        comp = set(_composition_merges(self.inventory))
        learned = [m for m in self.merges if m not in comp]
        return SwaramTokenizer(vocab=dict(self.vocab), merges=learned)
