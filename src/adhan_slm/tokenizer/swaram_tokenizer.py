"""சுவரம் டோக்கனைசர் (Swaram Tokenizer) — தமிழ் அக்சர மற்றும் மார்பீம் பகுப்பாய்வு.
Swaram tokenizer — native Tamil akshara + morpheme tokenization for Adhan SLM.

இரண்டு இழப்பற்ற அடுக்குகள் (Two Lossless Layers):
  அடுக்கு அ (Layer A): அக்சரப் பிரிப்பு (Akshara Segmentation) — மாறாநிலை, மூடிய கணம் (Deterministic, Closed Inventory)
  அடுக்கு ஆ (Layer B): மார்பீம் இணைப்புகள் (Morpheme BPE Merges) — அக்சரங்களின் மேல் கற்றறிந்த BPE அடுக்கு

அடிப்படை அலகு (Atomic Unit):
  தமிழ் எழுத்துக்கூட்டு / உயிர்மெய் (Akshara grapheme cluster).
  ஒரு மெய் எழுத்து தன் உயிர்மெய் குறியீட்டையோ (Matra) அல்லது புள்ளியையோ (Pulli/Virama) தன்னுடன் இணைத்துக் கொள்ளும்.
  இது தமிழின் சொல் இலக்கண அமைப்பை (Morphology & Sandhi) சிதைக்காமல் காக்கிறது.

CLI பயன்பாடு:
    python -m adhan_slm.tokenizer.swaram_tokenizer "படித்துக்கொண்டிருந்தேன்"
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)

# --- தமிழ் யுனிகோட் எல்லைகள் (Tamil Unicode Block U+0B80–U+0BFF) ----------------
# உயிர்மெய் குறியீடுகள் (Vowel Signs / Matras: ா ி ீ ு ூ ெ ே ை ொ ோ ௌ)
_TAMIL_MATRAS = set(range(0x0BBE, 0x0BCD))
# புள்ளி / மெய் எழுத்து வடிவம் (Virama / Pulli: ்)
_TAMIL_PULLI = 0x0BCD
# மெய்யோடு இணையும் சார்பு எழுத்துக் குறிகள் (Combining Marks)
_COMBINING = _TAMIL_MATRAS | {_TAMIL_PULLI}

# --- அடிப்படைத் தமிழ் எழுத்துக்களின் பட்டியல் (Closed Base Inventory) -----------
# 12 உயிர் எழுத்துக்கள் (12 Swaram / Pure Vowels)
UYIR = list("அஆஇஈஉஊஎஏஐஒஓஔ")
# 18 தமிழ் மெய் எழுத்துக்கள் + கிரந்த மெய்கள் (Consonants + Grantha Consonants)
_CONSONANTS = list("கஙசஜஞடணதநனபமயரறலளழவஶஷஸஹ")
# ஆய்த எழுத்து (Aytham)
AYTHAM = "ஃ"

# சிறப்பு குறியீடுகள் (Special Tokens for Neural Modeling)
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
# சொல் எல்லைக் குறிப்பான் (Word-Boundary Marker for Layer B BPE)
WORD_MARK = "▁"


def _is_combining(cp: int) -> bool:
    """கொடுக்கப்பட்ட யுனிகோட் குறியீடு மெய்யோடு இணையும் குறியா என சரிபார்த்தல்."""
    return cp in _COMBINING


def default_akshara_inventory() -> List[str]:
    """முழுமையான தமிழ் அக்சர கணம் (Complete Base Akshara Inventory).

    அடங்கியவை:
      - 12 உயிர் எழுத்துக்கள் (Pure Vowels)
      - 1 ஆய்த எழுத்து (Aytham)
      - 18 தூய மெய் எழுத்துக்கள் (Pure Consonants with Pulli: க், ங், ச்...)
      - 216 உயிர்மெய் எழுத்துக்கள் (Consonant-Vowel Ligatures: க, கா, கி, கீ...)
      - கிரந்த எழுத்துக்கள் (ஜ, ஷ, ஸ, ஹ, க்ஷ, ஸ்ரீ)

    எந்தவொரு தமிழ் சொல்லும் OOV (Out-of-Vocabulary) பிழையின்றி பிரிக்கப்படுவதை இது உறுதி செய்கிறது.
    """
    inv: List[str] = list(UYIR) + [AYTHAM]
    matras = [chr(cp) for cp in range(0x0BBE, 0x0BCD)]
    for c in _CONSONANTS:
        inv.append(c)  # அகர வரிசை உயிர்மெய் (Inherent 'a' consonant: க, ச, த...)
        inv.append(
            c + chr(_TAMIL_PULLI)
        )  # மெய் எழுத்து (Pure consonant with virama: க், ச், த்...)
        for m in matras:  # உயிர்மெய் சேர்க்கை (க + ா = கா, க + ி = கி...)
            inv.append(c + m)

    # நகல்களை நீக்கி வரிசையை முறைப்படுத்துதல் (Deduplicate while preserving order)
    seen, out = set(), []
    for a in inv:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def segment_aksharas(text: str) -> List[str]:
    """அடுக்கு அ: சொற்களை முழுமையான அக்சரங்களாகப் பிரித்தல் (Layer A: Akshara Segmentation).

    இழப்பற்ற பிரிப்பு முறை (Lossless): ''.join(out) == text.
    - ஒவ்வொரு தமிழ் எழுத்துத் தொடக்கத்திலும் புதிய அக்சரம் தொடங்கும்.
    - புள்ளி மற்றும் துணைக்கால்/உயிர்மெய்க் குறிகள் முந்தைய மெய் எழுத்துடன் இணையும்.
    - ஆங்கிலம், எண்கள் மற்றும் நிறுத்தற்குறிகள் தனித்தனி குறியீடுகளாகப் பாதுகாக்கப்படும்.
    """
    text = unicodedata.normalize("NFC", text)
    out: List[str] = []
    for ch in text:
        cp = ord(ch)
        if out and _is_combining(cp):
            # முந்தைய மெய் எழுத்துடன் உயிர்மெய்க் குறியையோ புள்ளியையோ இணைத்தல்
            out[-1] += ch
        else:
            out.append(ch)
    return out


@dataclass
class SwaramTokenizer:
    """சுவரம் டோக்கனைசர்: அக்சர அடிப்படை மற்றும் கற்றறிந்த மார்பீம் இணைப்பு அடுக்கு.
    Akshara-native tokenizer with an optional learned morpheme-merge layer (Layer B BPE).
    """

    vocab: Dict[str, int] = field(default_factory=dict)
    merges: List[Tuple[str, str]] = field(default_factory=list)
    _ranks: Dict[Tuple[str, str], int] = field(default_factory=dict, repr=False)
    _inv_vocab: Dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._reindex()

    # -- அகராதி வரிசையமைப்பு (Vocabulary Indexing) -------------------------------
    def _reindex(self):
        """இணைப்புத் தரவரிசை மற்றும் தலைகீழ் அகராதியை உருவாக்குதல்."""
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._inv_vocab = {i: t for t, i in self.vocab.items()}

    @property
    def unk_id(self) -> int:
        return self.vocab.get("<unk>", 0)

    def __len__(self) -> int:
        return len(self.vocab)

    # -- அக்சரப் பிரிப்பு முறை (Segmentation Hook) --------------------------------
    def _segment(self, text: str) -> List[str]:
        """அடுக்கு அ பிரிப்பு முறை. பிற இந்திய மொழிகளுக்கு இம்முறையை மாற்றி அமைக்கலாம்."""
        return segment_aksharas(text)

    @classmethod
    def _base_inventory(cls) -> List[str]:
        """அடிப்படை அக்சரப் பட்டியல்."""
        return default_akshara_inventory()

    def aksharas(self, text: str) -> List[str]:
        """அடுக்கு அ முடிவுகளை மட்டும் பெற (Layer A inspection/eval)."""
        return self._segment(text)

    # -- அடுக்கு ஆ: அக்சரங்களின் மேல் BPE மார்பீம் இணைப்பு (Layer B BPE over Aksharas)
    def _apply_merges(self, pieces: List[str]) -> List[str]:
        """அடிக்கடி வரும் அக்சரத் தொடர்களை ஒன்றிணைத்தல் (எ.கா: ப + டி + த் + து -> படித்து)."""
        if not self._ranks:
            return pieces
        pieces = list(pieces)
        while len(pieces) > 1:
            best_rank, best_i = None, -1
            for i in range(len(pieces) - 1):
                r = self._ranks.get((pieces[i], pieces[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            pieces[best_i : best_i + 2] = [pieces[best_i] + pieces[best_i + 1]]
        return pieces

    def _pretokenize(self, text: str) -> List[str]:
        """அக்சரங்களாகப் பிரித்து சொல் எல்லைக் குறியீடுகளை (WORD_MARK) சேர்த்தல்."""
        text = unicodedata.normalize("NFC", text)
        pieces: List[str] = []
        for i, cluster in enumerate(self._segment(text)):
            if cluster == " ":
                pieces.append(WORD_MARK)
            else:
                if i == 0 or (pieces and pieces[-1] == WORD_MARK) is False:
                    pass
                pieces.append(cluster)
        return pieces

    # -- குறியாக்கம் மற்றும் குறிமீட்பு (Encode / Decode) -------------------------
    def tokenize(self, text: str) -> List[str]:
        """உரையை டோக்கன் சரங்களின் பட்டியலாக மாற்றுதல் (Token strings)."""
        return self._apply_merges(self._pretokenize(text))

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """உரையை எண்சார் குறியீடுகளாக மாற்றுதல் (Token IDs)."""
        pieces = self.tokenize(text)
        ids = [self.vocab.get(p, self.unk_id) for p in pieces]
        if add_special:
            ids = [self.vocab.get("<bos>", self.unk_id), *ids, self.vocab.get("<eos>", self.unk_id)]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """எண்சார் குறியீடுகளை மீண்டும் தமிழ் உரையாக மாற்றுதல்."""
        specials = set(SPECIAL_TOKENS)
        toks = []
        for i in ids:
            t = self._inv_vocab.get(i, "<unk>")
            if skip_special and t in specials:
                continue
            toks.append(t)
        return "".join(toks).replace(WORD_MARK, " ")

    def fertility(self, text: str) -> float:
        """வளமை விகிதம் (Fertility Rate = Tokens / Akshara).

        இலக்கு: < 1.15 டோக்கன்கள்/அக்சரம்.
        மதிப்பு குறைவாக இருப்பது அதிகப்படியான மொழிச் சுருக்கத் திறனைக் குறிக்கிறது.
        """
        n_aks = sum(1 for a in self._segment(text) if a.strip())
        if n_aks == 0:
            return 0.0
        n_tok = sum(1 for t in self._apply_merges(self._pretokenize(text)) if t != WORD_MARK)
        return n_tok / n_aks

    # -- பயிற்சி முறை (Training BPE Merge Layer) ---------------------------------
    @classmethod
    def train(
        cls, corpus: List[str], vocab_size: int = 8000, min_freq: int = 2
    ) -> "SwaramTokenizer":
        """தமிழ் அக்சரங்களின் மேல் BPE மார்பீம் அடுக்கைப் பயிற்றுவித்தல்.

        - அடிப்படை அகராதி = சிறப்பு குறியீடுகள் + 247 அடிப்படை அக்சரங்கள்.
        - அதிக பயன்பாட்டில் உள்ள ஒட்டுச் சொற்கள் (-கள், -இல், -உடைய, -கொண்டு, -இருந்தேன்)
          தானாகவே கண்டறியப்பட்டு vocab_size வரை சேர்க்கப்படுகின்றன.
        """
        base = list(SPECIAL_TOKENS) + [WORD_MARK] + cls._base_inventory()
        seen = set(base)
        word_pieces: List[List[str]] = []
        for line in corpus:
            for tok in cls().__class__._pretokenize(cls(), line):
                if tok not in seen:
                    seen.add(tok)
                    base.append(tok)
            # சொல் பிரிப்புகளின் அடிப்படையில் மார்பீம்களைக் கணக்கிடுதல்
            pieces = cls()._pretokenize(line)
            word: List[str] = []
            for p in pieces:
                if p == WORD_MARK:
                    if word:
                        word_pieces.append(word)
                    word = []
                else:
                    word.append(p)
            if word:
                word_pieces.append(word)

        vocab = {t: i for i, t in enumerate(base)}
        merges: List[Tuple[str, str]] = []
        words = [list(w) for w in word_pieces]

        while len(vocab) < vocab_size:
            pairs: Counter = Counter()
            for w in words:
                for i in range(len(w) - 1):
                    pairs[(w[i], w[i + 1])] += 1
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            if freq < min_freq:
                break
            merged = a + b
            merges.append((a, b))
            if merged not in vocab:
                vocab[merged] = len(vocab)

            # பயிற்சி உரையில் புதிய இணைப்பைப் புதுப்பித்தல் (Update active sequences)
            new_words = []
            for w in words:
                new_w = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        new_w.append(merged)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                new_words.append(new_w)
            words = new_words

        return cls(vocab=vocab, merges=merges)

    # -- சேமித்தல் மற்றும் மீட்டெடுத்தல் (Save / Load Files) ----------------------
    def save(self, vocab_path: str, merges_path: str) -> None:
        """அகராதி மற்றும் இணைப்பு விதிகளை சேமித்தல்."""
        Path(vocab_path).write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        Path(merges_path).write_text(
            "\n".join(f"{a}\t{b}" for a, b in self.merges), encoding="utf-8"
        )

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str) -> "SwaramTokenizer":
        """சேமிக்கப்பட்ட கோப்புகளிலிருந்து டோக்கனைசரை ஏற்றுதல்."""
        vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        merges = []
        mtext = Path(merges_path).read_text(encoding="utf-8").strip()
        if mtext:
            for line in mtext.splitlines():
                if "\t" in line:
                    parts = line.split("\t", 1)
                    merges.append((parts[0], parts[1]))
        return cls(vocab=vocab, merges=merges)


def _demo(text: str) -> None:
    """சுவரம் டோக்கனைசரின் செயல்விளக்கம் (Demonstration Demo)."""
    aks = segment_aksharas(text)
    logger.info(f"உள்ளீடு (Input)      : {text}")
    logger.info(f"அக்சரங்கள் (Aksharas): {aks}")
    logger.info(f"அக்சர எண்ணிக்கை     : {sum(1 for a in aks if a.strip())}")

    tok = SwaramTokenizer.train([text], vocab_size=len(default_akshara_inventory()) + 64)
    ids = tok.encode(text, add_special=True)
    back = tok.decode(ids)
    logger.info(f"டோக்கன் எண்ணிக்கை   : {len(ids)} (with <bos>/<eos>)")
    logger.info(f"டோக்கன் குறியீடுகள் : {ids}")
    logger.info(f"மீட்கப்பட்ட உரை      : {back}")
    logger.info(f"வளமை விகிதம்         : {tok.fertility(text):.3f} tokens/akshara")


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "படித்துக்கொண்டிருந்தேன்"
    _demo(sample)
