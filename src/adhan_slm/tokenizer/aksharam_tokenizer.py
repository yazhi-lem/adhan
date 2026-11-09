"""अक्षरम् டோக்கனாக்கி (Aksharam tokenizer) — Devanagari akshara + morpheme tokenization.

Swaram (Dravidian) டோக்கனாக்கியின் இந்தோ-ஆரிய உடன்பிறப்புக் கட்டமைப்பு (the
Indo-Aryan sibling), sharing the same two lossless layers:

  அடுக்கு அ / परत अ (Layer A): தேவநாகரி அக்ஷரப் பிரிப்பு (Akshara Segmentation)
  அடுக்கு ஆ / परत आ (Layer B): வரம்புடைய BPE உருபன் இணைப்பு (Bounded Morpheme-Merge BPE)

The Devanagari Unicode facts (the whole U+0900–U+097F block) live in
:mod:`adhan_slm.common.scripts`; this module only names them and adds the BPE
layer. Rules the shared table encodes:

  - உயிர்க் குறிகள் / matras (vowel signs) attach to the preceding consonant.
  - The virama (halant ्) *does* build conjuncts here: C + ् + C is one akshara
    (क्ष, त्र, ज्ञ, प्र) — ``join_conjuncts`` is True for Devanagari, False for
    Tamil.
  - அனுஸ்வாரம் (ं), சந்திரபிந்து (ँ), விசர்க்கம் (ः) and the Vedic accents
    U+0951–U+0954 attach to the preceding letter; the nukta (़) is part of the
    letter it modifies (क़ किला).
  - The prishthamatra and vocalic signs (U+094E, U+094F, U+0955–U+0957) and the
    vocalic signs U+0962–U+0963 are vowel signs too.
  - ZWJ/ZWNJ (``क्‍ष`` half-form control) attach to the preceding akshara.
  - Devanagari digits (०–९) stand alone as their own aksharas.

CLI பயன்பாடு:
    python -m adhan_slm.tokenizer.aksharam_tokenizer "हम भारतीय हैं"
"""

from __future__ import annotations

import sys
from typing import List

from adhan_slm.common.scripts import (
    DEVANAGARI,
    akshara_inventory,
    segment,
)
from adhan_slm.core.logging import get_logger
from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer

logger = get_logger(__name__)

# --- Devanagari letters, straight from the validated registry ----------------
# 17 தனித்த உயிரெழுத்துக்கள் (svara, including ऄ ऌ ऍ ऎ ऑ ऒ)
VOWELS = DEVANAGARI.vowels()
# 45 மெய்யெழுத்துக்கள் (vyanjana): 37 classical + 8 precomposed nukta letters
# (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़)
_CONSONANTS = DEVANAGARI.consonants()
# 10 தேவநாகரி எண்கள் (अंक ०–९)
_DEVA_DIGITS = DEVANAGARI.digits()
# 20 உயிர்க் குறிகள் (vowel signs, including the prishthamatra and vocalic ones)
_DEVA_VOWEL_SIGNS = DEVANAGARI.vowel_signs()
# ஹலந்த் / விராமம் (virama / halant ्) — creates conjuncts in this script
_DEVA_VIRAMA = chr(DEVANAGARI.virama)


def _is_combining(cp: int) -> bool:
    """True if this codepoint attaches to the preceding unit (matra, halant, sign…)."""
    return DEVANAGARI.is_combining(cp)


def default_aksharam_inventory() -> List[str]:
    """தேவநாகரி மூடிய அக்ஷர கணம் (Closed Devanagari base set).

    Counts are computed from the registry, never written by hand:
      - every independent vowel the block defines (17: अ … औ, with ऄ ऌ ऍ ऎ ऑ ऒ)
      - the 10 native digits (०–९)
      - 45 consonants × (inherent अ + halant form + 20 vowel signs)
      - deduplicated, order-stable
    """
    return akshara_inventory(DEVANAGARI)


def segment_devanagari(text: str) -> List[str]:
    """அடுக்கு அ: இழப்பற்ற தேவநாகரி அக்ஷரப் பிரிப்பு (Layer A Devanagari segmentation).

    இழப்பற்ற பிரிப்பு (Lossless): ''.join(out) == NFC(text).
    - உயிர்க் குறிகள், நுக்தா, அனுஸ்வாரம், விசர்க்கம், வேத ஒலிக்குறிகள் முந்தைய
      எழுத்துடன் இணையும்.
    - ஹலந்திற்குப் பின் வரும் மெய்யெழுத்து கூட்டெழுத்தாக (क्ष, त्र) இணையும்.
    - ZWJ/ZWNJ முந்தைய அக்ஷரத்துடன் இணையும்.

    செயலாக்கம் :func:`adhan_slm.common.scripts.segment` இல் ஒன்றே ஒன்று.
    """
    return segment(text, (DEVANAGARI,))


class AksharamTokenizer(SwaramTokenizer):
    """இந்தி மற்றும் தேவநாகரி எழுத்துக்களுக்கான அக்ஷர டோக்கனாக்கி.
    Hindi/Devanagari akshara tokenizer (Indic-script prototype).
    """

    def _segment(self, text: str) -> List[str]:
        """தேவநாகரி அக்ஷரப் பிரிப்புக் கொக்கி (Segmentation hook)."""
        return segment_devanagari(text)

    @classmethod
    def _base_inventory(cls) -> List[str]:
        """தேவநாகரி அடிப்படை அக்ஷரப் பட்டியல் (Base inventory hook)."""
        return default_aksharam_inventory()


def _demo(text: str) -> None:
    """அக்ஷரம் டோக்கனாக்கியின் செயல்விளக்கம் (Demonstration demo)."""
    aks = segment_devanagari(text)
    logger.info(f"इनपुट / உள்ளீடு (Input)         : {text}")
    logger.info(f"अक्षर / அக்ஷரங்கள் (Aksharas)   : {aks}")
    logger.info(f"अक्षर संख्या / அக்ஷர எண்ணிக்கை : {len(aks)}")

    tok = AksharamTokenizer.train([text], vocab_size=len(default_aksharam_inventory()) + 32)
    ids = tok.encode(text, add_special=True)
    back = tok.decode(ids)
    logger.info(f"टोकन संख्या / டோக்கன் எண்ணிக்கை : {len(ids)} (with <bos>/<eos>)")
    logger.info(f"टोकन आईडी / டோக்கன் குறியீடுகள் : {ids}")
    logger.info(f"पुनर्प्राप्त / மீட்கப்பட்ட உரை : {back}")
    logger.info(f"फर्टिलिटी / வளமை விகிதம்      : {tok.fertility(text):.3f}")


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "पढ़ रहा था"
    _demo(sample)
