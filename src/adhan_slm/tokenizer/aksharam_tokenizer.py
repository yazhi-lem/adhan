"""अक्षरम् டோக்கனைசர் / अक्षरम् टोकनाइज़र (Aksharam Tokenizer) — Indic Devanagari Akshara & Morpheme Tokenization.

Swaram (திராவிட) டோக்கனைசரின் இந்தோ-ஆரிய உடன்பிறப்புக் கட்டமைப்பு (Indo-Aryan Sibling).
அதே இரு இழப்பற்ற அடுக்குகள் (Same Two-Layer Lossless Architecture):
  அடுக்கு அ (Layer A): தேவநாகரி அக்சரப் பிரிப்பு (Devanagari Akshara Segmentation)
  அடுக்கு ஆ (Layer B): வரம்புடைய BPE மார்பீம் இணைப்பு (Bounded Morpheme-Merge BPE)

தேவநாகரி யுனிகோட் இலக்கண விதிகள் (Devanagari Unicode Phonological Rules):
  - மாத்திரைகள் (Matras / Vowel Signs U+093E–U+094C) முந்தைய மெய்யோடு இணையும்.
  - ஹலந்த் / விராமம் (Virama/Halant U+094D) கூட்டெழுத்துக்களை உருவாக்கும்: C + ् + C → ஒரே அக்சரம் (எ.கா: क्ष, त्र, ज्ञ, प्र).
  - அனுஸ்வாரம் (ं), விசர்க்கம் (ः), சந்திரபிந்து (ँ), நுக்தா (़) அடிப்படை எழுத்தோடு இணையும்.
  - தேவநாகரி எண்கள் (०–९) தனி அக்சரங்களாகக் கையாளப்படும்.

CLI பயன்பாடு:
    python -m adhan_slm.tokenizer.aksharam_tokenizer "हम भारतीय हैं"
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer  # noqa: E402

logger = get_logger(__name__)

# --- தேவநாகரி யுனிகோட் எல்லைகள் (Devanagari Block U+0900–U+097F) ---------------
# உயிர்மெய்க் குறிகள் / மாத்திரைகள் (Vowel Signs / Matras: ा ि ी ु ू ृ ॄ ॅ ॆ े ै ॉ ॊ ो ौ)
_DEVA_MATRAS = set(range(0x093E, 0x094D))
# ஹலந்த் / புள்ளி / விராமம் (Virama / Halant ् — கூட்டெழுத்துக்களை உருவாக்கும்)
_DEVA_VIRAMA = 0x094D
# நுக்தா புள்ளி (Nukta ़ — கடன்/ஒலிப்பு மாறுபாடுகள்: क़, ख़, ग़, ज़, ड़, ढ़, फ़)
_DEVA_NUKTA = 0x093C
# ஒலிக்குறிப்புகள் (Chandrabindu ँ, Anusvara ं, Visarga ः, Short Signs)
_DEVA_SIGNS = {0x0900, 0x0901, 0x0902, 0x0903}
# மெய்யோடு இணையும் சார்புக் குறிகளின் முழுமையான கணம்
_COMBINING = _DEVA_MATRAS | {_DEVA_VIRAMA, _DEVA_NUKTA} | _DEVA_SIGNS

# தனித்த உயிரெழுத்துக்கள் (Independent Vowels / Swara: 11 स्वर)
VOWELS = list("अआइईउऊऋएऐओऔ")
# அடிப்படை மெய்யெழுத்துக்கள் (Base Consonants / Vyanjana: 33 व्यंजन)
_CONSONANTS = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
# தேவநாகரி எண்கள் (Devanagari Digits: 0-9)
_DEVA_DIGITS = list("०१२३४५६७८९")


def _is_combining(cp: int) -> bool:
    """கொடுக்கப்பட்ட யுனிகோட் புள்ளி தேவநாகரி சார்புக் குறியா என சரிபார்த்தல்."""
    return cp in _COMBINING


def default_aksharam_inventory() -> List[str]:
    """தேவநாகரி மூடிய அக்சர கணம் (Closed Devanagari Base Set).

    அடங்கியவை:
      - 11 தனித்த உயிரெழுத்துக்கள் (स्वर)
      - 10 தேவநாகரி எண்கள் (अंक ०-९)
      - 33 அகர மெய்கள் (व्यंजन)
      - 33 ஹலந்த் பெற்ற அரை மெய்கள் (Half-consonants / Virama: क्, ख्, ग्...)
      - 495 உயிர்மெய் சேர்க்கைகள் (Consonant + Matra ligatures)
    """
    inv: List[str] = list(VOWELS) + list(_DEVA_DIGITS)
    matras = [chr(cp) for cp in range(0x093E, 0x094D)]
    for c in _CONSONANTS:
        inv.append(c)  # அகர உயிர்மெய் (Inherent 'a')
        inv.append(c + chr(_DEVA_VIRAMA))  # அரை மெய் / ஹலந்த் (Half consonant: क्)
        for m in matras:  # உயிர்மெய் (க + ா = का, कि, की...)
            inv.append(c + m)
    seen, out = set(), []
    for a in inv:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def segment_devanagari(text: str) -> List[str]:
    """அடுக்கு அ: இழப்பற்ற தேவநாகரி அக்சரப் பிரிப்பு (Layer A Devanagari Segmentation).

    இழப்பற்ற பிரிப்பு (Lossless): ''.join(out) == text.
    - மாத்திரைகள், நுக்தா, அனுஸ்வாரம், விசர்க்கம் முந்தைய மெய்யுடன் இணையும்.
    - ஹலந்த் (விராமம்) வரும்போது, அதைத் தொடர்ந்து வரும் மெய்யெழுத்தும் அதே அக்சரக் கூட்டில்
      இணைந்து கூட்டெழுத்தை (Samyuktakshar / Conjunct: क् + ष → क्ष, त् + र → त्र) உருவாக்கும்.
    """
    text = unicodedata.normalize("NFC", text)
    out: List[str] = []
    prev_was_virama = False
    for ch in text:
        cp = ord(ch)
        if out and _is_combining(cp):
            out[-1] += ch
            prev_was_virama = cp == _DEVA_VIRAMA
        elif out and prev_was_virama and 0x0915 <= cp <= 0x0939:
            # ஹலந்திற்குப் பின் வரும் மெய் கூட்டெழுத்தாக முந்தைய அக்சரத்தோடு இணைகிறது
            out[-1] += ch
            prev_was_virama = False
        else:
            out.append(ch)
            prev_was_virama = False
    return out


class AksharamTokenizer(SwaramTokenizer):
    """இந்தி மற்றும் தேவநாகரி எழுத்துக்களுக்கான அக்சர டோக்கனைசர்.
    Hindi/Devanagari akshara tokenizer (Indic-script prototype).
    """

    def _segment(self, text: str) -> List[str]:
        """தேவநாகரி அக்சரப் பிரிப்பு கொக்கி (Segmentation hook)."""
        return segment_devanagari(text)

    @classmethod
    def _base_inventory(cls) -> List[str]:
        """தேவநாகரி அடிப்படை அக்சரப் பட்டியல் (Base inventory hook)."""
        return default_aksharam_inventory()


def _demo(text: str) -> None:
    """அக்சரம் டோக்கனைசரின் செயல்விளக்கம் (Demonstration Demo)."""
    aks = segment_devanagari(text)
    logger.info(f"इनपुट / உள்ளீடு (Input)      : {text}")
    logger.info(f"अक्षर / அக்சரங்கள் (Aksharas): {aks}")
    logger.info(f"अक्षर संख्या / அக்சர எண்ணிக்கை: {len(aks)}")

    tok = AksharamTokenizer.train(
        [text], vocab_size=len(default_aksharam_inventory()) + 32
    )
    ids = tok.encode(text, add_special=True)
    back = tok.decode(ids)
    logger.info(f"टोकन संख्या / டோக்கன் எண்ணிக்கை : {len(ids)} (with <bos>/<eos>)")
    logger.info(f"टोकन आईडी / டோக்கன் குறியீடுகள் : {ids}")
    logger.info(f"पुनर्प्राप्त / மீட்கப்பட்ட உரை  : {back}")
    logger.info(f"फर्टिलिटी / வளமை விகிதம்      : {tok.fertility(text):.3f}")


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "पढ़ रहा था"
    _demo(sample)
