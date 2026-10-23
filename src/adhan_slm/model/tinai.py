"""Tiṇai (திணை) — the five Sangam landscapes used to condition the Adhan SLM.

Classical Tamil poetics (Tolkāppiyam, *Poruḷatikāram*) organises a poem's setting
around five landscapes. Each one fixes a *karupporul* (the natural setting: land,
flora, fauna, season) and an *uripporul* (the human feeling that belongs to it):

    kurinji   குறிஞ்சி   mountains          punartal   புணர்தல்   union
    mullai    முல்லை     forest / pastoral   iruttal    இருத்தல்   patient waiting
    marutham  மருதம்     farmland            ootal      ஊடல்       lovers' quarrel
    neithal   நெய்தல்    seashore            irankal    இரங்கல்    anxious pining
    paalai    பாலை       arid wasteland      pirital    பிரிதல்    separation

This module is pure Python (no JAX): the names, a conservative keyword lexicon and a
weak labeler. The conditioning layer itself (``TinaiRouter`` behaviour inside
``AdhanSLM``) lives in ``transformer.py``; this module supplies its vocabulary and
a cheap way to get *weak* labels from raw text.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, Tuple

#: Label meaning "no tiṇai specified / could not tell" (used for ignore-in-loss and
#: for letting the router decide on its own).
UNSPECIFIED = -1


@dataclass(frozen=True)
class Tinai:
    key: str  # ASCII id, stable across versions
    tamil: str  # name in Tamil script
    landscape: str  # karupporul: the setting
    uripporul: str  # the feeling that belongs to it


#: Fixed order — a tiṇai's position here is its class index in the model.
TINAI: Tuple[Tinai, ...] = (
    Tinai("kurinji", "குறிஞ்சி", "mountains", "punartal (union)"),
    Tinai("mullai", "முல்லை", "forest / pastoral land", "iruttal (patient waiting)"),
    Tinai("marutham", "மருதம்", "farmland", "ootal (lovers' quarrel)"),
    Tinai("neithal", "நெய்தல்", "seashore", "irankal (anxious pining)"),
    Tinai("paalai", "பாலை", "arid wasteland", "pirital (separation)"),
)
N_TINAI = len(TINAI)


def tinai_index(name: str) -> int:
    """Class index for an ASCII key (``"mullai"``) or a Tamil name (``"முல்லை"``).

    Raises ``KeyError`` for anything else, so a typo never silently becomes a label.
    """
    needle = unicodedata.normalize("NFC", name.strip()).lower()
    for i, t in enumerate(TINAI):
        if needle in (t.key, unicodedata.normalize("NFC", t.tamil)):
            return i
    raise KeyError(f"unknown tiṇai {name!r}; expected one of {[t.key for t in TINAI]}")


#: Deliberately conservative: only words that point at one landscape and are rare
#: elsewhere. Matching is by substring on NFC text, which suits Tamil's agglutination
#: (``மலையில்`` "on the hill" contains ``மலை``). The tiṇai names themselves are the
#: strongest cue. Extend with care — a wrong keyword teaches the router a wrong rule.
TINAI_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "kurinji": ("குறிஞ்சி", "மலை", "குன்று", "அருவி", "வேங்கை"),
    "mullai": ("முல்லை", "காடு", "கார்காலம்", "ஆயர்", "கொன்றை"),
    "marutham": ("மருதம்", "வயல்", "நெல்", "உழவர்", "தாமரை", "ஊடல்"),
    "neithal": ("நெய்தல்", "கடல்", "கடற்கரை", "பரதவர்", "கானல்"),
    "paalai": ("பாலை", "வெயில்", "வறண்ட", "கள்ளி", "வேனில்"),
}


def label_text(text: str) -> int:
    """Weak tiṇai label for ``text``: the class index, or ``UNSPECIFIED``.

    A label is returned only when exactly one tiṇai has the most keyword hits and that
    count is at least one. Ties and keyword-free text give ``UNSPECIFIED`` — never a
    guess. These are *weak* labels for supervising the router, not ground truth.
    """
    norm = unicodedata.normalize("NFC", text)
    scores = [
        sum(norm.count(unicodedata.normalize("NFC", kw)) for kw in TINAI_KEYWORDS[t.key])
        for t in TINAI
    ]
    best = max(scores)
    if best == 0 or scores.count(best) != 1:
        return UNSPECIFIED
    return scores.index(best)
