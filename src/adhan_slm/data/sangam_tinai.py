"""Tiṇai canon loader: tiṇai → kalam, uripporul, karupporul.

Sangam poetics is not a flat label space. A poem carries at least four dimensions,
and a corpus that only records "kurinji" throws away most of the supervision:

  tiṇai      the five landscapes (kurinji, mullai, marutam, neytal, palai)
  kalam      the season the landscape implies (கூதிர், கார், வேனில் …)
  uripporul  the essential *action/emotion* (புணர்தல், இருத்தல், ஊடல், இரங்கல், பிரிதல்)
  karupporul the essential *substances/imagery* (deity, tree, flower, bird, beast, drum,
             water, people) that the poem's diction is built from

The data lives in ``sangam_tinai_canon.json`` next to this module, deliberately as
data rather than code so a Tamil scholar can correct it without touching Python.
Fields marked ``attested: false`` are the ones commentaries disagree about — the
loader returns them, but callers that need certainty can filter on ``attested``.

This module is intentionally dependency-free and import-safe (no corpus, no JAX).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

_CANON_PATH = Path(__file__).with_name("sangam_tinai_canon.json")

#: Canonical spelling -> the forms that actually appear in corpus metadata.
_ALIASES = {
    "kurinji": "kurinji",
    "kurinci": "kurinji",
    "குறிஞ்சி": "kurinji",
    "mullai": "mullai",
    "முல்லை": "mullai",
    "marutam": "marutam",
    "marudam": "marutam",
    "மருதம்": "marutam",
    "neytal": "neytal",
    "neydal": "neytal",
    "நெய்தல்": "neytal",
    "palai": "palai",
    "பாலை": "palai",
}


def canonical_tinai(value: str | None) -> Optional[str]:
    """Normalise a raw ``tinai`` value; ``None`` when it is unknown/absent."""
    if not value:
        return None
    key = value.strip().lower()
    return _ALIASES.get(key)


@dataclass(frozen=True)
class TinaiProfile:
    """Everything the canon says about one tiṇai."""

    name: str
    tamil: str
    landscape: str
    kalam: List[str]
    kalam_tamil: List[str]
    uripporul: str
    uripporul_tamil: str
    uripporul_gloss: str
    karupporul: Dict[str, Dict[str, object]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def karupporul_tamil(self, attested_only: bool = False) -> List[str]:
        """The imagery items, optionally restricted to well-attested ones."""
        out = []
        for _role, item in sorted(self.karupporul.items()):
            if attested_only and not item.get("attested", False):
                continue
            out.append(str(item.get("tamil", "")))
        return [t for t in out if t]

    def as_labels(self, attested_only: bool = False) -> Dict[str, object]:
        """Flat label dict, the shape the trainer/label sidecar wants."""
        return {
            "tinai": self.name,
            "tinai_tamil": self.tamil,
            "kalam": self.kalam[0] if self.kalam else None,
            "kalam_all": list(self.kalam),
            "uripporul": self.uripporul,
            "uripporul_tamil": self.uripporul_tamil,
            "karupporul": self.karupporul_tamil(attested_only=attested_only),
        }


@lru_cache(maxsize=1)
def load_canon() -> Dict[str, TinaiProfile]:
    """Load and index the canon (cached; the file is tiny and static)."""
    raw = json.loads(_CANON_PATH.read_text(encoding="utf-8"))
    out: Dict[str, TinaiProfile] = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        notes = [entry[k] for k in ("kalam_note",) if entry.get(k)]
        out[name] = TinaiProfile(
            name=name,
            tamil=entry.get("tamil", name),
            landscape=entry.get("landscape", ""),
            kalam=list(entry.get("kalam", [])),
            kalam_tamil=list(entry.get("kalam_tamil", [])),
            uripporul=entry.get("uripporul", ""),
            uripporul_tamil=entry.get("uripporul_tamil", ""),
            uripporul_gloss=entry.get("uripporul_gloss", ""),
            karupporul=dict(entry.get("karupporul", {})),
            notes=notes,
        )
    return out


def profile_for(value: str | None) -> Optional[TinaiProfile]:
    """Profile for a raw corpus value, or ``None`` if unknown."""
    name = canonical_tinai(value)
    return load_canon().get(name) if name else None


def labels_for(value: str | None, attested_only: bool = False) -> Dict[str, object]:
    """Label dict for a raw ``tinai`` value; safe on unknown input.

    Unknown tiṇai yields explicit nulls rather than a guessed label — a wrong
    landscape label is worse than a missing one for a conditioning signal.
    """
    prof = profile_for(value)
    if prof is None:
        return {
            "tinai": None,
            "tinai_tamil": None,
            "kalam": None,
            "kalam_all": [],
            "uripporul": None,
            "uripporul_tamil": None,
            "karupporul": [],
        }
    return prof.as_labels(attested_only=attested_only)


def quantise_kalam(profile: TinaiProfile, kalam: str | None) -> str:
    """Map a free-text season onto the canon's primary kalam for that tiṇai.

    Used when a corpus record carries a season that is attested-but-not-primary
    (e.g. palai poems set in முன்பனி): the *primary* label keeps the class
    balanced, and the raw value is preserved separately by the caller.
    """
    if not profile.kalam:
        return "unknown"
    if kalam:
        low = kalam.strip().lower()
        if low in profile.kalam:
            return low
        # Tamil season names in the canon table
        for tamil, slug in zip(profile.kalam_tamil, profile.kalam):
            if kalam and tamil in kalam:
                return slug
    return profile.kalam[0]
