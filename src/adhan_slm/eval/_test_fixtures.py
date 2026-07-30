"""Shared Tamil fixtures for the eval-probe test modules.

Leading underscore so pytest never collects this as a test module. The eval probes
all depend on **open-tamil**, which is optional (see `requirements-jax.txt`), so this
module also exposes the single skip guard they share — `requires_open_tamil`.
"""

from __future__ import annotations

from typing import List

from adhan_slm.core.selftest import skip_unless
from adhan_slm.external.open_tamil_bridge import HAS_OPEN_TAMIL

#: Heavily inflected verbs — the agglutination the morphology probe is built for.
INFLECTED_WORDS: List[str] = [
    "படித்துக்கொண்டிருந்தேன்",
    "எழுதினேன்",
    "பேசினோம்",
    "வந்தார்கள்",
    "சொன்னாள்",
    "ஓடுகிறான்",
    "படிக்கிறாள்",
    "வருவேன்",
    "செய்தார்",
    "நடந்தது",
    "பார்த்தேன்",
    "கேட்டான்",
]

#: Already-correct phrases: the sandhi probe must find nothing to fix in these.
GRAMMATICAL_PHRASES: List[str] = [
    "மரம் வளர்ந்தது",
    "நான் பள்ளிக்கு போனேன்",
    "அவன் நன்றாக பேசுகிறான்",
]

#: A miniature corpus for the n-gram perplexity floor.
CORPUS: List[str] = [
    "தமிழ் ஒரு அழகான மொழி",
    "நான் தமிழில் படிக்கிறேன்",
    "இது ஒரு நல்ல நாள்",
    "அவன் பள்ளிக்கு சென்றான்",
    "நாய் ஓடுகிறது",
    "பூனை பாலைக் குடிக்கிறது",
]

#: Decorator: skip (not silently pass) when open-tamil is absent.
requires_open_tamil = skip_unless(HAS_OPEN_TAMIL, "open-tamil not installed")
