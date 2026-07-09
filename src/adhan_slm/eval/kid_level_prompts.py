"""Kid-level Tamil eval prompt generation (ROADMAP_JAX_SLM.md §4: "Instruct
variant produces kid-level grammatical Tamil on a 50-prompt set").

Seeds the prompt set from open-tamil's `solthiruthi` categorized word lists
(animals, objects, adjectives, adverbs, relations, pronouns, verbs) — basic
everyday vocabulary that matches the "5-7 year old speaker" quality bar far
better than a classical/literary corpus (e.g. Thirukkural) would.

Requires open-tamil; see adhan_slm.external.open_tamil_bridge.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from adhan_slm.external.open_tamil_bridge import load_lexicon

# category -> (task_type, prompt template with {word})
_TEMPLATES = {
    "animals": ("define", "{word} என்றால் என்ன?"),
    "objects": ("define", "{word} என்றால் என்ன?"),
    "adjectives": ("define", "{word} என்றால் என்ன?"),
    "adverbs": ("define", "{word} என்றால் என்ன?"),
    "relations": ("define", "{word} என்றால் யார்?"),
    "pronouns": ("use_in_sentence", "'{word}' என்ற வார்த்தையை வைத்து ஒரு வாக்கியம் சொல்."),
    "verbs": ("paraphrase", "'{word}' என்பதை வேறு வார்த்தையில் சொல்."),
}


@dataclass
class KidLevelPrompt:
    category: str
    task_type: str
    word: str
    prompt: str


def build_kid_level_prompts(n: int = 50, seed: int = 0,
                             max_word_len: int = 12) -> List[KidLevelPrompt]:
    """Deterministically sample `n` kid-level prompts across all categories.

    `max_word_len` filters out long/compound lexicon entries (e.g. multi-word
    phrases like "படுக்கை அறை") that read as advanced vocabulary, not
    kid-level. Sampling is seeded so the eval set is stable across runs.
    """
    pool: List[KidLevelPrompt] = []
    for category, (task_type, template) in _TEMPLATES.items():
        for word in load_lexicon(category):
            if " " in word or len(word) > max_word_len:
                continue
            pool.append(KidLevelPrompt(
                category=category, task_type=task_type, word=word,
                prompt=template.format(word=word),
            ))

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def _demo() -> None:
    for p in build_kid_level_prompts(n=10):
        print(f"[{p.category}/{p.task_type}] {p.prompt}")


if __name__ == "__main__":
    _demo()
