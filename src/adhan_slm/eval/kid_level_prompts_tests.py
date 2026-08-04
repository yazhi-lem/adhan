"""Tests for the kid-level eval prompt set (`adhan_slm.eval.kid_level_prompts`).

Roadmap §4 definition of done: "produces kid-level grammatical Tamil on a 50-prompt
set". These check the prompt set itself is well-formed and reproducible.

Run: PYTHONPATH=src python -m adhan_slm.eval.kid_level_prompts_tests
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.eval._test_fixtures import requires_open_tamil  # noqa: E402


@requires_open_tamil
def test_builds_the_requested_number_of_prompts() -> None:
    from adhan_slm.eval.kid_level_prompts import build_kid_level_prompts

    assert len(build_kid_level_prompts(n=50, seed=0)) == 50


@requires_open_tamil
def test_seed_words_are_distinct() -> None:
    """A repeated seed word would silently shrink the effective prompt set."""
    from adhan_slm.eval.kid_level_prompts import build_kid_level_prompts

    prompts = build_kid_level_prompts(n=50, seed=0)
    assert len({p.word for p in prompts}) == 50, "expected 50 distinct seed words"


@requires_open_tamil
def test_every_prompt_contains_its_seed_word() -> None:
    from adhan_slm.eval.kid_level_prompts import build_kid_level_prompts

    for p in build_kid_level_prompts(n=50, seed=0):
        assert p.word in p.prompt


@requires_open_tamil
def test_same_seed_gives_the_same_prompts() -> None:
    """Eval runs are only comparable across checkpoints if the prompts are fixed."""
    from adhan_slm.eval.kid_level_prompts import build_kid_level_prompts

    a = build_kid_level_prompts(n=20, seed=42)
    b = build_kid_level_prompts(n=20, seed=42)
    assert [p.prompt for p in a] == [p.prompt for p in b]


if __name__ == "__main__":
    run_module_tests(globals(), "eval kid-level prompts")
