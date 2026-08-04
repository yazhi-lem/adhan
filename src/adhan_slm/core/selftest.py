"""Shared helpers for Adhan's dual-mode test modules.

Every `*_tests.py` in this repo runs two ways:

  * under pytest — `pytest src/adhan_slm/tokenizer/swaram_tokenizer_tests.py`
  * standalone   — `PYTHONPATH=src python -m adhan_slm.tokenizer.swaram_tokenizer_tests`

The standalone path matters because the tokenizer and data-pipeline cores are
deliberately dependency-free (pure python, no numpy/JAX/pytest), so their tests have
to be runnable in that same minimal environment. This module holds the two pieces
that were otherwise copy-pasted into every test file:

  * `run_module_tests()` — the `if __name__ == "__main__"` runner
  * `skip_unless()` — an optional-dependency guard that reports a real pytest SKIP
    under pytest and degrades to a printed note when pytest is absent

Nothing here imports pytest at module scope, so it stays importable in the minimal
environment.
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Callable, Dict, List, Optional


class SkippedTest(Exception):
    """Raised by `skip_unless` in the standalone runner (pytest raises its own)."""


def _emit_skip(reason: str) -> None:
    """Real pytest skip when collected by pytest; `SkippedTest` when standalone.

    Both paths *raise* rather than return, so a skipped test can never be reported as
    a pass — the whole reason `skip_unless` exists instead of `if not HAS_X: return`.
    """
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(reason)
    raise SkippedTest(reason)


def skip_unless(condition: bool, reason: str) -> Callable:
    """Skip the decorated test unless `condition` holds.

    Preferred over a bare `if not HAS_X: return`, which pytest reports as a **pass** —
    a silently-passing test for a probe that never ran is worse than no test. Applied
    as a decorator (not `pytest.mark.skipif`) so the guard is honoured in the
    standalone runner too, where marks are inert.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not condition:
                _emit_skip(reason)
                return None
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def collect_module_tests(namespace: Dict[str, Any]) -> List[Callable]:
    """Zero-argument `test_*` callables in a module namespace, in definition order.

    Functions that take arguments are skipped: those are pytest tests driven by
    fixtures, which the standalone runner cannot supply.
    """
    import inspect

    found = []
    for name, value in namespace.items():
        if not name.startswith("test_") or not callable(value):
            continue
        try:
            required = [
                p
                for p in inspect.signature(value).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
        except (TypeError, ValueError):  # builtins / C callables
            continue
        if not required:
            found.append(value)
    return found


def run_module_tests(namespace: Dict[str, Any], label: Optional[str] = None) -> int:
    """Run a module's fixture-free `test_*` functions. Returns the number that passed.

    Use from a test module's entry point:

        if __name__ == "__main__":
            run_module_tests(globals(), "swaram tokenizer")
    """
    title = label or namespace.get("__name__", "tests")
    tests = collect_module_tests(namespace)
    if not tests:
        print(f"{title}: no fixture-free tests to run standalone (use pytest)")
        return 0
    print(f"{title}:")
    passed, skipped = 0, 0
    for fn in tests:
        try:
            fn()
        except SkippedTest as e:
            skipped += 1
            print(f"  SKIP {fn.__name__} ({e})")
            continue
        passed += 1
        print(f"  PASS {fn.__name__}")
    tail = f", {skipped} skipped" if skipped else ""
    print(f"\n{passed} passed{tail} ({title}).")
    return passed
