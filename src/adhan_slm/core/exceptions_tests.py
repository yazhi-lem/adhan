"""Tests for the AdhanError exception hierarchy (`adhan_slm.core.exceptions`).

Run: PYTHONPATH=src python -m adhan_slm.core.exceptions_tests
"""

from __future__ import annotations

from .exceptions import AdhanError, ExternalDependencyError, TokenizerError
from .selftest import run_module_tests


def test_adhan_error_formats_code_and_message() -> None:
    err = AdhanError("Test error", error_code="TEST_001")
    assert "TEST_001" in str(err)
    assert "Test error" in str(err)


def test_adhan_error_renders_context_pairs() -> None:
    """Context is what makes a raised error debuggable without a repro."""
    err = AdhanError(
        "Tokenization failed",
        error_code="TOKEN_001",
        context={"text": "தமிழ்", "position": 42},
    )
    assert "TOKEN_001" in str(err)
    assert "text=தமிழ்" in str(err)
    assert "position=42" in str(err)


def test_subclasses_share_the_adhan_error_base() -> None:
    """One `except AdhanError` must catch everything the package raises."""
    assert isinstance(TokenizerError("Vocab mismatch"), AdhanError)


def test_external_dependency_error_suggests_an_install() -> None:
    err = ExternalDependencyError("jax", operation="train")
    assert "jax" in str(err)
    assert "train" in str(err)
    assert "pip install jax" in err.install_cmd


if __name__ == "__main__":
    run_module_tests(globals(), "core exceptions")
