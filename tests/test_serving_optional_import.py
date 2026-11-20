"""Regression tests: the API server must import without the optional `serving` extra (adhan#121).

`scripts/run_api_server.py` used to call ``exit(1)`` at import time when ``fastapi`` was
missing. Any test module importing it therefore killed pytest collection with an
``INTERNALERROR`` — the whole suite reported "no tests ran". These tests pin the fixed
behaviour by importing the module in a subprocess where the dependency is genuinely
unavailable, so they hold on machines that do have the extra installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Import the server with `fastapi`/`uvicorn` blocked via `sys.modules[name] = None`.
_BLOCKED = textwrap.dedent("""
    import sys

    for name in ("fastapi", "uvicorn"):
        sys.modules[name] = None          # `import name` now raises ImportError
        for sub in [m for m in list(sys.modules) if m.startswith(name + ".")]:
            sys.modules[sub] = None

    sys.path.insert(0, {src!r})
    sys.path.insert(0, {root!r})

    import scripts.run_api_server as srv

    assert srv.SERVING_IMPORT_ERROR, "the missing extra should be detected"
    assert "fastapi" in srv.SERVING_IMPORT_ERROR.lower() or "uvicorn" in srv.SERVING_IMPORT_ERROR.lower()

    # The dependency only surfaces when the app is built, not at import time.
    try:
        srv.create_app()
    except ImportError as exc:
        message = str(exc)
        assert "serving" in message, message
        assert "pip install" in message, message
    else:
        raise AssertionError("create_app() must raise ImportError without the serving extra")

    print("IMPORT_OK")
    """)


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script.format(src=str(SRC), root=str(ROOT))],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_importing_the_api_server_does_not_exit_without_the_extra() -> None:
    proc = _run(_BLOCKED)
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_OK" in proc.stdout
    assert "SystemExit" not in proc.stderr


def test_main_returns_a_message_not_a_crash_without_the_extra() -> None:
    script = _BLOCKED.replace(
        'print("IMPORT_OK")',
        textwrap.dedent("""
            sys.argv = ["run_api_server.py"]
            code = srv.main()
            assert code == 2, code
            print("MAIN_CODE", code)
            """),
    )
    proc = _run(script)
    assert proc.returncode == 0, proc.stderr
    assert "MAIN_CODE 2" in proc.stdout
    assert "serving" in proc.stderr


def test_create_app_works_when_the_extra_is_installed() -> None:
    """The other half of the contract, when this environment has fastapi."""
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(ROOT))
    srv = pytest.importorskip("scripts.run_api_server")
    if srv.SERVING_IMPORT_ERROR is not None:
        pytest.skip("the serving extra is not installed here")

    app = srv.create_app()
    assert getattr(app, "title", "") == "Adhan SLM Inference API"
