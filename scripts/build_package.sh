#!/usr/bin/env bash
# Build + verify the adhan-slm distribution.
#
# Runs the same checks CI gates on, then builds sdist + wheel and asserts that
# every console script declared in pyproject.toml resolves to a real callable.
# Fails loudly instead of producing a package whose entry points are broken.
#
#   ./scripts/build_package.sh              # checks + build
#   ./scripts/build_package.sh --fast       # skip tests (lint + build only)
#
# Needs: hatchling (build backend), black, isort, ruff, pytest.
#   pip install build hatchling black isort ruff pytest
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "Format check (black)"
"$PY" -m black --check src/ tests/ scripts/

say "Import order (isort)"
"$PY" -m isort --check-only src/ tests/ scripts/

say "Lint (ruff)"
"$PY" -m ruff check src/ tests/ scripts/

if [ "$FAST" -eq 0 ]; then
  say "Tests"
  "$PY" -m pytest tests/ src/adhan_slm -q --ignore=tests/integration
fi

say "Entry-point resolution"
"$PY" - <<'EOF'
import importlib
import pathlib
import tomllib

pyproject = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
scripts = pyproject["project"].get("scripts", {})
assert scripts, "no [project.scripts] declared"
for name, target in sorted(scripts.items()):
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path)
    func = getattr(module, attr, None)
    assert callable(func), f"{name}: {target} is not callable"
    print(f"  {name:<20} -> {target}")
print(f"{len(scripts)} console scripts OK")
EOF

say "Build sdist + wheel"
if ! "$PY" -c "import build" 2>/dev/null; then
  echo "ERROR: the 'build' package is not installed in $PY." >&2
  echo "       pip install build hatchling    # build backend is hatchling" >&2
  exit 1
fi
rm -rf dist/
"$PY" -m build --sdist --wheel

say "Contents"
ls -lh dist/
"$PY" - <<'EOF'
import pathlib
import zipfile

wheel = next(pathlib.Path("dist").glob("*.whl"))
with zipfile.ZipFile(wheel) as zf:
    names = zf.namelist()
required = [
    "adhan_slm/chat_cli.py",
    "adhan_slm/common/__init__.py",
    "adhan_slm/common/scripts.py",
    "adhan_slm/data/minhash_dedup.py",
    "adhan_slm/data/contamination.py",
    "adhan_slm/safety/crisis.py",
    "adhan_slm/training/debug_diagnostics.py",
]
missing = [r for r in required if r not in names]
assert not missing, f"wheel is missing: {missing}"
print(f"{wheel.name}: {len(names)} files, all required modules present")
EOF

say "Done"
echo "Installable with: pip install dist/$(ls dist | grep '\.whl$' | head -1)"
