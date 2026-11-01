#!/usr/bin/env python3
"""Reproduce the Tamil tokenizer benchmark with one command (adhan#74, part 3).

    python scripts/reproduce_benchmark.py            # rebuild everything, write tables
    python scripts/reproduce_benchmark.py --check    # rebuild and compare to committed tables

Steps: check pinned versions, fetch missing public sources, rebuild the frozen eval corpus
(and compare it with the committed manifest), run the harness, write or check the tables.
See docs/REPRODUCE_BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.eval import reproduce  # noqa: E402

DEFAULT_CONFIG = "eval_manifests/tokenizer_bench.reproduce.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--check", action="store_true", help="compare with committed tables; write nothing"
    )
    ap.add_argument(
        "--refreeze", action="store_true", help="accept a changed corpus and rewrite the manifest"
    )
    ap.add_argument(
        "--allow-version-drift", action="store_true", help="warn instead of failing on pin mismatch"
    )
    args = ap.parse_args()
    try:
        cfg = reproduce.load_config(ROOT / args.config)
        reproduce.run(
            cfg,
            ROOT,
            check=args.check,
            refreeze=args.refreeze,
            allow_version_drift=args.allow_version_drift,
        )
    except (reproduce.ReproduceError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
