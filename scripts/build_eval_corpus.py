#!/usr/bin/env python3
"""Build or verify the frozen tokenizer-benchmark eval corpus (adhan#74, part 1).

Build:
    python scripts/build_eval_corpus.py --spec eval_manifests/tokenizer_bench.sources.json

Verify (rebuilds from the raw inputs and compares to the committed manifest):
    python scripts/build_eval_corpus.py --verify

The spec lists every source with its register, path, licence and URL. See
docs/EVAL_CORPUS.md and eval_manifests/tokenizer_bench.sources.example.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.eval import eval_corpus as ec  # noqa: E402

DEFAULT_SPEC = "eval_manifests/tokenizer_bench.sources.json"
DEFAULT_OUT = "data/eval/tokenizer_bench.jsonl"
DEFAULT_MANIFEST = "eval_manifests/tokenizer_bench.manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--verify", action="store_true", help="rebuild and compare to the manifest")
    ap.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="do not fail when a register has fewer sentences than per_register",
    )
    args = ap.parse_args()

    try:
        if args.verify:
            manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
            problems = ec.verify(ROOT / args.out, manifest, ROOT)
            if problems:
                for p in problems:
                    print(f"FAIL: {p}", file=sys.stderr)
                return 1
            print(f"OK: {args.out} matches {args.manifest}")
            return 0

        spec = ec.load_spec(ROOT / args.spec)
        result = ec.build(spec, ROOT)
        manifest = result["manifest"]
        short = ec.shortfalls(manifest)
        if short and not args.allow_shortfall:
            print("FAIL: not enough usable sentences: " + "; ".join(short), file=sys.stderr)
            print("Add sources, lower per_register, or pass --allow-shortfall.", file=sys.stderr)
            return 1
        ec.write_outputs(result, ROOT / args.out, ROOT / args.manifest)
    except ec.EvalCorpusError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    counts = manifest["counts"]
    print(f"wrote {counts['total']} sentences to {args.out}")
    for reg, n in counts["per_register"].items():
        print(f"  {reg:9s} {n}")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
