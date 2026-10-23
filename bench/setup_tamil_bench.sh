#!/usr/bin/env bash
# Set up the tamil-bench harness for grading a local Adhan checkpoint.
#
#   ./bench/setup_tamil_bench.sh
#
# Clones LogicIncZo/tamil-bench (shallow), applies the local endpoint patch
# (bench/tamil-bench-local.patch) and fetches the open Tamil datasets.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/tamil-bench"
PATCH="$HERE/tamil-bench-local.patch"

if [ ! -d "$REPO/.git" ]; then
  echo ">> cloning tamil-bench"
  git clone --depth 1 https://github.com/LogicIncZo/tamil-bench.git "$REPO"
fi

if ! grep -q "TAMIL_BENCH_ENDPOINT" "$REPO/bench.py"; then
  echo ">> applying endpoint patch"
  git -C "$REPO" apply "$PATCH"
else
  echo ">> endpoint patch already applied"
fi

echo ">> fetching IndicQA + IndicXNLI Tamil splits"
python - <<'PY'
import os, shutil, pathlib
from huggingface_hub import hf_hub_download

# invoked from the repo root (see bench/README.md)
d = pathlib.Path(os.environ.get("TAMIL_BENCH_DIR", "bench/tamil-bench")) / "data"
d.mkdir(parents=True, exist_ok=True)
for repo, file, dest in [
    ("ai4bharat/IndicQA", "data/indicqa.ta.json", "indicqa.ta.json"),
    ("AdaMLLab/indicxnli_repaired", "data/ta/test-00000-of-00001.parquet", "indicxnli_ta_test.parquet"),
]:
    target = d / dest
    if target.exists():
        print("   have", target.name)
        continue
    p = hf_hub_download(repo, file, repo_type="dataset")
    shutil.copy(p, target)
    print("   fetched", target.name)
PY

echo ">> done. MILU is gated: see bench/README.md"
