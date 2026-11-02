#!/usr/bin/env python3
"""Fast likelihood-based MCQ eval for the SFT checkpoint (no generation).

For each option, score the target ``பதில்: <option>`` under the protocol prompt and
pick the lowest-NLL option. One forward per option instead of autoregressive
generation, so this runs in ~minutes on CPU.

    python scripts/eval_sft_mcq.py --checkpoint checkpoints/adhan-nano-sft \
        --bench bench/tamil-bench/data/sangam_ta_test.jsonl --n 182 \
        --out docs/eval_sft_sangam_mcq.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm import protocol as p  # noqa: E402
from adhan_slm.data.sangam_sft import SYSTEM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/adhan-nano-sft")
    ap.add_argument("--bench", default="bench/tamil-bench/data/sangam_ta_test.jsonl")
    ap.add_argument("--tokenizer", default="data/final/nano_300m_s512")
    ap.add_argument("--n", type=int, default=182)
    ap.add_argument("--out", default="docs/reports/eval_sft_sangam_mcq.json")
    args = ap.parse_args()

    from adhan_slm.training.device import configure_backend

    configure_backend("cpu")
    import jax
    import jax.numpy as jnp
    import optax

    from adhan_slm.inference import load_model, load_tokenizer

    tok = load_tokenizer(ROOT / args.tokenizer)
    model, params, cfg = load_model(None, ROOT / args.checkpoint, vocab_size=len(tok))
    rows = [json.loads(line) for line in open(ROOT / args.bench) if line.strip()][: args.n]
    max_len = cfg.max_seq_len

    @jax.jit
    def nll(ids, mask):
        logits = model.apply({"params": params}, ids[None, :-1], deterministic=True)
        targets = ids[1:][None, :]
        ll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        m = mask[1:]
        return (ll * m).sum() / jnp.maximum(m.sum(), 1.0)

    bos, eos = tok.vocab["<bos>"], tok.vocab["<eos>"]
    correct = 0
    by_tinai = {}
    details = []
    for i, row in enumerate(rows):
        label = row.get("context_label", "பாடல்")
        prompt = p.render_prompt(
            [{"role": "user", "content": f"{label}:\n{row['context']}\n\n{row['question']}"}],
            system=SYSTEM,
        )
        pid = tok.encode(prompt)
        scores = []
        for opt in row["options"]:
            tgt = tok.encode(p.render_target(opt))
            budget = max_len - len(tgt) - 2
            p_ids = pid[-budget:] if len(pid) > budget else pid
            ids = np.asarray([bos, *p_ids, *tgt, eos], dtype=np.int32)
            mask = np.asarray([0] * (1 + len(p_ids)) + [1] * (len(tgt) + 1), dtype=np.float32)
            scores.append(float(nll(jnp.asarray(ids), jnp.asarray(mask))))
        pred = int(np.argmin(scores))
        ok = pred == row["answer_index"]
        correct += int(ok)
        t = row.get("tinai") or "unknown"
        c, n = by_tinai.get(t, (0, 0))
        by_tinai[t] = (c + int(ok), n + 1)
        details.append(
            {
                "id": row["id"],
                "tinai": t,
                "gold": row["answer_index"],
                "pred": pred,
                "correct": ok,
                "nll": scores,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(rows)}  acc={correct/(i+1):.3f}", flush=True)

    acc = correct / len(rows)
    report = {
        "checkpoint": str(args.checkpoint),
        "n": len(rows),
        "accuracy": round(acc, 4),
        "chance": 0.2,
        "by_tinai": {
            k: {"correct": c, "n": n, "accuracy": round(c / n, 4)}
            for k, (c, n) in sorted(by_tinai.items())
        },
        "details": details,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSangam tiṇai MCQ (likelihood): {correct}/{len(rows)} = {acc:.3f}")
    for k, v in report["by_tinai"].items():
        print(f"  {k}: {v['correct']}/{v['n']} = {v['accuracy']:.3f}")
    print(f"saved: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
