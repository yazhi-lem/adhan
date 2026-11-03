#!/usr/bin/env python3
"""Probe whether `google/muril-base-cased` is useful to Adhan.

Two questions, one script:

1. **Tamil understanding** — frozen MuRIL sentence embeddings + a linear probe on
   IndicXNLI-Tamil. This is an *encoder* baseline, not a tamil-bench chat result.
2. **Tanglish political content** — frozen embeddings + a linear probe on a
   keyword-derived weak label over the Tanglish corpus. This measures how separable
   political text is; it is not a production political classifier.

Usage:

    pip install -e ".[muril]"
    python scripts/probe_muril.py \
        --tanglish /path/to/tanglish_corpus.jsonl \
        --xnli-dir /path/to/xnli_ta/ \
        --out docs/reports/muril_probe.json

`--xnli-dir` expects `train.parquet`, `validation.parquet`, `test.parquet` from
`AdaMLLab/indicxnli_repaired` (Tamil split). The script is CPU-friendly: it never
fine-tunes MuRIL, it only extracts frozen `[CLS] + mean-pool` embeddings and fits a
logistic regression. That is enough to answer "is there signal?", not enough to ship.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

MODEL = "google/muril-base-cased"
REVISION = "afd9f36c7923d54e97903922ff1b260d091d202f"

# High-recall weak label for political / sensitive public-affairs text. It is a
# *seed*, not the classifier: the point of the probe is to see whether MuRIL
# representations can learn beyond these literal strings.
POLITICAL_RE = re.compile(
    r"\b(tvk|dmk|admk|aiadmk|bjp|ntk|pmk|vck|mdmk|congress|cpi|cpim|"
    r"election|elections|vote|votes|voter|voters|government|govt|minister|"
    r"governor|mla|mp|parliament|assembly|politics|political|party|alliance|"
    r"constituency|modi|stalin|udhayanidhi|vijay|seeman|rahul|amit shah|"
    r"corruption|scam|protest|reservation|caste|religion|temple|church|"
    r"mosque|hindu|christian|muslim)\b",
    re.I,
)
TAMIL = re.compile(r"[\u0B80-\u0BFF]")


def _load_muril():
    try:
        from transformers import AutoTokenizer, FlaxBertModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "transformers is required: pip install -e '.[muril]' "
            "(Flax backend; torch is not needed)"
        ) from exc
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = FlaxBertModel.from_pretrained(MODEL, revision=REVISION, dtype="float32")
    return tok, model


def embed(tok, model, texts, pairs=None, batch=32, max_len=96, label="embed"):
    """Return concat([CLS], mean-pool) embeddings, 1536-d for BERT-base."""
    vectors = []
    started = time.time()
    for start in range(0, len(texts), batch):
        if pairs is None:
            enc = tok(
                texts[start : start + batch],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="np",
            )
        else:
            a, b = pairs
            enc = tok(
                a[start : start + batch],
                b[start : start + batch],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="np",
            )
        out = model(**enc)
        hidden = np.asarray(out.last_hidden_state)
        mask = np.asarray(enc["attention_mask"])[..., None]
        mean = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1)
        vectors.append(np.concatenate([hidden[:, 0, :], mean], axis=1))
        if start % (batch * 10) == 0:
            print(f"{label}: {start}/{len(texts)} ({time.time() - started:.0f}s)", flush=True)
    return np.vstack(vectors)


def _xnli(xnli_dir: Path, tok, model, max_train: int, max_test: int) -> dict:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    train = pd.read_parquet(xnli_dir / "train.parquet")
    val = pd.read_parquet(xnli_dir / "validation.parquet")
    test = pd.read_parquet(xnli_dir / "test.parquet")
    rng = np.random.default_rng(42)
    train = train.iloc[rng.choice(len(train), size=min(max_train, len(train)), replace=False)]
    val = val.iloc[rng.choice(len(val), size=min(2000, len(val)), replace=False)]
    test = test.iloc[rng.choice(len(test), size=min(max_test, len(test)), replace=False)]

    x_train = embed(tok, model, train.premise.tolist(), pairs=(train.premise.tolist(), train.hypothesis.tolist()), label="xnli-train")
    x_val = embed(tok, model, val.premise.tolist(), pairs=(val.premise.tolist(), val.hypothesis.tolist()), label="xnli-val")
    x_test = embed(tok, model, test.premise.tolist(), pairs=(test.premise.tolist(), test.hypothesis.tolist()), label="xnli-test")

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x_train, train.label.values)
    pred = clf.predict(x_test)
    return {
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
        "val_acc": round(float(accuracy_score(val.label.values, clf.predict(x_val))), 4),
        "test_acc": round(float(accuracy_score(test.label.values, pred)), 4),
        "majority_baseline": round(float(test.label.value_counts(normalize=True).max()), 4),
        "per_class": classification_report(
            test.label.values, pred, output_dict=True, zero_division=0
        ),
    }


def _tanglish(path: Path, tok, model, sample: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (record.get("text") or "").strip()
            if 15 <= len(text) <= 300 and not TAMIL.search(text):
                rows.append(text)

    political = [t for t in rows if POLITICAL_RE.search(t)]
    neutral = [t for t in rows if not POLITICAL_RE.search(t)]
    rng = np.random.default_rng(42)
    political = [political[i] for i in rng.choice(len(political), size=min(sample, len(political)), replace=False)]
    neutral = [neutral[i] for i in rng.choice(len(neutral), size=min(sample, len(neutral)), replace=False)]
    texts = political + neutral
    labels = np.array([1] * len(political) + [0] * len(neutral))
    x = embed(tok, model, texts, max_len=64, label="tanglish")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    proba = cross_val_predict(clf, x, labels, cv=cv, method="predict_proba")[:, 1]
    clf.fit(x, labels)
    return {
        "pool_size": len(rows),
        "political_keyword_hits": len(political),
        "neutral_keyword_misses": len(neutral),
        "sample_political": len(political),
        "sample_neutral": len(neutral),
        "cv_accuracy": round(float(((proba >= 0.5) == labels).mean()), 4),
        "cv_auc": round(float(roc_auc_score(labels, proba)), 4),
        "note": (
            "Weak labels come from POLITICAL_RE; this measures separability/learnability, "
            "not true political-opinion detection. Human-labeled audit is still required."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tanglish", type=Path, help="Tanglish JSONL with a `text` field")
    ap.add_argument("--xnli-dir", type=Path, help="dir with train/validation/test parquet")
    ap.add_argument("--out", type=Path, default=Path("docs/reports/muril_probe.json"))
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-test", type=int, default=5010)
    ap.add_argument("--sample", type=int, default=1000)
    args = ap.parse_args()
    if not args.tanglish and not args.xnli_dir:
        ap.error("pass --tanglish and/or --xnli-dir")

    tok, model = _load_muril()
    report = {
        "model": MODEL,
        "revision": REVISION,
        "embedding": "concat(CLS, mean-pool), 1536-d",
        "frozen": True,
    }
    if args.xnli_dir:
        report["xnli_tamil"] = _xnli(args.xnli_dir, tok, model, args.max_train, args.max_test)
    if args.tanglish:
        report["tanglish_political_weak_label_probe"] = _tanglish(
            args.tanglish, tok, model, args.sample
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
