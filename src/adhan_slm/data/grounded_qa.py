"""Build grounded QA supervision: answer + supporting span + citation, or refusal.

Two adapters, one schema:

``indicqa``
    Uses the Tamil IndicQA file already vendored by ``bench/tamil-bench``. Real
    questions with gold answers and character offsets; split by article so the same
    passage never appears in train and test. Unanswerable examples are made by
    pairing a question with a *different* article that provably lacks its answer.

``corpus``
    The SCERT adapter. Drop the textbook text under a directory (``.txt`` or
    ``.jsonl`` with a ``text`` field) and it produces fill-in-the-blank items whose
    answer is an exact token span in the passage. 15% of rows remove the
    answer-bearing sentence from the context, so the correct behaviour is refusal.

Everything is deterministic. No teacher model writes the targets: a wrong gold
answer is worse than a missing example, and there is no verifier in this loop yet.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from adhan_slm import protocol as p

TASK_QA = "qa_grounded"
TASK_CLOZE = "qa_cloze"

SYSTEM = (
    "கொடுக்கப்பட்ட பத்தியை மட்டும் பயன்படுத்தி, சுருக்கமாகத் தமிழில் பதில் சொல். "
    "பதில் பத்தியில் இல்லையெனில், தகவல் இல்லை என்று சொல்."
)

_TAMIL_WORD = re.compile(r"[\u0B80-\u0BFF]{4,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def split_of(key: str, val_frac: float = 0.10, test_frac: float = 0.10) -> str:
    bucket = (
        int(hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16)
        % 10_000
        / 10_000
    )
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


# --------------------------------------------------------------------------- #
# IndicQA adapter
# --------------------------------------------------------------------------- #
def load_indicqa(path: str | Path) -> List[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    rows: List[dict] = []
    for article in raw:
        for para in article.get("paragraphs", []):
            context = (para.get("context") or "").strip()
            if not context:
                continue
            context_id = _sha(context)
            for qa in para.get("qas", []):
                answers = [a for a in qa.get("answers", []) if (a.get("text") or "").strip()]
                if not answers:
                    continue
                rows.append(
                    {
                        "id": str(qa.get("id")),
                        "context_id": context_id,
                        "context": context,
                        "question": (qa.get("question") or "").strip(),
                        "answers": answers,
                    }
                )
    rows.sort(key=lambda r: (r["context_id"], r["id"]))
    return rows


def _qa_example(
    row: dict,
    answer: str,
    answerable: bool,
    context: Optional[str] = None,
    citation_ref: Optional[str] = None,
    reason: Optional[str] = None,
) -> dict:
    context = context if context is not None else row["context"]
    messages = [{"role": "user", "content": f"பத்தி:\n{context}\n\nகேள்வி: {row['question']}"}]
    citations = (
        [p.citation("IndicQA", citation_ref or f"{row['context_id']}.{row['id']}")]
        if answerable
        else []
    )
    prompt, target = p.render_example(messages, answer, citations=citations, system=SYSTEM)
    gold = row["answers"][0]
    return {
        "id": f"{TASK_QA}:{row['id']}",
        "task": TASK_QA,
        "prompt": prompt,
        "target": target,
        "answer": answer,
        "citations": [c for c in citations],
        "messages": messages,
        "meta": {
            "source": "indicqa",
            "context_id": row["context_id"],
            "qa_id": row["id"],
            "answerable": answerable,
            "answer_start": gold.get("answer_start"),
            "gold_answers": [a["text"] for a in row["answers"]],
            **({"unanswerable_reason": reason} if reason else {}),
        },
    }


def build_indicqa_split(
    rows: Sequence[dict],
    split: str,
    seed: int = 42,
    unanswerable_frac: float = 0.15,
) -> List[dict]:
    """Rows are all from one split; contexts may repeat across questions."""
    rng = random.Random(f"{seed}:{split}")
    rows = list(rows)
    rng.shuffle(rows)
    contexts = sorted({r["context_id"]: r["context"] for r in rows}.items())
    n_unanswerable = round(len(rows) * unanswerable_frac)
    unanswerable_ids = {r["id"] for r in rows[:n_unanswerable]}

    out: List[dict] = []
    for row in rows:
        if row["id"] not in unanswerable_ids:
            out.append(_qa_example(row, row["answers"][0]["text"], True))
            continue
        # Pick a context from another article that does not contain the gold answer.
        gold = _norm(row["answers"][0]["text"])
        candidates = [ctx for cid, ctx in contexts if cid != row["context_id"]]
        if not candidates:
            out.append(_qa_example(row, row["answers"][0]["text"], True))
            continue
        rng.shuffle(candidates)
        chosen = next((c for c in candidates if gold and gold not in _norm(c)), None)
        if chosen is None:
            out.append(_qa_example(row, row["answers"][0]["text"], True))
            continue
        out.append(
            _qa_example(
                row,
                p.REFUSAL,
                False,
                context=chosen,
                reason="context_does_not_contain_answer",
            )
        )
    out.sort(key=lambda r: r["id"])
    return out


# --------------------------------------------------------------------------- #
# Corpus / SCERT adapter (fill-in-the-blank with an exact span)
# --------------------------------------------------------------------------- #
def _read_corpus(dir_path: Path) -> Iterable[Tuple[str, str]]:
    root = Path(dir_path)
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if path.is_dir():
            continue
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = (obj.get("text") or "").strip()
                    if text:
                        yield path.stem, text
        elif path.suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                yield path.stem, text


def build_corpus_examples(
    corpus_dir: str | Path,
    unanswerable_frac: float = 0.15,
    seed: int = 42,
    max_examples: Optional[int] = None,
) -> List[dict]:
    """Cloze items from a plain-text/JSONL corpus (SCERT textbooks, handbooks).

    A question is the sentence with one content word blanked; the answer is that
    word and its span. For an unanswerable row the answer-bearing sentence is
    removed from the context, so a correct model refuses instead of guessing.
    """
    rng = random.Random(f"{seed}:corpus")
    rows: List[dict] = []
    for doc_id, text in _read_corpus(Path(corpus_dir)):
        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        for i, sent in enumerate(sentences):
            words = _TAMIL_WORD.findall(sent)
            if len(sent) < 40 or not words:
                continue
            # Prefer a word that occurs once in the sentence: an unambiguous blank.
            counts = {w: sent.count(w) for w in set(words)}
            word = next((w for w in sorted(counts) if counts[w] == 1 and len(w) >= 5), None)
            if word is None:
                continue
            blanked = sent.replace(word, "____", 1)
            context = " ".join(sentences[max(0, i - 2) : i + 3]).strip()
            rows.append(
                {
                    "doc_id": doc_id,
                    "sent_idx": i,
                    "question": f"கோடிட்ட இடத்தை நிரப்புக: {blanked}",
                    "context": context,
                    "sentence": sent,
                    "answer": word,
                    "span_start": sent.find(word),
                }
            )
            if max_examples and len(rows) >= max_examples:
                break
        if max_examples and len(rows) >= max_examples:
            break

    n_unanswerable = round(len(rows) * unanswerable_frac)
    unanswerable_idx = set(rng.sample(range(len(rows)), n_unanswerable)) if rows else set()
    out: List[dict] = []
    for i, row in enumerate(rows):
        answerable = i not in unanswerable_idx
        context = row["context"]
        answer = row["answer"]
        withheld = None
        if not answerable:
            # Remove the answer-bearing sentence; the question keeps the blank.
            withheld = row["answer"]
            context = " ".join(
                s for s in _SENT_SPLIT.split(context) if s.strip() != row["sentence"]
            ).strip()
            answer = p.REFUSAL
        ref = f"{row['doc_id']}.{row['sent_idx']}"
        messages = [{"role": "user", "content": f"பத்தி:\n{context}\n\n{row['question']}"}]
        citations = [p.citation("SCERT", ref)] if answerable else []
        prompt, target = p.render_example(messages, answer, citations=citations, system=SYSTEM)
        out.append(
            {
                "id": f"{TASK_CLOZE}:{row['doc_id']}.{row['sent_idx']}",
                "task": TASK_CLOZE,
                "prompt": prompt,
                "target": target,
                "answer": answer,
                "citations": citations,
                "messages": messages,
                "meta": {
                    "source": "corpus",
                    "doc_id": row["doc_id"],
                    "sent_idx": row["sent_idx"],
                    "answerable": answerable,
                    "answer_start": row["span_start"] if answerable else None,
                    **(
                        {
                            "unanswerable_reason": "answer_sentence_removed",
                            "withheld_answer": withheld,
                        }
                        if not answerable
                        else {}
                    ),
                },
            }
        )
    out.sort(key=lambda r: r["id"])
    return out


def stats(examples: Dict[str, List[dict]]) -> dict:
    all_rows = [e for xs in examples.values() for e in xs]

    def counts(values: Iterable[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items()))

    return {
        "by_split": {k: len(v) for k, v in examples.items()},
        "by_task": counts(e["task"] for e in all_rows),
        "sources": counts(e["meta"]["source"] for e in all_rows),
        "answerable": {
            "true": sum(1 for e in all_rows if e["meta"]["answerable"]),
            "false": sum(1 for e in all_rows if not e["meta"]["answerable"]),
        },
        "total": len(all_rows),
    }
