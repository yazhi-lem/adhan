"""Tests for grounded-QA construction and refusal/split behaviour."""

import json

from adhan_slm import protocol as p
from adhan_slm.data import grounded_qa as g


def _indicqa_file(tmp_path, n_contexts=3):
    data = {"data": []}
    for i in range(n_contexts):
        context = f"இது பத்தி எண் {i}. இதில் தகவல் {i} உள்ளது."
        data["data"].append(
            {
                "title": f"t{i}",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": f"{i}-{j}",
                                "question": f"பத்தி எண் என்ன {i}?",
                                "answers": [{"text": f"{i}", "answer_start": 23}],
                            }
                            for j in range(2)
                        ],
                    }
                ],
            }
        )
    path = tmp_path / "indicqa.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_indicqa_loader_and_split_are_article_scoped(tmp_path):
    rows = g.load_indicqa(_indicqa_file(tmp_path))
    assert len(rows) == 6
    assert len({r["context_id"] for r in rows}) == 3
    # Every row of one article hashes to exactly one split.
    assert g.split_of(rows[0]["context_id"]) == g.split_of(rows[1]["context_id"])


def test_indicqa_unanswerable_uses_refusal_and_no_citation(tmp_path):
    rows = g.load_indicqa(_indicqa_file(tmp_path, n_contexts=6))
    ex = g.build_indicqa_split(rows, "train", seed=1, unanswerable_frac=1.0)
    assert ex
    for row in ex:
        assert row["meta"]["answerable"] is False
        assert row["answer"] == p.REFUSAL
        assert row["citations"] == []
        assert p.REFUSAL in row["target"]


def test_indicqa_answerable_keeps_span_and_citation(tmp_path):
    rows = g.load_indicqa(_indicqa_file(tmp_path))
    ex = g.build_indicqa_split(rows, "train", seed=1, unanswerable_frac=0.0)
    assert all(r["meta"]["answerable"] for r in ex)
    assert all(r["citations"] for r in ex)
    assert ex[0]["meta"]["answer_start"] == 23


def test_corpus_cloze_blank_and_removed_sentence(tmp_path):
    corpus = tmp_path / "lesson1.txt"
    corpus.write_text(
        "ஒளிச்சேர்க்கை என்பது தாவரங்கள் சூரிய ஒளியைப் பயன்படுத்தி உணவைத் தயாரிக்கும் செயலாகும்.\n"
        "இந்தச் செயலில் தாவரங்கள் கார்பன் டை ஆக்சைடையும் நீரையும் எடுத்துக்கொள்கின்றன.\n"
        "ஒளிச்சேர்க்கையின் விளைவாக ஆக்சிஜன் வெளியிடப்படுகிறது.\n"
        "பச்சையம் என்னும் நிறமி இதற்குத் தேவைப்படுகிறது.\n",
        encoding="utf-8",
    )
    answerable = g.build_corpus_examples(corpus, unanswerable_frac=0.0, seed=1)
    assert answerable
    assert all(r["meta"]["answerable"] for r in answerable)
    assert all("____" in r["messages"][0]["content"] for r in answerable)
    assert all(r["answer"] in r["messages"][0]["content"] for r in answerable)

    unanswerable = g.build_corpus_examples(corpus, unanswerable_frac=1.0, seed=1)
    assert unanswerable
    for row in unanswerable:
        assert row["answer"] == p.REFUSAL
        assert row["citations"] == []
        # The answer-bearing sentence is not part of the context shown to the model.
        assert row["meta"]["withheld_answer"] not in row["prompt"]
