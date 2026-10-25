"""Tests for Sangam SFT task construction (verse, urai, eurai, citations)."""

from adhan_slm.data import sangam_sft as s


def _rec(**over):
    rec = {
        "poem_id": "thirumurugatrupadai_01",
        "work": "thirumurugatrupadai",
        "number": None,
        "verse": "உலகம் உவப்ப வலன்ஏர்பு திரிதரு",
        "urai": "உலகம் மகிழுமாறு கதிரவன் திரிகிறது.",
        "english": "As the world rejoices, the sun circles.",
        "tinai": None,
        "tinai_tamil": None,
        "kalam": None,
        "uripporul_tamil": None,
        "karupporul": ["ஞாயிறு"],
        "poet": "நக்கீரர்",
    }
    rec.update(over)
    return rec


def test_reference_of_falls_back_to_id_suffix():
    assert s.reference_of(_rec()) == "01"
    assert s.reference_of(_rec(number=7)) == "7"


def test_build_examples_covers_citation_urai_and_eurai():
    ex = s.build_examples([_rec()])
    tasks = {e["task"] for e in ex}
    assert s.TASK_CITATION in tasks
    assert s.TASK_URAI in tasks
    assert s.TASK_EURAI_EN in tasks
    assert s.TASK_KARUPPORUL in tasks

    eurai = next(e for e in ex if e["task"] == s.TASK_EURAI_EN)
    assert "மொழிபெயர்ப்பாளர்" in eurai["prompt"]
    assert eurai["answer"] == "As the world rejoices, the sun circles."
    assert "திருமுருகாற்றுப்படை 01" in eurai["target"]
    assert "None" not in eurai["target"]

    cite = next(e for e in ex if e["task"] == s.TASK_CITATION)
    assert cite["answer"] == "திருமுருகாற்றுப்படை 01"
    assert cite["citations"] == ["[மேற்கோள்: திருமுருகாற்றுப்படை 01]"]


def test_eurai_requires_a_verse_not_just_urai():
    ex = s.build_examples([_rec(verse=None)])
    assert s.TASK_EURAI_EN not in {e["task"] for e in ex}


def test_split_of_is_deterministic_and_poem_scoped():
    assert s.split_of("kurunthokai_1") == s.split_of("kurunthokai_1")
    assert s.split_of("kurunthokai_1") in {"train", "val", "test"}


def test_benchmark_has_correct_answer_and_five_options():
    rows = s.build_benchmark([_rec(tinai="kurinji", tinai_tamil="குறிஞ்சி")])
    assert len(rows) == 1
    row = rows[0]
    assert len(row["options"]) == 5
    assert row["options"][row["answer_index"]] == "குறிஞ்சி"
    assert row["context"] == _rec()["verse"]
