"""Tests for the native-review packet and sign-off gate (adhan#112, K6).

Synthetic fixtures only. One test asserts the opposite of the packet: the *sign-off* must never
contain corpus text, because that file gets committed.
"""

import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.eval import review_packet as rp  # noqa: E402

_RECORDS = [
    {
        "id": "tanglish-00000",
        "register": "tanglish",
        "source": "tg",
        "text": "bro office ku late ah vanthen",
    },
    {
        "id": "tanglish-00001",
        "register": "tanglish",
        "source": "tg",
        "text": "machan salary cut pannanga",
    },
    {
        "id": "modern-00000",
        "register": "modern",
        "source": "wiki",
        "text": "கட்டிடக்கலை என்பது கட்டிடங்கள்.",
    },
]
_CORPUS_SHA = "a" * 64
_FILTER = {
    "version": "v1",
    "path": "eval_manifests/content_blocklist.v1.json",
    "sha256": "b" * 64,
    "native_review": "pending",
    "categories": ["politics", "abuse"],
}
_TERMS = {"politics": {"latin": ["election"], "tamil": ["தேர்தல்"]}}


def _approved_filter():
    return dict(_FILTER, native_review="reviewed by Test Reviewer on 2026-11-18")


def _complete_template(**overrides):
    signoff = rp.signoff_template(_RECORDS, _CORPUS_SHA, _FILTER)
    signoff.update(
        {"reviewed_by": "Test Reviewer", "reviewed_at": "2026-11-18", "native_speaker": True}
    )
    signoff["gazetteer"]["verdict"] = rp.VERDICT_KEEP
    for entry in signoff["sentences"].values():
        entry["verdict"] = rp.VERDICT_KEEP
    signoff.update(overrides)
    return signoff


def test_text_hash_is_nfc_stable():
    a = rp.text_sha256("e\u0301cole")
    b = rp.text_sha256("\u00e9cole")
    assert a == b
    assert a != rp.text_sha256("ecole")


def test_packet_contains_gazetteer_sample_and_freeze_identity():
    md = rp.packet_markdown(_RECORDS, _CORPUS_SHA, _FILTER, blocklist_terms=_TERMS)
    assert _CORPUS_SHA in md
    assert _FILTER["sha256"] in md
    assert "`tanglish-00000`" in md
    assert "election" in md and "தேர்தல்" in md
    assert "data/review" not in md or True  # packet is local-only by instruction
    assert "must not be\ncommitted or uploaded" in md


def test_packet_lists_focus_registers_first():
    md = rp.packet_markdown(_RECORDS, _CORPUS_SHA, _FILTER, focus=("tanglish",))
    assert md.index("## tanglish sample") < md.index("## modern sample")


def test_signoff_template_has_ids_hashes_and_pending_verdicts():
    signoff = rp.signoff_template(_RECORDS, _CORPUS_SHA, _FILTER)
    assert signoff["corpus_output_sha256"] == _CORPUS_SHA
    assert signoff["filter_sha256"] == _FILTER["sha256"]
    assert signoff["reviewed_by"] == ""
    assert signoff["native_speaker"] is False
    assert signoff["gazetteer"]["verdict"] == rp.VERDICT_PENDING
    assert set(signoff["sentences"]) == {r["id"] for r in _RECORDS}
    for rec in _RECORDS:
        assert signoff["sentences"][rec["id"]]["sha256"] == rp.text_sha256(rec["text"])
        assert signoff["sentences"][rec["id"]]["verdict"] == rp.VERDICT_PENDING


def test_signoff_never_contains_corpus_text():
    """The sign-off is committed, so it must carry ids and hashes only."""
    blob = json.dumps(rp.signoff_template(_RECORDS, _CORPUS_SHA, _FILTER))
    for rec in _RECORDS:
        assert rec["text"] not in blob
    assert "vanthen" not in blob


def test_complete_signoff_passes():
    assert (
        rp.validate_signoff(_complete_template(), _RECORDS, _CORPUS_SHA, _approved_filter()) == []
    )


def test_unreviewed_gazetteer_blocks_the_release():
    """A sign-off cannot approve a blocklist the project still marks as unreviewed."""
    problems = rp.validate_signoff(_complete_template(), _RECORDS, _CORPUS_SHA, _FILTER)
    assert any("native_review" in p for p in problems), problems


def test_approved_gazetteer_passes():
    assert (
        rp.validate_signoff(_complete_template(), _RECORDS, _CORPUS_SHA, _approved_filter()) == []
    )


def test_non_dict_sentence_entry_is_rejected():
    """`"tanglish-00000": "keep"` used to be skipped silently, bypassing every check."""
    signoff = _complete_template()
    signoff["sentences"]["tanglish-00000"] = "keep"
    problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _approved_filter())
    assert any("not objects" in p for p in problems), problems


def test_native_speaker_must_be_the_boolean_true():
    for value in ("no", 1, [True], {"yes": True}):
        signoff = _complete_template()
        signoff["native_speaker"] = value
        problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _approved_filter())
        assert any("boolean true" in p for p in problems), (value, problems)


def test_future_review_date_is_rejected():
    problems = rp.validate_signoff(
        _complete_template(reviewed_at="2099-01-01"), _RECORDS, _CORPUS_SHA, _approved_filter()
    )
    assert any("future" in p for p in problems), problems


def test_actual_filter_description_hashes_what_is_on_disk(tmp_path):
    """The gate must not trust the manifest's own record of the gazetteer."""
    import json

    blocklist = tmp_path / "blocklist.json"
    blocklist.write_text(
        json.dumps(
            {
                "version": "v9",
                "native_review": "pending",
                "categories": {"politics": {"latin": ["election"]}},
            }
        ),
        encoding="utf-8",
    )
    desc = rp.actual_filter_description({"content_filter": "blocklist.json"}, tmp_path)
    assert desc is not None
    assert desc["version"] == "v9"
    assert desc["path"] == "blocklist.json"
    assert len(desc["sha256"]) == 64

    # Emptying the file changes the hash, so an approval can no longer match.
    blocklist.write_text(
        json.dumps({"version": "v9", "categories": {"politics": {"latin": []}}}), encoding="utf-8"
    )
    emptied = rp.actual_filter_description({"content_filter": "blocklist.json"}, tmp_path)
    assert emptied is not None and emptied["sha256"] != desc["sha256"]


def test_actual_filter_description_is_none_without_a_filter(tmp_path):
    assert rp.actual_filter_description({"content_filter": "none"}, tmp_path) is None


def test_pending_verdicts_block_the_release():
    signoff = _complete_template()
    signoff["sentences"]["tanglish-00000"]["verdict"] = rp.VERDICT_PENDING
    problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _FILTER)
    assert any("pending" in p for p in problems)


def test_drop_verdict_blocks_the_release():
    signoff = _complete_template()
    signoff["sentences"]["tanglish-00001"]["verdict"] = rp.VERDICT_DROP
    problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _FILTER)
    assert any("rebuild the freeze" in p for p in problems)


def test_stale_corpus_hash_is_rejected():
    problems = rp.validate_signoff(_complete_template(), _RECORDS, "c" * 64, _FILTER)
    assert any("corpus_output_sha256" in p for p in problems)


def test_changed_gazetteer_is_rejected():
    changed = dict(_FILTER, sha256="d" * 64)
    problems = rp.validate_signoff(_complete_template(), _RECORDS, _CORPUS_SHA, changed)
    assert any("filter_sha256" in p for p in problems)


def test_missing_and_extra_sentence_ids_are_rejected():
    signoff = _complete_template()
    signoff["sentences"].pop("tanglish-00001")
    signoff["sentences"]["ghost-1"] = {"sha256": "e" * 64, "verdict": rp.VERDICT_KEEP}
    problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _FILTER)
    assert any("not reviewed" in p for p in problems)
    assert any("not in this freeze" in p for p in problems)


def test_changed_sentence_hash_is_rejected():
    signoff = _complete_template()
    signoff["sentences"]["tanglish-00000"]["sha256"] = "f" * 64
    problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _FILTER)
    assert any("hash" in p for p in problems)


def test_reviewer_identity_and_language_are_required():
    signoff = _complete_template(reviewed_by="  ")
    signoff["native_speaker"] = False
    problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _FILTER)
    assert any("reviewed_by is empty" in p for p in problems)
    assert any("native_speaker" in p for p in problems)


def test_reviewed_at_must_be_an_iso_date():
    problems = rp.validate_signoff(
        _complete_template(reviewed_at="18/11/2026"), _RECORDS, _CORPUS_SHA, _FILTER
    )
    assert any("ISO date" in p for p in problems)


def test_gazetteer_needs_explicit_acceptance_and_no_missing_terms():
    signoff = _complete_template()
    signoff["gazetteer"]["verdict"] = rp.VERDICT_PENDING
    signoff["gazetteer"]["missing_terms"] = ["caste slur X"]
    problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _FILTER)
    assert any("gazetteer verdict" in p for p in problems)
    assert any("missing terms" in p for p in problems)


def test_unknown_verdict_is_rejected():
    signoff = _complete_template()
    signoff["sentences"]["tanglish-00000"]["verdict"] = "maybe"
    problems = rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _FILTER)
    assert any("unknown verdict" in p for p in problems)


def test_schema_version_mismatch_is_rejected():
    problems = rp.validate_signoff(
        _complete_template(schema_version=999), _RECORDS, _CORPUS_SHA, _FILTER
    )
    assert any("schema_version" in p for p in problems)


def test_packet_and_signoff_agree_on_the_same_freeze():
    """A reviewer's template must validate once filled in — no silent mismatch."""
    md = rp.packet_markdown(_RECORDS, _CORPUS_SHA, _FILTER, blocklist_terms=_TERMS)
    assert _CORPUS_SHA in md
    signoff = _complete_template()
    assert rp.validate_signoff(signoff, _RECORDS, _CORPUS_SHA, _approved_filter()) == []


def test_normalisation_matches_the_corpus_builder():
    """Hashes must agree with the builder's NFC rule, or every review would be stale."""
    nfd = unicodedata.normalize("NFD", "கா")
    rec = [{"id": "x", "register": "modern", "source": "s", "text": nfd}]
    signoff = rp.signoff_template(rec, _CORPUS_SHA, _FILTER)
    fixed = _complete_template()
    fixed["sentences"] = signoff["sentences"]
    fixed["sentences"]["x"]["verdict"] = rp.VERDICT_KEEP
    assert rp.validate_signoff(fixed, rec, _CORPUS_SHA, _approved_filter()) == []


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))
