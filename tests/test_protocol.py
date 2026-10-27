"""Tests for the textual response protocol (docs/behaviour/RESPONSE_PROTOCOL.md)."""

import pytest

from adhan_slm import protocol as p


def test_render_prompt_multiturn_with_system():
    out = p.render_prompt(
        [
            {"role": "user", "content": "என் பெயர் மீரா."},
            {"role": "assistant", "content": "வணக்கம் மீரா."},
            {"role": "user", "content": "என் பெயர் என்ன?"},
        ],
        system="சுருக்கமாகப் பதில் சொல்.",
    )
    assert out.startswith(f"{p.SYSTEM} சுருக்கமாகப் பதில் சொல்.")
    assert f"{p.USER} என் பெயர் மீரா." in out
    assert f"{p.ASSISTANT} வணக்கம் மீரா." in out
    assert out.endswith(p.ASSISTANT)
    assert out.count(p.USER) == 2


def test_render_prompt_rejects_bad_role_and_empty_content():
    with pytest.raises(ValueError):
        p.render_prompt([{"role": "tool", "content": "x"}])
    with pytest.raises(ValueError):
        p.render_prompt([{"role": "user", "content": "   "}])


def test_render_target_thought_and_citations():
    target = p.render_target(
        "சென்னை",
        thought="தலைநகரம் பற்றிய தகவல் பத்தியில் உள்ளது.",
        citations=[p.citation("SCERT", "4.2")],
    )
    assert target.splitlines()[0].startswith(p.THOUGHT)
    assert target.splitlines()[1].startswith(p.ANSWER)
    assert "[மேற்கோள்: SCERT 4.2]" in target


def test_render_target_rejects_malformed_citation_and_empty_answer():
    with pytest.raises(ValueError):
        p.render_target("x", citations=["SCERT 4.2"])
    with pytest.raises(ValueError):
        p.render_target("   ")


def test_parse_roundtrip_with_thought_and_citation():
    target = p.render_target(
        "குறிஞ்சி.",
        thought="மலை, யானை, மயில் ஆகியன குறிஞ்சிக் கருப்பொருள்கள்.",
        citations=[p.citation("ஐங்குறுநூறு", "201")],
    )
    parsed = p.parse_response(target)
    assert parsed["found_answer"] is True
    assert parsed["thought"].startswith("மலை")
    assert parsed["answer"] == "குறிஞ்சி."
    assert parsed["citations"] == ["ஐங்குறுநூறு 201"]


def test_parse_without_thought():
    parsed = p.parse_response("பதில்: சென்னை [மேற்கோள்: SCERT 4.2]")
    assert parsed["thought"] is None
    assert parsed["answer"] == "சென்னை"
    assert parsed["citations"] == ["SCERT 4.2"]


def test_parse_plain_text_is_flagged_as_protocol_miss():
    parsed = p.parse_response("சென்னை")
    assert parsed["found_answer"] is False
    assert parsed["answer"] == "சென்னை"


def test_refusal_helpers():
    rendered = p.render_target(p.REFUSAL)
    assert p.parse_response(rendered)["answer"] == p.REFUSAL
    assert p.is_refusal(p.REFUSAL) is True
    assert p.is_refusal("  " + " ".join(p.REFUSAL.split()) + " ") is True
    assert p.is_refusal("சென்னை") is False


def test_citation_requires_both_parts():
    with pytest.raises(ValueError):
        p.citation("", "1")
    with pytest.raises(ValueError):
        p.citation("SCERT", " ")
