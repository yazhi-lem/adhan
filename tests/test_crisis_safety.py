"""Tests for crisis detection and safety footer module."""

from __future__ import annotations

import pytest

from adhan_slm.safety.crisis import CRISIS_FOOTER, needs_footer

pytestmark = pytest.mark.unit


def test_needs_footer_english():
    assert needs_footer("I want to kill myself") is True
    assert needs_footer("thinking about suicide tonight") is True
    assert needs_footer("I feel happy today") is False


def test_needs_footer_tamil_script():
    assert needs_footer("நான் தற்கொலை செய்ய விரும்புகிறேன்") is True
    assert needs_footer("எனக்கு சாகணும்") is True
    assert needs_footer("இன்று நல்ல நாள்") is False


def test_needs_footer_tanglish():
    assert needs_footer("naan saaga poren") is True
    assert needs_footer("sethuduven da") is True
    assert needs_footer("vanakkam nanba") is False


def test_case_insensitive():
    assert needs_footer("SUICIDE") is True
    assert needs_footer("தற்கொலை") is True


def test_crisis_footer_present():
    assert "Tele-MANAS" in CRISIS_FOOTER
    assert "14416" in CRISIS_FOOTER
    assert "112" in CRISIS_FOOTER
    assert "Tamil" in CRISIS_FOOTER or "தமிழ்" in CRISIS_FOOTER
