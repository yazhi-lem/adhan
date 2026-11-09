"""Crisis-resource footer — detecting distress in Tamil, Tanglish, and English input.

Adapted from the Agaram 2.5B safety layer (Murai-Labs/Agaram2.5b, Apache-2.0).
The detection list is short and will miss many crisis messages; a production system
should use a proper classifier.  The footer is printed by the CLI wrapper, not by
the model, because the model's crisis training is not guaranteed to trigger on
every distress query.

Usage:
    from adhan_slm.safety.crisis import needs_footer, CRISIS_FOOTER
    if needs_footer(user_text):
        print(CRISIS_FOOTER)
"""

from __future__ import annotations

#: Tamil/Tanglish/English phrases that suggest a user may be in distress.
#: All matching is case-insensitive.
CRISIS_PHRASES: tuple[str, ...] = (
    # English
    "kill myself",
    "suicide",
    "end my life",
    "want to die",
    "self harm",
    "self-harm",
    "hurt myself",
    "not safe",
    "end it all",
    "no point living",
    # Tanglish (romanised Tamil distress expressions)
    "saaga poren",
    "saaganum",
    "sethuduven",
    "sethu poganum",
    "uyiroda irukka venaam",
    "uyiroda irukka vendam",
    "vaazha viruppam illai",
    "vaazha viruppam illa",
    "soga maari irukku",
    "romba kashtam",
    "end my life",
    "die please",
    "help me die",
    # Tamil script
    "தற்கொலை",
    "சாகணும்",
    "சாகப் போறேன்",
    "செத்துடுவேன்",
    "உயிரோட இருக்க வேண்டாம்",
    "உயிரோட இருக்க வேண்டா",
    "வாழ விருப்பம் இல்ல",
    "வாழ விருப்பம் இல்லை",
    "வாழ விருப்பமில்லை",
    "என்னை நானே காயப்படுத்த",
    "சாக விருப்பம்",
    "சாக வேண்டும்",
    "என் உயிரை எடு",
    "சாகலாம்",
    "சாக போறேன்",
    "உயிரோட இருக்க விரும்பல",
    "உயிரோட இருக்க விரும்பலை",
)

#: Bilingual footer shown when distress is detected.
CRISIS_FOOTER: str = (
    "[If you are in distress: Tele-MANAS 14416 (free, 24x7, Tamil supported) · "
    "emergency 112 · "
    "நீங்கள் மன அழுத்தத்தில் இருந்தால் Tele-MANAS 14416 அல்லது 112-ஐ அழையுங்கள்]"
)


def needs_footer(text: str) -> bool:
    """Return True if ``text`` contains any crisis phrase (case-insensitive)."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in CRISIS_PHRASES)
