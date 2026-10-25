"""Aram Harness — Tamil Cultural Safety & Alignment Evaluation Framework.

அறம் (Aram) = Righteousness, virtue, ethical conduct.

The Aram Harness tests Tamil AI models against:
  • Tamil cultural values and classical ethics
  • Safety (harmful request refusal in Tamil)
  • Citation accuracy (no hallucinated Sangam references)
  • Linguistic purity (Tamil-only responses when requested)
  • Caste/religious/gender sensitivity
  • Child safety (COPPA compliance)
  • Constitutional AI principles in Tamil context
"""

from .aram_harness import AramHarness, AramResult, AramSeverity
from .tamil_constitution import TAMIL_CONSTITUTION, evaluate_against_constitution
from .red_team_prompts import RED_TEAM_PROMPTS, RedTeamCategory

__all__ = [
    "AramHarness",
    "AramResult", 
    "AramSeverity",
    "TAMIL_CONSTITUTION",
    "evaluate_against_constitution",
    "RED_TEAM_PROMPTS",
    "RedTeamCategory",
]
