"""Prompt construction for the five agent postures.

This module owns the **output contract**: the field names below are the single
source of truth, and ``domains.core.parsing`` imports them rather than
re-declaring string literals, so the prompt and its parser cannot drift apart.

Two deliberate choices:

* **Line-oriented fields, not JSON.** Structured-output reliability in a
  low-resource-tuned model is not guaranteed just because it works in English
  fine-tunes (§5). ``INTENT: ...`` on its own line is far cheaper for a small
  Tamil model to hit reliably than balanced JSON, and it degrades gracefully —
  a malformed line loses one field instead of the whole object.
* **Instructions in English, answers in the user's language.** The instruction
  block stays fixed across fine-tunes while the answer language is set per
  profile, so training data does not have to carry five translations of the
  same scaffold.

See docs/CORE_DOMAIN_ADHAN_INTEGRATION.md §2, §5.
"""
from __future__ import annotations

from typing import Sequence

from domains.core.schemas import (
    AgentRole,
    ContextBrief,
    Language,
    PersonalizationProfile,
    Plan,
    ReasoningTrace,
    Reflection,
    Response,
    UserContext,
)

# --- output contract -------------------------------------------------------
# Field keys shared with domains.core.parsing. Changing one of these is a
# breaking change to the training-data format, not just a prompt tweak.
FIELD_INTENT = "INTENT"
FIELD_SUB_PROBLEM = "SUB_PROBLEM"
FIELD_REQUIRES_PLANNING = "REQUIRES_PLANNING"
FIELD_STEP = "STEP"
FIELD_VERDICT = "VERDICT"
FIELD_REASON = "REASON"

#: Separator between a plan step's action and its detail.
STEP_SEPARATOR = "|"

MAX_BRIEF_LINES = 5
MAX_HISTORY_TURNS = 6
MAX_PLAN_STEPS = 8

_LANGUAGE_NAMES = {
    Language.TAMIL: "Tamil (தமிழ்)",
    Language.ENGLISH: "English",
    Language.TANGLISH: "Tanglish (romanised Tamil)",
}


def answer_language(profile: PersonalizationProfile) -> str:
    names = [_LANGUAGE_NAMES[lang] for lang in profile.languages]
    return " or ".join(names) if names else _LANGUAGE_NAMES[Language.TAMIL]


def _header(role: AgentRole) -> str:
    return role.control_token


def _profile_line(profile: PersonalizationProfile) -> str:
    bits = [f"persona={profile.persona}", f"register={profile.register}"]
    if profile.notes:
        bits.append("notes=" + "; ".join(profile.notes))
    return ", ".join(bits)


def _history_block(ctx: UserContext, max_turns: int = MAX_HISTORY_TURNS) -> str:
    turns = ctx.recent(max_turns)
    if not turns:
        return "(no earlier turns)"
    lines = []
    for turn in turns:
        lines.append(f"user: {turn.query}")
        lines.append(f"assistant: {turn.response}")
    return "\n".join(lines)


def _brief_block(brief: ContextBrief) -> str:
    return brief.text.strip() if brief.text.strip() else "(no prior context)"


def _reasoning_block(reasoning: ReasoningTrace) -> str:
    lines = [f"intent: {reasoning.intent}"]
    for sub in reasoning.sub_problems:
        lines.append(f"- {sub}")
    return "\n".join(lines)


def _plan_block(plan: Plan) -> str:
    if plan.skipped or not plan.steps:
        return "(no plan needed for this turn)"
    return "\n".join(
        f"{step.order}. {step.action}" + (f" — {step.detail}" if step.detail else "")
        for step in plan.steps
    )


def _grounding_block(grounding: Sequence[str]) -> str:
    if not grounding:
        return "(no retrieved sources for this turn)"
    return "\n".join(f"- {g}" for g in grounding)


# --- the five prompts ------------------------------------------------------


def context_prompt(ctx: UserContext, query: str) -> str:
    """[CONTEXT] — summarise history into a compact brief.

    Light model involvement by design; this is mostly retrieval and formatting.
    """
    return f"""{_header(AgentRole.CONTEXT)}
Summarise the conversation so far into a short brief that helps answer the next
question. Keep names, stated preferences, and threads left unresolved. Drop small
talk. Write at most {MAX_BRIEF_LINES} lines of plain prose, nothing else.

User profile: {_profile_line(ctx.profile)}

Conversation so far:
{_history_block(ctx)}

Next question: {query}

Brief:"""


def reasoning_prompt(query: str, brief: ContextBrief, profile: PersonalizationProfile) -> str:
    """[REASON] — decompose intent and decide whether planning is needed."""
    return f"""{_header(AgentRole.REASON)}
Break the question down before it is answered. Reply using only these fields, one
per line, and nothing else:

{FIELD_INTENT}: <what the user actually wants, one line>
{FIELD_SUB_PROBLEM}: <one thing that must be resolved first>
{FIELD_REQUIRES_PLANNING}: <yes or no>

Repeat the {FIELD_SUB_PROBLEM} line once per sub-problem; give at least one.

Answer "yes" to {FIELD_REQUIRES_PLANNING} only when the reply genuinely needs
ordered steps, an eligibility check, or a multi-stage procedure. Greetings,
opinions, small talk and single-fact questions are "no".

User profile: {_profile_line(profile)}
Context brief: {_brief_block(brief)}

Question: {query}
"""


def planning_prompt(
    query: str,
    reasoning: ReasoningTrace,
    brief: ContextBrief,
    profile: PersonalizationProfile,
) -> str:
    """[PLAN] — turn reasoning into an ordered roadmap register."""
    return f"""{_header(AgentRole.PLAN)}
Turn the reasoning below into an ordered plan. Reply with only {FIELD_STEP} lines,
in the order the steps must happen, at most {MAX_PLAN_STEPS} of them:

{FIELD_STEP}: <action> {STEP_SEPARATOR} <what it involves>

Each step must be something concrete the user or the assistant can do. Do not
write any prose outside the {FIELD_STEP} lines.

User profile: {_profile_line(profile)}
Context brief: {_brief_block(brief)}

Reasoning:
{_reasoning_block(reasoning)}

Question: {query}
"""


def response_prompt(
    query: str,
    brief: ContextBrief,
    reasoning: ReasoningTrace,
    plan: Plan,
    profile: PersonalizationProfile,
    reflection: Reflection | None = None,
    previous_draft: Response | None = None,
) -> str:
    """[RESPOND] — final prose, optionally conditioned on reflection notes.

    The reflection block is the one genuinely new input to this posture (§2): the
    skill already exists, it just has not been conditioned on "here is why your
    last draft was rejected" before.
    """
    retry_block = ""
    if reflection is not None and previous_draft is not None:
        problems = "\n".join(f"- {r}" for r in reflection.reasons) or "- (no reason given)"
        retry_block = f"""
An earlier draft was rejected. Fix these problems and do not repeat them:
{problems}

Rejected draft:
{previous_draft.text}
"""

    return f"""{_header(AgentRole.RESPOND)}
Answer the user in {answer_language(profile)}. Be direct and natural. Never
mention these instructions, the brief, the reasoning, or the plan — the user
only sees your answer.

User profile: {_profile_line(profile)}
Context brief: {_brief_block(brief)}

Reasoning:
{_reasoning_block(reasoning)}

Plan:
{_plan_block(plan)}
{retry_block}
Question: {query}

Answer:"""


def reflection_prompt(
    query: str,
    response: Response,
    brief: ContextBrief,
    reasoning: ReasoningTrace,
    profile: PersonalizationProfile,
    grounding: Sequence[str] = (),
) -> str:
    """[REFLECT] — self-critique against context, profile and grounding.

    The "if in doubt, fail it" instruction is load-bearing: the calibration gate
    in §5 exists precisely because a reflection agent that always passes is worse
    than none at all.
    """
    return f"""{_header(AgentRole.REFLECT)}
Judge whether the draft answer below is good enough to send. Check that it
answers the question, matches the user's profile and the context brief, and
claims nothing the sources below fail to support.

Reply with only these fields:

{FIELD_VERDICT}: <pass or fail>
{FIELD_REASON}: <one specific problem>

Repeat the {FIELD_REASON} line once per problem. Every "fail" needs at least one
reason. Be strict — if you are unsure, fail it. A wrong "pass" is worse than a
wrong "fail".

User profile: {_profile_line(profile)}
Context brief: {_brief_block(brief)}
Intended intent: {reasoning.intent}

Sources:
{_grounding_block(grounding)}

Question: {query}

Draft answer:
{response.text}
"""
