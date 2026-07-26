"""Parsers for the structured agent postures — the inverse of ``prompts``.

Design rule for everything in this file: **degrade loudly, never silently.**

A parser that quietly invents a default turns a model failure into a data
failure. Every function here returns the best object it can *and* flags
``parse_ok=False`` with a human-readable reason, so §5's structure-compliance
gate measures something real rather than measuring how forgiving the parser is.

Two fallbacks that are deliberately *not* treated as success:

* a plan written as a numbered list instead of ``STEP:`` lines still yields
  steps, but ``parse_ok`` is false — the schema was not followed;
* an unreadable reflection verdict resolves to **fail**, never pass (§5).

Tamil affirmatives/negatives are accepted alongside English ones, because a
Tamil-first model answering "ஆம்" to a yes/no field is the model working, not
the model failing.
"""
from __future__ import annotations

import re

from domains.core.prompts import (
    FIELD_INTENT,
    FIELD_REASON,
    FIELD_REQUIRES_PLANNING,
    FIELD_STEP,
    FIELD_SUB_PROBLEM,
    FIELD_VERDICT,
    MAX_PLAN_STEPS,
    STEP_SEPARATOR,
)
from domains.core.schemas import Plan, PlanStep, ReasoningTrace, Reflection

_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_FIELD = re.compile(r"^([A-Za-z_]+)\s*:\s*(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.+)$")

_TRUE_WORDS = {
    "yes", "y", "true", "1", "yeah", "yep", "required", "needed",
    "ஆம்", "ஆமாம்", "சரி", "தேவை", "aam", "aamaam", "sari", "thevai",
}
_FALSE_WORDS = {
    "no", "n", "false", "0", "nope", "none", "not required", "not needed",
    "இல்லை", "வேண்டாம்", "illai", "illa", "vendam",
}
_PASS_WORDS = {"pass", "passed", "ok", "okay", "good", "accept", "accepted", "சரி", "நல்லது"}
_FAIL_WORDS = {"fail", "failed", "reject", "rejected", "bad", "no", "தவறு", "இல்லை"}


def _clean(line: str) -> str:
    """Strip bullets, numbering and stray markdown emphasis from one line."""
    line = _BULLET.sub("", line.strip())
    return line.strip("*_ ").strip()


def _fields(text: str) -> list[tuple[str, str]]:
    """Extract ``KEY: value`` pairs, key upper-cased, in document order."""
    out: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = _clean(raw_line)
        if not line:
            continue
        match = _FIELD.match(line)
        if match:
            out.append((match.group(1).upper(), match.group(2).strip()))
    return out


def _first(fields: list[tuple[str, str]], key: str) -> str | None:
    for k, v in fields:
        if k == key and v:
            return v
    return None


def _all(fields: list[tuple[str, str]], key: str) -> list[str]:
    return [v for k, v in fields if k == key and v]


def _truthy(value: str) -> bool | None:
    """Map a yes/no field to a bool, or ``None`` when it is not recognisable."""
    token = value.strip().strip(".!।").lower()
    if token in _TRUE_WORDS:
        return True
    if token in _FALSE_WORDS:
        return False
    # tolerate "yes — it needs steps" style answers by looking at the first word
    head = token.split()[0] if token.split() else ""
    if head in _TRUE_WORDS:
        return True
    if head in _FALSE_WORDS:
        return False
    return None


def parse_reasoning(text: str) -> ReasoningTrace:
    """Parse a ``[REASON]`` completion into a :class:`ReasoningTrace`.

    On an unreadable ``REQUIRES_PLANNING`` the trace falls back to ``False``.
    That is the cost-safe default — most conversational turns should never reach
    the Planning Agent (§6) — but it is always recorded as a parse error so a
    model that has stopped emitting the field shows up in the gate instead of
    quietly halving the pipeline.
    """
    fields = _fields(text)
    errors: list[str] = []

    intent = _first(fields, FIELD_INTENT)
    if intent is None:
        errors.append(f"missing {FIELD_INTENT}")
        intent = ""

    sub_problems = tuple(_all(fields, FIELD_SUB_PROBLEM))
    if not sub_problems:
        errors.append(f"no {FIELD_SUB_PROBLEM} lines")

    raw_flag = _first(fields, FIELD_REQUIRES_PLANNING)
    if raw_flag is None:
        errors.append(f"missing {FIELD_REQUIRES_PLANNING}")
        requires_planning = False
    else:
        parsed = _truthy(raw_flag)
        if parsed is None:
            errors.append(f"unreadable {FIELD_REQUIRES_PLANNING}: {raw_flag!r}")
            requires_planning = False
        else:
            requires_planning = parsed

    return ReasoningTrace(
        intent=intent,
        sub_problems=sub_problems,
        requires_planning=requires_planning,
        raw=text,
        parse_ok=not errors,
        parse_errors=tuple(errors),
    )


def parse_plan(text: str) -> Plan:
    """Parse a ``[PLAN]`` completion into an ordered :class:`Plan`."""
    fields = _fields(text)
    errors: list[str] = []

    raw_steps = _all(fields, FIELD_STEP)
    if not raw_steps:
        # Fallback: the model wrote a numbered list instead of STEP: lines. Usable,
        # but the schema was not followed — do not report it as compliant.
        raw_steps = [
            m.group(1).strip()
            for m in (_NUMBERED.match(line) for line in text.splitlines())
            if m
        ]
        if raw_steps:
            errors.append(f"no {FIELD_STEP} lines; recovered from numbered list")
        else:
            errors.append(f"no {FIELD_STEP} lines and no numbered list to recover from")

    steps: list[PlanStep] = []
    for i, raw in enumerate(raw_steps[:MAX_PLAN_STEPS], start=1):
        action, _, detail = raw.partition(STEP_SEPARATOR)
        steps.append(PlanStep(order=i, action=action.strip(), detail=detail.strip()))

    if len(raw_steps) > MAX_PLAN_STEPS:
        errors.append(f"truncated {len(raw_steps)} steps to {MAX_PLAN_STEPS}")

    return Plan(
        steps=tuple(steps),
        raw=text,
        parse_ok=not errors,
        parse_errors=tuple(errors),
    )


def parse_reflection(text: str) -> Reflection:
    """Parse a ``[REFLECT]`` completion into a :class:`Reflection`.

    Fails closed: a missing or unreadable verdict resolves to **fail**, never
    pass (§5). Only failures have to justify themselves — a bare
    ``VERDICT: pass`` is well-formed, a ``VERDICT: fail`` with no ``REASON``
    line is not.
    """
    fields = _fields(text)
    errors: list[str] = []

    reasons = tuple(_all(fields, FIELD_REASON))
    raw_verdict = _first(fields, FIELD_VERDICT)

    if raw_verdict is None:
        errors.append(f"missing {FIELD_VERDICT}; failing closed")
        return Reflection(
            passed=False,
            reasons=reasons or ("reflection output could not be parsed",),
            raw=text,
            parse_ok=False,
            parse_errors=tuple(errors),
        )

    token = raw_verdict.strip().strip(".!।").lower()
    head = token.split()[0] if token.split() else ""
    if token in _PASS_WORDS or head in _PASS_WORDS:
        passed = True
    elif token in _FAIL_WORDS or head in _FAIL_WORDS:
        passed = False
    else:
        errors.append(f"unreadable {FIELD_VERDICT}: {raw_verdict!r}; failing closed")
        passed = False

    if not passed and not reasons:
        errors.append(f"fail verdict with no {FIELD_REASON} line")
        reasons = ("no reason given",)

    return Reflection(
        passed=passed,
        reasons=reasons,
        raw=text,
        parse_ok=not errors,
        parse_errors=tuple(errors),
    )
