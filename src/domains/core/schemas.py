"""Typed schemas for the Core Domain reasoning kernel.

Pure dataclasses, standard library only — the kernel must import cleanly on a
machine with no JAX/torch stack, because the whole point of Phase 1 is validating
plumbing before any training run happens.

Two design notes worth keeping in mind when reading this file:

* Every structured agent output carries ``parse_ok``. That is not decoration —
  it is how the "reasoning-trace structure compliance" eval gate (§5) gets
  measured off live traffic instead of a separate offline harness.
* ``PreferencePair`` exists because the Reflection Agent's pass/fail *is* a DPO
  signal (§4.1). A rejected draft plus the reply that passed after it is a
  labelled (chosen, rejected) pair, produced for free by ordinary usage.

See docs/CORE_DOMAIN_ADHAN_INTEGRATION.md §2, §4.1, §5.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AgentRole(Enum):
    """The five postures Core Domain asks Adhan to take.

    The control token is prepended to every prompt. This is the Option A shape
    from §3: one role-tagged adapter rather than four hot-swapped ones, so the
    role has to travel *inside* the prompt rather than in the serving route.
    """

    CONTEXT = "context"
    REASON = "reason"
    PLAN = "plan"
    REFLECT = "reflect"
    RESPOND = "respond"

    @property
    def control_token(self) -> str:
        return f"[{self.name}]"


#: Roles whose output must parse into a schema — the ones the structure-compliance
#: gate applies to. CONTEXT and RESPOND are free prose and are exempt by design.
STRUCTURED_ROLES = (AgentRole.REASON, AgentRole.PLAN, AgentRole.REFLECT)


class Language(Enum):
    TAMIL = "ta"
    ENGLISH = "en"
    TANGLISH = "tanglish"


@dataclass(frozen=True)
class PersonalizationProfile:
    """Per-user preferences a persona layers on top of the kernel.

    ``persona`` is the thin-persona name (yazh, annachi, tamil-gov, tamil-legal).
    ``register`` matters more than it looks: the gov/legal verticals need formal
    Tamil, and that is a different register from conversational Yazh.
    """

    user_id: str
    languages: tuple[Language, ...] = (Language.TAMIL,)
    register: str = "casual"
    persona: str = "yazh"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Interaction:
    """One completed turn, as retained for ``previous_interactions``."""

    query: str
    response: str
    ts: float = field(default_factory=time.time)


@dataclass
class UserContext:
    """Everything the Memory Manager hands the kernel at the start of a turn."""

    user_id: str
    profile: PersonalizationProfile
    previous_interactions: list[Interaction] = field(default_factory=list)

    def recent(self, n: int) -> list[Interaction]:
        return self.previous_interactions[-n:] if n > 0 else []


@dataclass(frozen=True)
class ContextBrief:
    """Context Agent output — compact prose, no schema to comply with."""

    text: str
    n_interactions: int = 0
    skipped: bool = False


@dataclass(frozen=True)
class ReasoningTrace:
    """Reasoning Agent output.

    ``requires_planning`` is the cost gate for the whole kernel (§6): it decides
    whether a turn costs one Adhan call or four.
    """

    intent: str
    sub_problems: tuple[str, ...] = ()
    requires_planning: bool = False
    raw: str = ""
    parse_ok: bool = True
    parse_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanStep:
    order: int
    action: str
    detail: str = ""


@dataclass(frozen=True)
class Plan:
    """Planning Agent output — the roadmap register."""

    steps: tuple[PlanStep, ...] = ()
    raw: str = ""
    parse_ok: bool = True
    parse_errors: tuple[str, ...] = ()
    skipped: bool = False

    @classmethod
    def not_needed(cls) -> "Plan":
        """The Planning Agent was never called because ``requires_planning`` was false."""
        return cls(skipped=True)

    def __bool__(self) -> bool:
        return bool(self.steps)


@dataclass(frozen=True)
class Reflection:
    """Reflection Agent output — pass/fail plus the reasons behind it.

    Unparseable output is recorded as a **fail**, never a pass. A reflection
    agent that rubber-stamps is worse than no reflection agent: it manufactures
    false confidence and poisons the DPO loop with bad labels (§5).
    """

    passed: bool
    reasons: tuple[str, ...] = ()
    raw: str = ""
    parse_ok: bool = True
    parse_errors: tuple[str, ...] = ()
    skipped: bool = False

    @classmethod
    def not_run(cls) -> "Reflection":
        """Reflection disabled for this turn — records as a non-verdict, not a pass."""
        return cls(passed=True, reasons=("reflection disabled",), skipped=True)


@dataclass(frozen=True)
class Response:
    """Response Generator output."""

    text: str
    incorporated_reflection: bool = False


@dataclass(frozen=True)
class PreferencePair:
    """A (chosen, rejected) pair harvested from a reflection-driven retry.

    Retention and any downstream training use is gated on DPDP consent (§4.3) —
    see ``domains.core.memory``.
    """

    prompt: str
    chosen: str
    rejected: str
    reason: str
    source: str = "core_domain.reflection"


@dataclass(frozen=True)
class Span:
    """One agent call, for per-agent latency tracing (§6)."""

    role: AgentRole
    latency_ms: float = 0.0
    prompt_chars: int = 0
    output_chars: int = 0
    parse_ok: bool = True
    skipped: bool = False

    @property
    def model_call(self) -> bool:
        return not self.skipped


@dataclass
class TurnResult:
    """Everything one Core Domain turn produced, including its own telemetry."""

    query: str
    brief: ContextBrief
    reasoning: ReasoningTrace
    plan: Plan
    response: Response
    reflection: Reflection
    spans: tuple[Span, ...] = ()
    attempts: int = 1
    preference_pair: Optional[PreferencePair] = None

    @property
    def text(self) -> str:
        return self.response.text

    def metrics(self) -> dict:
        """The per-turn payload to ship to LangFuse.

        Wired from day one on purpose: this is also the interference-monitoring
        data behind the Option A vs B decision in §3.
        """
        latency_by_role: dict[str, float] = {}
        for span in self.spans:
            latency_by_role[span.role.value] = round(
                latency_by_role.get(span.role.value, 0.0) + span.latency_ms, 3
            )

        structured = [
            s for s in self.spans if s.role in STRUCTURED_ROLES and not s.skipped
        ]
        compliance = (
            sum(1 for s in structured if s.parse_ok) / len(structured)
            if structured
            else None
        )

        return {
            "model_calls": sum(1 for s in self.spans if s.model_call),
            "attempts": self.attempts,
            "latency_ms_total": round(sum(s.latency_ms for s in self.spans), 3),
            "latency_ms_by_role": latency_by_role,
            "requires_planning": self.reasoning.requires_planning,
            "planning_invoked": not self.plan.skipped,
            "reflection_passed": None if self.reflection.skipped else self.reflection.passed,
            "structure_compliance": compliance,
            "produced_preference_pair": self.preference_pair is not None,
        }
