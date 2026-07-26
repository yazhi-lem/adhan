"""The five agent postures.

Agents are deliberately thin — build a prompt, call the one client with this
role's control token, parse the result. Timing and tracing live in the
orchestrator (via a metering wrapper around the client), so agents stay pure
functions of their inputs and can be tested against a scripted client.

**Per-role sampling is not cosmetic.** Structured postures want near-greedy
decoding: a creatively-worded ``REQUIRES_PLANNING`` field is a bug, and it is a
bug that costs three extra model calls per turn when it goes the wrong way.
Response generation wants enough temperature to sound like a person. Reflection
is coldest of all — a judge that samples its verdict is a judge with a random
false-pass rate, which is precisely what the calibration gate in §5 is watching
for.

See docs/CORE_DOMAIN_ADHAN_INTEGRATION.md §2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from domains.core import prompts
from domains.core.model_client import ModelClient
from domains.core.parsing import parse_plan, parse_reasoning, parse_reflection
from domains.core.schemas import (
    AgentRole,
    ContextBrief,
    PersonalizationProfile,
    Plan,
    ReasoningTrace,
    Reflection,
    Response,
    UserContext,
)


@dataclass(frozen=True)
class Sampling:
    max_tokens: int
    temperature: float


ROLE_SAMPLING: dict[AgentRole, Sampling] = {
    AgentRole.CONTEXT: Sampling(max_tokens=192, temperature=0.2),
    AgentRole.REASON: Sampling(max_tokens=256, temperature=0.2),
    AgentRole.PLAN: Sampling(max_tokens=320, temperature=0.2),
    AgentRole.RESPOND: Sampling(max_tokens=512, temperature=0.7),
    AgentRole.REFLECT: Sampling(max_tokens=256, temperature=0.1),
}


def _call(client: ModelClient, role: AgentRole, prompt: str) -> str:
    sampling = ROLE_SAMPLING[role]
    return client.generate(
        prompt,
        role=role,
        max_tokens=sampling.max_tokens,
        temperature=sampling.temperature,
    )


@dataclass
class ContextAgent:
    """Summarise ``previous_interactions`` into a compact brief.

    Skips the model entirely on a cold conversation. A first turn has nothing to
    summarise, and paying an Adhan call to be told so is the cheapest waste in
    the pipeline to remove.
    """

    client: ModelClient
    max_history: int = prompts.MAX_HISTORY_TURNS

    def run(self, ctx: UserContext, query: str) -> ContextBrief:
        if not ctx.previous_interactions:
            return ContextBrief(text="", n_interactions=0, skipped=True)
        text = _call(self.client, AgentRole.CONTEXT, prompts.context_prompt(ctx, query))
        return ContextBrief(text=text, n_interactions=len(ctx.recent(self.max_history)))


@dataclass
class ReasoningAgent:
    """Decompose intent and set the ``requires_planning`` flag."""

    client: ModelClient

    def run(
        self, query: str, brief: ContextBrief, profile: PersonalizationProfile
    ) -> ReasoningTrace:
        prompt = prompts.reasoning_prompt(query, brief, profile)
        return parse_reasoning(_call(self.client, AgentRole.REASON, prompt))


@dataclass
class PlanningAgent:
    """Convert reasoning into an ordered roadmap register."""

    client: ModelClient

    def run(
        self,
        query: str,
        reasoning: ReasoningTrace,
        brief: ContextBrief,
        profile: PersonalizationProfile,
    ) -> Plan:
        prompt = prompts.planning_prompt(query, reasoning, brief, profile)
        return parse_plan(_call(self.client, AgentRole.PLAN, prompt))


@dataclass
class ResponseGenerator:
    """Produce the final prose the user actually sees."""

    client: ModelClient

    def run(
        self,
        query: str,
        brief: ContextBrief,
        reasoning: ReasoningTrace,
        plan: Plan,
        profile: PersonalizationProfile,
        reflection: Reflection | None = None,
        previous_draft: Response | None = None,
    ) -> Response:
        prompt = prompts.response_prompt(
            query, brief, reasoning, plan, profile, reflection, previous_draft
        )
        text = _call(self.client, AgentRole.RESPOND, prompt)
        return Response(
            text=text,
            incorporated_reflection=reflection is not None and previous_draft is not None,
        )


@dataclass
class ReflectionAgent:
    """Self-critique: pass/fail plus reasons, checked against grounding.

    The highest-leverage posture in the kernel — its output is both the retry
    trigger and the DPO preference signal (§4.1).
    """

    client: ModelClient

    def run(
        self,
        query: str,
        response: Response,
        brief: ContextBrief,
        reasoning: ReasoningTrace,
        profile: PersonalizationProfile,
        grounding: Sequence[str] = (),
    ) -> Reflection:
        prompt = prompts.reflection_prompt(
            query, response, brief, reasoning, profile, grounding
        )
        return parse_reflection(_call(self.client, AgentRole.REFLECT, prompt))
