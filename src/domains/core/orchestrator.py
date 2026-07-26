"""The Core Orchestrator — five postures, one model, one turn.

Flow::

    context ──▶ reason ──▶ [plan?] ──▶ respond ──▶ reflect ──┐
                                          ▲                  │
                                          └── retry on fail ─┘

The branch marked ``[plan?]`` is the part worth getting right early. Most
conversational turns — greetings, small talk, single-fact questions — must never
reach the Planning Agent. That gate is the difference between one Adhan call per
turn and four, and B2G contract margins depend on inference cost staying low
(§6). ``metrics()`` reports ``planning_invoked`` on every turn so the ratio is
measurable from day one rather than reconstructed later.

The orchestrator takes exactly one :class:`ModelClient`. That is Option A (§3)
made structural: five postures, one role-tagged checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Callable, Sequence

from domains.core.agents import (
    ContextAgent,
    PlanningAgent,
    ReasoningAgent,
    ReflectionAgent,
    ResponseGenerator,
)
from domains.core.model_client import ModelClient
from domains.core.schemas import (
    AgentRole,
    ContextBrief,
    Plan,
    PreferencePair,
    ReasoningTrace,
    Reflection,
    Response,
    Span,
    TurnResult,
    UserContext,
)


@dataclass
class OrchestratorConfig:
    """Knobs that change cost or safety, and nothing else."""

    #: One draft plus one reflection-driven retry. More than two attempts
    #: multiplies cost for a model that has not been trained to use the notes yet.
    max_response_attempts: int = 2
    reflect: bool = True
    #: Force the Planning Agent on regardless of ``requires_planning`` — for
    #: measuring what the routing gate is actually saving, not for production.
    always_plan: bool = False
    #: Harvest a (chosen, rejected) pair when a retry rescues a failed draft.
    harvest_preference_pairs: bool = True


class _MeteredClient:
    """Wraps the real client so every call is timed and sized automatically.

    Keeps the agents free of tracing code while still producing one accurate
    :class:`Span` per model call, including the extra RESPOND call on a retry.
    """

    def __init__(self, inner: ModelClient, on_span: Callable[[Span], None] | None = None):
        self._inner = inner
        self._on_span = on_span
        self._pending: list[Span] = []

    def generate(self, prompt: str, *, role: AgentRole, **kwargs) -> str:
        started = perf_counter()
        output = self._inner.generate(prompt, role=role, **kwargs)
        self._pending.append(
            Span(
                role=role,
                latency_ms=(perf_counter() - started) * 1000.0,
                prompt_chars=len(prompt),
                output_chars=len(output),
            )
        )
        return output

    def take(self, parse_ok: bool = True) -> Span:
        """Pop the span for the call that just happened, stamping its parse result."""
        span = self._pending.pop()
        if not parse_ok:
            span = replace(span, parse_ok=False)
        if self._on_span is not None:
            self._on_span(span)
        return span

    def skipped(self, role: AgentRole) -> Span:
        span = Span(role=role, skipped=True)
        if self._on_span is not None:
            self._on_span(span)
        return span


@dataclass
class CoreOrchestrator:
    """Runs one turn through the five postures."""

    client: ModelClient
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    #: Called with each :class:`Span` as it completes — the seam LangFuse plugs
    #: into (§6). Left as a plain callback so the kernel takes no new dependency.
    on_span: Callable[[Span], None] | None = None

    def __post_init__(self) -> None:
        self._meter = _MeteredClient(self.client, self.on_span)
        self.context_agent = ContextAgent(self._meter)
        self.reasoning_agent = ReasoningAgent(self._meter)
        self.planning_agent = PlanningAgent(self._meter)
        self.response_generator = ResponseGenerator(self._meter)
        self.reflection_agent = ReflectionAgent(self._meter)

    # -- the full pipeline --------------------------------------------------

    def run_turn(
        self,
        ctx: UserContext,
        query: str,
        grounding: Sequence[str] = (),
    ) -> TurnResult:
        profile = ctx.profile
        spans: list[Span] = []

        brief = self.context_agent.run(ctx, query)
        spans.append(
            self._meter.skipped(AgentRole.CONTEXT) if brief.skipped else self._meter.take()
        )

        reasoning = self.reasoning_agent.run(query, brief, profile)
        spans.append(self._meter.take(reasoning.parse_ok))

        if reasoning.requires_planning or self.config.always_plan:
            plan = self.planning_agent.run(query, reasoning, brief, profile)
            spans.append(self._meter.take(plan.parse_ok))
        else:
            plan = Plan.not_needed()
            spans.append(self._meter.skipped(AgentRole.PLAN))

        response, reflection, attempts, pair = self._respond_and_reflect(
            query, brief, reasoning, plan, profile, grounding, spans
        )

        return TurnResult(
            query=query,
            brief=brief,
            reasoning=reasoning,
            plan=plan,
            response=response,
            reflection=reflection,
            spans=tuple(spans),
            attempts=attempts,
            preference_pair=pair,
        )

    def _respond_and_reflect(
        self,
        query,
        brief,
        reasoning,
        plan,
        profile,
        grounding,
        spans: list[Span],
    ) -> tuple[Response, Reflection, int, PreferencePair | None]:
        attempts = 0
        rejected: Response | None = None
        rejection: Reflection | None = None
        response = Response(text="")
        reflection = Reflection.not_run()

        while attempts < max(1, self.config.max_response_attempts):
            attempts += 1
            response = self.response_generator.run(
                query, brief, reasoning, plan, profile,
                reflection=rejection, previous_draft=rejected,
            )
            spans.append(self._meter.take())

            if not self.config.reflect:
                reflection = Reflection.not_run()
                spans.append(self._meter.skipped(AgentRole.REFLECT))
                break

            reflection = self.reflection_agent.run(
                query, response, brief, reasoning, profile, grounding
            )
            spans.append(self._meter.take(reflection.parse_ok))

            if reflection.passed or attempts >= max(1, self.config.max_response_attempts):
                break

            # Keep the failed draft and the critique that killed it: together they
            # are the retry's input and, if the retry succeeds, a DPO pair.
            rejected, rejection = response, reflection

        pair = None
        if (
            self.config.harvest_preference_pairs
            and rejected is not None
            and rejection is not None
            and reflection.passed
            and not reflection.skipped
        ):
            pair = PreferencePair(
                prompt=query,
                chosen=response.text,
                rejected=rejected.text,
                reason="; ".join(rejection.reasons),
            )

        return response, reflection, attempts, pair

    # -- Phase 1 plumbing check ---------------------------------------------

    def passthrough(self, ctx: UserContext, query: str) -> TurnResult:
        """One model call, no pipeline — the Phase 1 "does the wire work" path.

        Every posture except RESPOND is recorded as skipped, so a passthrough
        turn is visibly distinguishable from a pipeline turn in the traces rather
        than looking like a pipeline that mysteriously made one call.
        """
        brief = ContextBrief(text="", n_interactions=0, skipped=True)
        reasoning = ReasoningTrace(intent=query, requires_planning=False)
        plan = Plan.not_needed()

        spans = [
            self._meter.skipped(AgentRole.CONTEXT),
            self._meter.skipped(AgentRole.REASON),
            self._meter.skipped(AgentRole.PLAN),
        ]
        response = self.response_generator.run(query, brief, reasoning, plan, ctx.profile)
        spans.append(self._meter.take())
        spans.append(self._meter.skipped(AgentRole.REFLECT))

        return TurnResult(
            query=query,
            brief=brief,
            reasoning=reasoning,
            plan=plan,
            response=response,
            reflection=Reflection.not_run(),
            spans=tuple(spans),
            attempts=1,
        )
