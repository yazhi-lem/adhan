"""Pure-python tests for the Core Domain kernel (no model, no network, no JAX).

Run: PYTHONPATH=src python -m domains.core.test_core_domain

The interesting assertions here are the ones about *cost* and *fail-closed
behaviour*, not the happy path:

  * the Planning Agent must not be called when ``requires_planning`` is false —
    that gate is worth three of every four model calls (§6);
  * unparseable reflection must resolve to fail, never pass (§5);
  * a retry that rescues a rejected draft must yield a DPO pair (§4.1);
  * memory writes must be refused without consent (§4.3).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from domains.core import (  # noqa: E402
    AgentRole,
    CoreOrchestrator,
    EchoClient,
    InMemoryStore,
    Interaction,
    OrchestratorConfig,
    PersonalizationProfile,
    UserContext,
    parse_plan,
    parse_reasoning,
    parse_reflection,
)
from domains.core.model_client import ModelClientError  # noqa: E402


class ScriptedClient:
    """Returns a queued completion per role; counts calls. Test double only."""

    def __init__(self, **by_role: list[str]):
        self.scripts = {AgentRole[k.upper()]: list(v) for k, v in by_role.items()}
        self.calls: list[tuple[AgentRole, str]] = []

    def generate(self, prompt, *, role, max_tokens=256, temperature=0.7, stop=None):
        self.calls.append((role, prompt))
        queue = self.scripts.get(role)
        if not queue:
            raise ModelClientError(f"no scripted output left for {role}")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def count(self, role: AgentRole) -> int:
        return sum(1 for r, _ in self.calls if r is role)


def _profile(**kw) -> PersonalizationProfile:
    return PersonalizationProfile(user_id=kw.pop("user_id", "u1"), **kw)


def _ctx(history=()) -> UserContext:
    return UserContext(
        user_id="u1",
        profile=_profile(),
        previous_interactions=[Interaction(q, a) for q, a in history],
    )


# --- parsing ---------------------------------------------------------------


def test_parse_reasoning_well_formed():
    trace = parse_reasoning(
        "INTENT: find the scheme\n"
        "SUB_PROBLEM: check eligibility\n"
        "SUB_PROBLEM: list documents\n"
        "REQUIRES_PLANNING: yes"
    )
    assert trace.parse_ok, trace.parse_errors
    assert trace.intent == "find the scheme"
    assert len(trace.sub_problems) == 2
    assert trace.requires_planning is True
    print("  parse_reasoning: well-formed trace OK")


def test_parse_reasoning_accepts_tamil_yes_no():
    yes = parse_reasoning("INTENT: x\nSUB_PROBLEM: y\nREQUIRES_PLANNING: ஆம்")
    no = parse_reasoning("INTENT: x\nSUB_PROBLEM: y\nREQUIRES_PLANNING: இல்லை")
    assert yes.requires_planning is True and yes.parse_ok
    assert no.requires_planning is False and no.parse_ok
    print("  parse_reasoning: Tamil ஆம்/இல்லை accepted OK")


def test_parse_reasoning_degrades_loudly():
    trace = parse_reasoning("I think the user wants to know about schemes.")
    assert not trace.parse_ok, "free prose must not count as schema-compliant"
    assert trace.requires_planning is False, "cost-safe default on unparseable flag"
    assert trace.parse_errors, "failure must carry a reason"
    print(f"  parse_reasoning: unstructured output flagged ({len(trace.parse_errors)} errors) OK")


def test_parse_plan_and_numbered_fallback():
    plan = parse_plan("STEP: check age | must be over 18\nSTEP: collect proof")
    assert plan.parse_ok and len(plan.steps) == 2
    assert plan.steps[0].detail == "must be over 18"
    assert plan.steps[1].order == 2 and plan.steps[1].detail == ""

    recovered = parse_plan("1. check age\n2. collect proof")
    assert len(recovered.steps) == 2, "numbered lists should still be usable"
    assert not recovered.parse_ok, "but a numbered list is not the STEP: schema"
    print("  parse_plan: STEP lines OK, numbered fallback usable-but-non-compliant OK")


def test_parse_reflection_fails_closed():
    passed = parse_reflection("VERDICT: pass")
    assert passed.passed and passed.parse_ok

    failed = parse_reflection("VERDICT: fail\nREASON: ignores the question")
    assert not failed.passed and failed.reasons == ("ignores the question",)

    for junk in ("the answer looks fine to me", "", "VERDICT: maybe"):
        verdict = parse_reflection(junk)
        assert not verdict.passed, f"unparseable reflection must fail closed: {junk!r}"
        assert not verdict.parse_ok
    print("  parse_reflection: pass/fail OK, unparseable fails closed OK")


# --- routing / cost --------------------------------------------------------


def test_planning_skipped_when_not_required():
    client = EchoClient(requires_planning=False)
    result = CoreOrchestrator(client).run_turn(_ctx(), "வணக்கம்")
    assert client.count(AgentRole.PLAN) == 0, "planning must not run for a simple turn"
    assert result.plan.skipped
    assert result.metrics()["planning_invoked"] is False
    print(f"  routing: simple turn used {result.metrics()['model_calls']} model calls, no PLAN OK")


def test_planning_runs_when_required():
    client = EchoClient(requires_planning=True)
    result = CoreOrchestrator(client).run_turn(_ctx(), "How do I apply for the scheme?")
    assert client.count(AgentRole.PLAN) == 1
    assert not result.plan.skipped and len(result.plan.steps) == 2
    assert result.metrics()["planning_invoked"] is True
    print("  routing: multi-step turn invoked PLAN once OK")


def test_context_agent_skips_cold_conversation():
    cold = EchoClient()
    CoreOrchestrator(cold).run_turn(_ctx(), "வணக்கம்")
    assert cold.count(AgentRole.CONTEXT) == 0, "nothing to summarise on a first turn"

    warm = EchoClient()
    CoreOrchestrator(warm).run_turn(_ctx(history=[("hi", "hello")]), "and then?")
    assert warm.count(AgentRole.CONTEXT) == 1
    print("  routing: CONTEXT skipped when cold, called when there is history OK")


# --- reflection / retry / DPO ---------------------------------------------


def test_retry_on_failed_reflection_yields_preference_pair():
    client = ScriptedClient(
        reason=["INTENT: answer\nSUB_PROBLEM: a\nREQUIRES_PLANNING: no"],
        respond=["first draft", "second draft"],
        reflect=["VERDICT: fail\nREASON: too vague", "VERDICT: pass"],
    )
    result = CoreOrchestrator(client).run_turn(_ctx(), "why?")

    assert result.attempts == 2
    assert result.text == "second draft"
    assert result.response.incorporated_reflection, "retry must be told why it is retrying"

    pair = result.preference_pair
    assert pair is not None, "a rescued draft is a labelled DPO pair (§4.1)"
    assert (pair.chosen, pair.rejected) == ("second draft", "first draft")
    assert pair.reason == "too vague"
    print("  reflection: failed draft retried, preference pair harvested OK")


def test_retry_gives_up_after_max_attempts():
    client = ScriptedClient(
        reason=["INTENT: answer\nSUB_PROBLEM: a\nREQUIRES_PLANNING: no"],
        respond=["draft"],
        reflect=["VERDICT: fail\nREASON: still wrong"],
    )
    result = CoreOrchestrator(client).run_turn(_ctx(), "why?")
    assert result.attempts == 2, "one draft plus one retry, then stop"
    assert not result.reflection.passed
    assert result.preference_pair is None, "never rescued, so there is no chosen side"
    print("  reflection: gives up after max attempts, no bogus pair OK")


def test_reflection_can_be_disabled():
    client = EchoClient()
    orch = CoreOrchestrator(client, OrchestratorConfig(reflect=False))
    result = orch.run_turn(_ctx(), "வணக்கம்")
    assert client.count(AgentRole.REFLECT) == 0
    assert result.metrics()["reflection_passed"] is None, "disabled is not a pass"
    print("  reflection: disabled records a non-verdict, not a pass OK")


# --- pass-through / metrics ------------------------------------------------


def test_passthrough_makes_exactly_one_call():
    client = EchoClient()
    result = CoreOrchestrator(client).passthrough(_ctx(), "வணக்கம்")
    assert len(client.calls) == 1 and client.calls[0][0] is AgentRole.RESPOND
    metrics = result.metrics()
    assert metrics["model_calls"] == 1
    assert metrics["structure_compliance"] is None, "no structured call was made"
    print("  passthrough: exactly one RESPOND call OK")


def test_control_tokens_are_prepended():
    client = EchoClient(requires_planning=True)
    CoreOrchestrator(client).run_turn(_ctx(history=[("hi", "hello")]), "how do I apply?")
    seen = {role: prompt for role, prompt in client.calls}
    for role, prompt in seen.items():
        assert prompt.startswith(role.control_token), f"{role} prompt missing control token"
    assert len(seen) == 5, f"expected all five postures, saw {sorted(r.name for r in seen)}"
    print("  prompts: all five control tokens present OK")


def test_metrics_shape_and_compliance():
    client = EchoClient(requires_planning=True)
    result = CoreOrchestrator(client).run_turn(_ctx(history=[("hi", "hello")]), "how?")
    metrics = result.metrics()
    assert metrics["model_calls"] == 5
    assert metrics["structure_compliance"] == 1.0, "EchoClient emits schema-valid output"
    assert set(metrics["latency_ms_by_role"]) == {r.value for r in AgentRole}
    assert metrics["latency_ms_total"] >= 0.0
    print(f"  metrics: {metrics['model_calls']} calls, compliance={metrics['structure_compliance']} OK")


def test_spans_emitted_to_callback():
    seen = []
    orch = CoreOrchestrator(EchoClient(), on_span=seen.append)
    orch.run_turn(_ctx(), "வணக்கம்")
    assert len(seen) == 5, "every posture emits a span, including skipped ones"
    assert sum(1 for s in seen if s.model_call) == 3, "cold + no-plan turn is three calls"
    print("  tracing: on_span callback fired for all five postures OK")


# --- memory / consent ------------------------------------------------------


def test_memory_requires_consent():
    store = InMemoryStore()
    profile = _profile()
    store.save_profile(profile)

    assert store.append_interaction("u1", Interaction("q", "a"), consented=False) is False
    assert store.load("u1").previous_interactions == []
    assert store.declined_writes == 1

    assert store.append_interaction("u1", Interaction("q", "a"), consented=True) is True
    assert len(store.load("u1").previous_interactions) == 1
    print("  memory: writes refused without DPDP consent OK")


def test_preference_pairs_are_consent_gated():
    client = ScriptedClient(
        reason=["INTENT: answer\nSUB_PROBLEM: a\nREQUIRES_PLANNING: no"],
        respond=["first draft", "second draft"],
        reflect=["VERDICT: fail\nREASON: too vague", "VERDICT: pass"],
    )
    result = CoreOrchestrator(client).run_turn(_ctx(), "why?")
    store = InMemoryStore()
    assert store.record_preference_pair(result.preference_pair, consented=False) is False
    assert store.preference_pairs == ()
    assert store.record_preference_pair(result.preference_pair, consented=True) is True
    assert len(store.preference_pairs) == 1
    print("  memory: DPO pairs gated on consent too OK")


def main():
    print("core domain tests:")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
