"""Core Domain — the Yazhi reasoning kernel.

An orchestration shell, not a model. It routes five agent postures (context,
reasoning, planning, reflection, generation) through a **single** Adhan
checkpoint, role-tagged by a control token rather than split across adapters.

Phase 1 scope: module structure, schemas, orchestrator skeleton and a
single-agent pass-through, wired to the existing Adhan checkpoint as-is to
validate plumbing. No role-tagged training has happened yet, so the structured
postures are expected to parse poorly — ``TurnResult.metrics()`` reports that as
``structure_compliance``, which is the baseline the first training run has to
beat.

Quick start (no model, no GPU)::

    from domains.core import CoreOrchestrator, EchoClient, UserContext
    from domains.core import PersonalizationProfile

    profile = PersonalizationProfile(user_id="u1")
    orch = CoreOrchestrator(EchoClient())
    result = orch.run_turn(UserContext("u1", profile), "வணக்கம்")
    print(result.text, result.metrics())

See docs/CORE_DOMAIN_ADHAN_INTEGRATION.md.
"""
from domains.core.agents import (
    ContextAgent,
    PlanningAgent,
    ReasoningAgent,
    ReflectionAgent,
    ResponseGenerator,
    ROLE_SAMPLING,
)
from domains.core.memory import InMemoryStore, MemoryStore
from domains.core.model_client import (
    AdhanCheckpointClient,
    EchoClient,
    ModelClient,
    ModelClientError,
    OllamaClient,
    VLLMClient,
)
from domains.core.orchestrator import CoreOrchestrator, OrchestratorConfig
from domains.core.parsing import parse_plan, parse_reasoning, parse_reflection
from domains.core.schemas import (
    AgentRole,
    ContextBrief,
    Interaction,
    Language,
    PersonalizationProfile,
    Plan,
    PlanStep,
    PreferencePair,
    ReasoningTrace,
    Reflection,
    Response,
    Span,
    TurnResult,
    UserContext,
)

__all__ = [
    # orchestration
    "CoreOrchestrator",
    "OrchestratorConfig",
    # agents
    "ContextAgent",
    "ReasoningAgent",
    "PlanningAgent",
    "ReflectionAgent",
    "ResponseGenerator",
    "ROLE_SAMPLING",
    # backends
    "ModelClient",
    "ModelClientError",
    "EchoClient",
    "OllamaClient",
    "VLLMClient",
    "AdhanCheckpointClient",
    # memory
    "MemoryStore",
    "InMemoryStore",
    # schemas
    "AgentRole",
    "Language",
    "PersonalizationProfile",
    "Interaction",
    "UserContext",
    "ContextBrief",
    "ReasoningTrace",
    "Plan",
    "PlanStep",
    "Reflection",
    "Response",
    "PreferencePair",
    "Span",
    "TurnResult",
    # parsing
    "parse_reasoning",
    "parse_plan",
    "parse_reflection",
]
