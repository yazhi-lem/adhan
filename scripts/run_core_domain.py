#!/usr/bin/env python3
"""Run one turn through the Core Domain kernel — the Phase 1 plumbing check.

Examples::

    # No model at all: prove the pipeline, the routing and the traces work.
    python scripts/run_core_domain.py --query "வணக்கம்" --trace

    # Prove the wire to a local Adhan checkpoint works, one call, no pipeline.
    python scripts/run_core_domain.py --client adhan --mode passthrough \\
        --config src/adhan_slm/configs/adhan_slm_tiny.yaml \\
        --checkpoint-dir models/adhan_slm/checkpoints \\
        --tokenizer-dir models/adhan_slm/tokenizer \\
        --query "வணக்கம்"

    # Dev-only: iterate on prompts.py against Ollama before a training run.
    python scripts/run_core_domain.py --client ollama --model qwen3:4b \\
        --query "How do I apply for the scheme?" --trace

    # Production shape: Adhan on vLLM.
    python scripts/run_core_domain.py --client vllm --base-url http://localhost:8000 \\
        --model adhan --query "How do I apply for the scheme?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domains.core import (  # noqa: E402
    AdhanCheckpointClient,
    CoreOrchestrator,
    EchoClient,
    Interaction,
    Language,
    OllamaClient,
    OrchestratorConfig,
    PersonalizationProfile,
    UserContext,
    VLLMClient,
)


def build_client(args):
    if args.client == "echo":
        return EchoClient(requires_planning=args.echo_requires_planning)
    if args.client == "ollama":
        return OllamaClient(model=args.model or "qwen3:4b", host=args.host)
    if args.client == "vllm":
        return VLLMClient(model=args.model or "adhan", base_url=args.base_url)
    if args.client == "adhan":
        missing = [
            name
            for name, value in (
                ("--config", args.config),
                ("--checkpoint-dir", args.checkpoint_dir),
                ("--tokenizer-dir", args.tokenizer_dir),
            )
            if not value
        ]
        if missing:
            raise SystemExit(f"--client adhan needs: {', '.join(missing)}")
        return AdhanCheckpointClient(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            tokenizer_dir=args.tokenizer_dir,
            tokenizer_kind=args.tokenizer_kind,
        )
    raise SystemExit(f"unknown client: {args.client}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", required=True, help="The user's question for this turn")
    parser.add_argument(
        "--mode",
        choices=["full", "passthrough"],
        default="full",
        help="full = five postures; passthrough = one RESPOND call (plumbing check)",
    )

    parser.add_argument(
        "--client", choices=["echo", "ollama", "vllm", "adhan"], default="echo"
    )
    parser.add_argument("--model", help="Model name for ollama/vllm")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--config", help="Adhan training YAML (--client adhan)")
    parser.add_argument("--checkpoint-dir", help="Orbax checkpoint dir (--client adhan)")
    parser.add_argument("--tokenizer-dir", help="vocab.json + merges.txt (--client adhan)")
    parser.add_argument(
        "--tokenizer-kind", choices=["swaram", "aksharam"], default="swaram"
    )
    parser.add_argument(
        "--echo-requires-planning",
        action="store_true",
        help="Make the echo client ask for planning, to exercise the PLAN posture",
    )

    parser.add_argument("--user-id", default="dev-user")
    parser.add_argument("--persona", default="yazh")
    parser.add_argument("--register", default="casual", choices=["casual", "formal"])
    parser.add_argument(
        "--language",
        action="append",
        choices=[lang.value for lang in Language],
        help="Answer language; repeatable (default: ta)",
    )
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        metavar="Q::A",
        help="Seed a prior turn as 'question::answer'; repeatable",
    )
    parser.add_argument(
        "--grounding",
        action="append",
        default=[],
        help="A retrieved source line the reflection agent must check against; repeatable",
    )

    parser.add_argument("--no-reflect", action="store_true", help="Skip the Reflection Agent")
    parser.add_argument(
        "--always-plan",
        action="store_true",
        help="Force the Planning Agent on, to measure what the routing gate saves",
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--trace", action="store_true", help="Print the per-agent spans")
    parser.add_argument("--json", action="store_true", help="Print the turn as JSON")

    args = parser.parse_args()

    languages = tuple(Language(v) for v in (args.language or ["ta"]))
    profile = PersonalizationProfile(
        user_id=args.user_id,
        languages=languages,
        register=args.register,
        persona=args.persona,
    )
    history = []
    for item in args.history:
        question, _, answer = item.partition("::")
        history.append(Interaction(query=question.strip(), response=answer.strip()))
    ctx = UserContext(user_id=args.user_id, profile=profile, previous_interactions=history)

    orchestrator = CoreOrchestrator(
        build_client(args),
        OrchestratorConfig(
            max_response_attempts=args.max_attempts,
            reflect=not args.no_reflect,
            always_plan=args.always_plan,
        ),
    )

    if args.mode == "passthrough":
        result = orchestrator.passthrough(ctx, args.query)
    else:
        result = orchestrator.run_turn(ctx, args.query, grounding=args.grounding)

    if args.json:
        print(
            json.dumps(
                {
                    "query": result.query,
                    "response": result.text,
                    "intent": result.reasoning.intent,
                    "requires_planning": result.reasoning.requires_planning,
                    "plan": [
                        {"order": s.order, "action": s.action, "detail": s.detail}
                        for s in result.plan.steps
                    ],
                    "reflection": {
                        "passed": result.reflection.passed,
                        "reasons": list(result.reflection.reasons),
                        "skipped": result.reflection.skipped,
                    },
                    "metrics": result.metrics(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"\nQ: {result.query}")
    print(f"A: {result.text}\n")

    if result.reasoning.intent:
        print(f"intent            : {result.reasoning.intent}")
    print(f"requires_planning : {result.reasoning.requires_planning}")
    if result.plan.steps:
        print("plan              :")
        for step in result.plan.steps:
            suffix = f" — {step.detail}" if step.detail else ""
            print(f"  {step.order}. {step.action}{suffix}")
    if not result.reflection.skipped:
        verdict = "pass" if result.reflection.passed else "fail"
        print(f"reflection        : {verdict}")
        for reason in result.reflection.reasons:
            print(f"  - {reason}")
    if result.preference_pair:
        print("dpo pair          : harvested (rejected draft rescued by retry)")

    if args.trace:
        print("\nspans:")
        for span in result.spans:
            state = "skipped" if span.skipped else f"{span.latency_ms:7.1f} ms"
            compliance = "" if span.parse_ok else "  PARSE-FAIL"
            print(f"  {span.role.name:<8} {state}{compliance}")

    print("\nmetrics:")
    for key, value in result.metrics().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
