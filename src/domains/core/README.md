# Core Domain — the Yazhi reasoning kernel

An **orchestration shell, not a model**. Five agent postures, one Adhan
checkpoint, role-tagged by a control token.

Plan: [`docs/CORE_DOMAIN_ADHAN_INTEGRATION.md`](../../../docs/CORE_DOMAIN_ADHAN_INTEGRATION.md).
This package is **Phase 1** of it: structure, schemas, orchestrator skeleton and
a single-agent pass-through, wired to the existing Adhan checkpoint as-is. No
training has happened yet.

## The turn

```
context ──▶ reason ──▶ [plan?] ──▶ respond ──▶ reflect ──┐
                                      ▲                  │
                                      └── retry on fail ─┘
```

`[plan?]` is the cost gate. Greetings, small talk and single-fact questions must
never reach the Planning Agent — that branch is the difference between one Adhan
call per turn and four, and B2G margins depend on it. A cold conversation also
skips the Context Agent, because summarising an empty history is the cheapest
waste in the pipeline to remove.

Measured on the built-in stub: a simple turn costs **3** model calls, a
multi-step turn costs **5**.

## Try it without a model

```bash
PYTHONPATH=src python -m domains.core.test_core_domain          # 17 tests, no deps
python scripts/run_core_domain.py --query "வணக்கம்" --trace      # one turn, traced
```

```python
from domains.core import CoreOrchestrator, EchoClient, PersonalizationProfile, UserContext

profile = PersonalizationProfile(user_id="u1")
result = CoreOrchestrator(EchoClient()).run_turn(UserContext("u1", profile), "வணக்கம்")
print(result.text, result.metrics())
```

## Wiring it to Adhan

| Backend | Use |
|---|---|
| `EchoClient` | No model. Plumbing, routing and tests. |
| `AdhanCheckpointClient` | A local JAX checkpoint via `adhan_slm.inference`. The Phase 1 "as-is" wire. |
| `OllamaClient` | **Dev only** — iterating on `prompts.py` before a training run. |
| `VLLMClient` | Production. Adhan served on vLLM. |

```bash
python scripts/run_core_domain.py --client adhan --mode passthrough \
    --config src/adhan_slm/configs/adhan_slm_tiny.yaml \
    --checkpoint-dir models/adhan_slm/checkpoints \
    --tokenizer-dir models/adhan_slm/tokenizer \
    --query "வணக்கம்"
```

`--mode passthrough` makes exactly one `RESPOND` call and marks every other
posture skipped — prove the wire before layering the pipeline on it.

The orchestrator takes **one** client, not five. That is Option A (§3) expressed
structurally: role separation travels in the prompt's control token, not the
serving route, so four adapters cannot grow by accident before the interference
data justifies them. Option B, if it lands, is a client that dispatches on
`role` — a change confined to `model_client.py`.

## Two rules the code enforces

**Degrade loudly.** Every structured output carries `parse_ok`. A parser that
quietly invents a default turns a model failure into a data failure, so a plan
recovered from a numbered list is *usable but non-compliant*, and it says so.
`TurnResult.metrics()["structure_compliance"]` is therefore the §5 gate measured
off live traffic rather than a separate harness — and against the untrained
checkpoint it is the baseline the first training run has to beat.

**Fail closed on reflection.** An unreadable verdict is a `fail`, never a `pass`.
A reflection agent that rubber-stamps is worse than none: it manufactures false
confidence and poisons the DPO loop with bad labels.

## What the kernel produces besides an answer

A retry that rescues a rejected draft is a labelled `(chosen, rejected)` pair —
the Reflection Agent's verdict *is* the DPO signal (§4.1), harvested from
ordinary usage.

Retention of both interactions and pairs takes a required `consented` keyword;
there is no permissive default to inherit. §4.3 makes the DPDP consent flow a
hard sequencing dependency, so the gate lives in the signature where a caller has
to decide.

## Layout

| File | |
|---|---|
| `schemas.py` | Dataclasses for every posture's I/O, plus `TurnResult.metrics()` |
| `prompts.py` | Prompt construction; **owns the output contract** (field names) |
| `parsing.py` | The inverse of `prompts`; imports its field names so they cannot drift |
| `agents.py` | The five postures + per-role sampling |
| `orchestrator.py` | The turn, the cost gate, the retry loop, span metering |
| `model_client.py` | `ModelClient` protocol + the four backends |
| `memory.py` | `MemoryStore` protocol + in-memory double (Postgres is the target) |

Standard library only. The kernel must import without JAX or torch, because
Phase 1 is about validating plumbing before there is anything trained to validate
against.

## Not in this phase

Role-tagged training data, the QLoRA run, the eval gates as a scored suite, the
Postgres store, and the LangFuse exporter — `CoreOrchestrator(on_span=...)` is
the seam the last of those plugs into.
