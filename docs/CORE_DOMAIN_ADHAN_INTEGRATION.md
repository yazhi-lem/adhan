# Core Domain × Adhan — Integration & Training Plan

*Draft v0.1 · Yazhi Sovereign Technologies · For review, not yet approved*

---

## 0. The one-sentence thesis

Core Domain is an **orchestration shell**, not a model. Right now it's model-agnostic on paper. The integration work is making Adhan the *only* model that shell ever calls — in five different postures (context, reasoning, planning, reflection, generation) — without breaking the Sovereignty Rule, without adding a second foreign-hosted dependency, and without inflating inference cost past what B2G contracts can bear.

That means this is really two projects wearing one name:
1. **An infra project** — get Core Domain's five agent calls routed through Adhan on vLLM instead of five separate model calls.
2. **A training project** — make Adhan *good* at each of those five postures, which it is not, by default, right now.

This doc covers both, in that order, because the infra decision changes what the training data needs to look like.

---

## 1. One reasoning kernel, personas on top

Core Domain is the **single reasoning kernel** for the Yazhi ecosystem. Every Yazhi product (Yazh, Annachi, the legal RAG assistant) sits on top of it as a **thin persona** — a system prompt plus a retrieval scope — rather than as its own codebase reimplementing context, reasoning, and reflection.

This is a deliberate consolidation. Instead of many parallel product-specific reasoning stacks, there is one kernel with five well-defined agent postures, and each product supplies only the persona and the data scope it needs. That is also the cheaper training story: one set of Adhan behaviors to teach, not one per product.

The five agent postures the kernel exposes are:

- **Context Agent** — assembles a compact brief from prior interactions and profile.
- **Reasoning Agent** — structured decomposition of intent, ending in a `requires_planning` flag.
- **Planning Agent** — converts reasoning into an ordered, staged plan.
- **Reflection Agent** — self-critique against context/profile/RAG grounding.
- **Response Generator** — final natural-language prose, incorporating reflection notes on a retry.

Everything below assumes this kernel-plus-personas shape.

---

## 2. Mapping the five agents onto Adhan

| Core Domain Agent | What it needs from the model | Is this a *new* Adhan skill, or does it exist? |
|---|---|---|
| **Context Agent** | Mostly retrieval + formatting — light model involvement (summarizing `previous_interactions` into a compact brief) | Minor — Adhan already needs summarization for persona memory. Reuse. |
| **Reasoning Agent** | Structured, step-by-step decomposition of intent in Tamil/English/Tanglish, ending in an explicit `requires_planning` flag | **New.** Adhan's current QLoRA fine-tune has not been trained to emit structured intermediate reasoning — this needs its own data. |
| **Planning Agent** | Converts reasoning into an ordered, staged plan (roadmap register — closer to the "scheme eligibility → steps" pattern of the Tamil-Gov vertical than to prose) | **New**, but close cousin of the B2G vertical use case already planned for Tamil-Gov. High reuse value. |
| **Reflection Agent** | Self-critique: pass/fail + reasons, checked against context/profile/RAG grounding | **New**, and the highest-leverage one — this is also exactly the mechanism that produces DPO preference pairs for the Sangam→Adhan→Yazh flywheel. Building this well pays for itself twice. |
| **Response Generator** | Final natural Tamil/English/Tanglish prose, incorporating reflection notes if a retry happened | Exists — this is closest to what Adhan already does. Needs light conditioning on "incorporate reflection notes" as a new input, not a new skill. |

So of five agents, **Reasoning, Planning, and Reflection are the actual training gap.** Context and Response Generation are largely infra/prompt work on capability Adhan already has.

---

## 3. Training architecture decision: one adapter or four

This is the first real fork in the road.

**Option A — Single multi-task LoRA, role-tagged.**
One Adhan checkpoint, with a `<role>` control token (`[CONTEXT]`, `[REASON]`, `[PLAN]`, `[REFLECT]`, `[RESPOND]`) prepended to the prompt, trained on a mixed dataset across all five postures.

- ✅ One model to serve, one model to version, one model to eval-gate against DPDP before launch.
- ✅ Matches "Adhan does not compete with Sarvam on general chat" positioning — it stays one coherent product, not five.
- ⚠️ Risk: cross-task interference — reasoning-trace training data (verbose, structured) can degrade fluent response generation (concise, natural) if not carefully weighted.

**Option B — Four separate LoRA adapters over one base**, hot-swapped per agent call via vLLM's multi-LoRA serving.

- ✅ Cleaner separation, easier to iterate on Reflection without touching Response Generation.
- ⚠️ Four adapters to eval, four to version, four things that can silently drift from each other's assumptions about the user.
- ⚠️ More serving complexity on a stack that's still standing up its first production vLLM deployment.

**Recommendation: Option A for launch.** Yazhi's inference stack (vLLM in production, still early) and intern-team bandwidth (FDE Foundry cohort, not a dedicated ML infra team yet) both argue for the simpler shape first. Revisit Option B only if role-tagged single-adapter training shows measurable interference (tracked via the eval gates in §5) — that's a Phase 3 decision, not a launch-time one.

---

## 4. Where the training data comes from

Three sources, in order of how much Yazhi already has vs. has to build:

**4.1 Reuse — the existing flywheel, extended**
The Sangam → Adhan → Yazh → DPO loop already documented is the backbone. What's new: the **Reflection Agent's pass/fail + reason output *is* a DPO preference signal** — every reflection pass/fail on a real Yazh conversation is a labeled (chosen, rejected) pair for free, once Core Domain is live. This is the single highest-leverage integration point: Core Domain doesn't just consume Adhan, it becomes a *better labeled-data generator* than the persona layer alone was, because reflection makes the "why this response failed" reasoning explicit instead of inferred from user behavior.

**4.2 Synthetic — reasoning and planning traces**
Structured reasoning traces don't occur naturally in Tamil corpora (Sangam literature isn't going to hand you `requires_planning: true` examples). This has to be synthetically generated:
- Seed with real Tamil-Gov / Tamil-Legal style multi-step queries (scheme eligibility, document checklists) — these already have a natural "reason → plan" shape.
- Generate synthetic (query, reasoning trace, plan) triples using a larger open-weight teacher model, in Tamil and Tanglish, then **native-reviewer sign-off before any of it touches training data** — this is not optional; it's the same brand rule that governs marketing copy, applied to training data instead, and arguably more important here because bad Tamil reasoning traces poison the model rather than just looking wrong on a page.
- Cap synthetic data's share of the training mix — this is exactly the kind of thing the "AI Tamil quality is bounded" honesty principle should apply to internally, not just externally: track and report synthetic:native ratio per training run.

**4.3 Consented production logs**
Once Core Domain is live in Yazh, the `previous_interactions` and reflection outcomes are real usage data — but this is explicitly gated by DPDP consent flows (same launch gate already blocking everything else). No training on production logs until that consent mechanism is live and audited. This is a hard sequencing dependency, not a nice-to-have — flagging it now so it doesn't get discovered as a blocker in Phase 3.

---

## 5. Eval gates (new additions to the existing model-release checklist)

Adhan already has release gates from the Yazh PRD lineage (bilingual quality, safety). Core Domain adds three, specific to the reasoning-layer behaviors:

| Gate | What it checks | Why it's new |
|---|---|---|
| **Reasoning-trace structure compliance** | Can the Reasoning Agent output be parsed into the expected schema (intent, sub-problems, `requires_planning`) reliably, in Tamil and Tanglish? | Structured-output reliability in a low-resource-tuned model is not guaranteed just because it works in English fine-tunes. |
| **Reflection calibration** | Does the Reflection Agent's pass/fail correlate with actual quality (spot-checked against native reviewer judgment), or is it just rubber-stamping? | A reflection agent that always says "pass" is worse than no reflection agent — it creates false confidence and poisons the DPO loop with bad labels. |
| **Cross-task interference** | Does adding reasoning/planning/reflection training data measurably degrade plain Response Generation quality on the existing bilingual eval suite? | This is the direct test for whether Option A (single adapter) is still the right call, per §3. |

None of these replace the existing DPDP compliance gate or the Tamil normalisation test suite — they sit alongside them.

---

## 6. Serving & infra notes

- Core Orchestrator's five agent calls → Adhan on vLLM, not Ollama. Ollama (Mac Mini, qwen3:4b) stays dev-only for iterating on prompts.py before a training run, consistent with current setup.
- Memory Manager (`UserContext`, `PersonalizationProfile`) is a new persistent store — this **must** be Postgres, same instance family as the legal pipeline's pgvector setup, not a new service. No new infra surface, no new DPDP audit surface.
- `requires_planning` skip logic (Orchestrator §4A in the Core Domain doc) matters for cost: most Yazh conversational turns (small talk, cultural Q&A) should *never* invoke the Planning Agent. Get this routing right early — it's the difference between one Adhan call and four per turn, and B2G contract margins depend on inference cost staying low.
- LangFuse tracing (already in the stack) should be wired to log per-agent latency and the `required_planning` / reflection pass-fail rate from day one — this is also your interference-monitoring data for the Option A/B decision in §3.

---

## 7. Phasing — mapped onto what's already in motion, not a new timeline

Rather than invent a separate calendar, this slots into the existing legal-pipeline and Adhan work already underway:

**Now → alongside legal pipeline Phase 1 (India Code scraping, pgvector schema)**
- Stand up `domains/core/` module structure, schemas, orchestrator skeleton, single-agent pass-through (per Core Domain doc §11 Phase 1) — no training yet, wire it to existing Adhan checkpoint as-is to validate plumbing.
- Reflection Agent design reviewed against native-reviewer workflow *before* any synthetic data generation starts.

**Next → alongside legal pipeline Phase 2 (State Acts ingestion, Q3 2026)**
- Generate + review synthetic reasoning/planning training data (§4.2), scoped first to Tamil-Gov / Tamil-Legal verticals since those already have natural multi-step structure.
- First QLoRA training run, Option A (single role-tagged adapter).
- Run new eval gates (§5) against existing bilingual + safety suite.

**Then → gated on DPDP launch gate clearing**
- Turn on consented production-log ingestion (§4.3) from live Yazh Core Domain usage.
- Revisit Option A vs B based on real interference data.
- Extend Reflection-Agent-as-DPO-generator into the standing Sangam→Adhan→Yazh flywheel loop as a permanent data source, not a one-time training run.

---

## 8. Open questions before Phase 1 starts

1. Does the Planning Agent's roadmap register need to match the existing Tamil-Gov "scheme → eligibility → steps" tone, or is it meant to generalize beyond government use cases? Affects how narrowly to scope the first training data pull.
2. Who signs off on synthetic reasoning-trace Tamil quality — same native reviewer pool as marketing copy, or a separate technical-Tamil reviewer (legal/gov register is a different skill than brand copy)?
3. Cost ceiling: is there a target inference cost per Core Domain-routed turn (comparable to Yazh's per-child COGS target in the PRD lineage), or does B2G pricing absorb it differently?

---

*This is a planning draft — flagging where it's genuinely uncertain rather than presenting false confidence: the reasoning/planning training-data generation approach (§4.2) is the least-proven piece here and should get a small pilot batch + native review before committing to a full training run.*
