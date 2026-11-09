# CURRENT — adhan · 2026-11-05

> Reset every Monday. Facts only; each number cites its source. The template and rules are in [`README.md`](README.md).

## 1. Headline

**🟡 Execution reset — base decided, now prove it.** On 3 Nov the base question closed:
Kutty v1 is a fine-tune of **`Qwen/Qwen3-1.7B-Base`** (Apache-2.0), with
**`sarvamai/sarvam-m` 24B** (Apache-2.0) as the server-side teacher/reference.
`Sarvam-1` 2B is rejected for product use because its licence is non-commercial/research
only. `adhan-nano` + Swaram continue as the research artifact, not the product base.
Full record: [`docs/reports/base_decision_2026-11-03.md`](../docs/reports/base_decision_2026-11-03.md).

The critical risk is now **Qwen3's Tamil tokenizer tax**: 2.027 tok/akshara vs 0.584 for
Sarvam-1 and 0.418 for Agaram on the same 300-doc sample (rerunnable rule:
`docs/reports/bases_tokenizer_bench.json`). The 48-hour bake-off (`K1`) decides whether
Qwen3-1.7B is good enough on Tamil quality, or whether we move to Qwen3-4B / a negotiated
Sarvam-1 licence. The other hard blocker is **GPU access** for SFT, distillation, and the
teacher.

**5 Nov — Agaram 2.5B read and scored, no base change.** Its tokenizer is the best
licence-clean Tamil tokenizer measured (0.418 tok/akshara, Apache-2.0) but its quality
fails our smoke (XNLI 12.0% vs Qwen3's 42.0%, IndicQA EM 0.000 vs 0.140). D1 stands; the
`base_decision` trigger is annotated as *partially fired, closed for v1*
(`docs/reports/agaram_reading_2026-11-05.md`). The reusable part of Agaram is the *recipe*,
not the weights: eval-leak removal, a shipped tokenizer rule, and honest weak-row
reporting. Those recipes landed as Adhan tooling on 5 Nov (see §2 and §7).

## 2. Health

| Workstream | RAG | Fact (source) | Next checkpoint |
|---|---|---|---|
| Base decision | 🟢 | **Qwen3-1.7B-Base student · Sarvam-M 24B teacher · Sarvam-1 rejected** (`docs/reports/base_decision_2026-11-03.md`) | Bake-off result K1 |
| Base bake-off | 🟡 | Preliminary CPU smoke: Qwen3-1.7B XNLI **42.0%** (n=50), IndicQA **EM 0.140/F1 0.192** (n=50); 4B/full-GPU run pending (`docs/reports/base_bakeoff_qwen3.md`) | K1 full run, due 5 Nov |
| Tokenizer bench | 🟢 | Frozen 3-register corpus now exists: 1,500 licensed sentences (Thirukkural + Kalithokai PD, Tamil Wikipedia CC BY-SA, Tanglish-185k @`927f5b4` CC-BY-4.0). Tokens/akshara: Agaram 0.364 < Sarvam-1 0.505 < Swaram-v3 0.564 < Sarvam-M 0.602 < Qwen3 1.346; Swaram splits 0.1% of aksharas vs Qwen3 89.1% (`docs/reports/tokenizer_bench_results.md`, `docs/TOKENIZER_BENCH.md`) | Re-run on any vocab change; quote the sample rule |
| SFT v0 | ⚪ | Data ready: Sangam 9,961 train / 1,303 val / 1,146 test, zero poem overlap; grounded QA 971/124/182. No Qwen3 SFT run yet | K2, due 10 Nov |
| Distillation | ⚪ | Sarvam-M 24B teacher not yet run; no Tamil prompt/verifier pipeline | K3, due 14 Nov |
| Export / phone | 🔴 | No Qwen3 INT4 GGUF/ONNX build; reference phone not chosen | K4/K5, due 21 Nov |
| Tanglish sanitization | 🟡 | Source scoped: `vishnu-n/Tanglish-Corpus-185k` @ `927f5b4` (CC-BY-4.0); 5.5% keyword-political, 171k neutral candidates. Gazetteer-first plan | K6, due 10 Nov |
| Safety (Aram) | 🟡 | Stage 1b baseline 0.13, 7 CRITICAL/9 (`docs/reports/aram_report_stage1b.json`); Qwen3 untested | K7, due 21 Nov |
| Serving | 🟢 | Adhan FastAPI validated against Stage 1b: `/generate` p50 1,757 ms / p95 2,540 ms CPU (`docs/reports/serving_stage1b.json`) | Re-test with Qwen3 endpoint |
| Research track | 🟡 | Swaram A/B: 0.527 vs Tamil BPE 0.346 → compression thesis unsupported; morphology 5.6–24.2%, gate 30%. Agaram's plain 64k byte BPE reaches 0.418 on the same sample, so "byte-level BPE cannot do Tamil" is false (`docs/reports/agaram_reading_2026-11-05.md`) | Keep as paper, not product gate |
| Corpus hygiene | 🟢 | MinHash dedup (`adhan-dedup`) + 3-method leak filter (`adhan-decontaminate`) landed with 16 tests (`src/adhan_slm/data/{minhash_dedup,contamination}.py`); **not yet wired into `scripts/prepare_slm_corpus.py`** | Wire into the corpus build; record removed-doc counts |
| Training diagnostics | 🟢 | Per-layer grad norms, activation stats, spike/NaN alerts behind `train.diagnostics_every` (off by default); 5 tests (`src/adhan_slm/training/debug_diagnostics.py`) | Enable on the next real pretraining run |
| Export path (research) | 🟡 | `scripts/export_gguf.py` scaffolds JAX→HF→GGUF with the full parameter map and a Tamil chat template; weight-copy is **not implemented** and no parity test exists | Implement mapping + token parity before any release claim |
| Tokenizer scripts | 🟢 | One validated registry (`src/adhan_slm/common/scripts.py`): 10 scripts, Tamil 12 uyir / 18 mei / 5 Grantha / 11 vowel signs, 48-column Dravidian matrix; Swaram + Aksharam both delegate to a single `segment`. 239 passed / 1 skipped on the tokenizer path (9 Nov) | Add a script, never a second table |
| CI | 🟢 | 322 passed / 2 skipped (9 Nov, `pytest tests/ src/adhan_slm/tokenizer src/adhan_slm/common`, integration excluded); 320 passed / 2 skipped (5 Nov, `pytest tests/ src/adhan_slm`) | Keep green |

## 3. Key metrics

| Metric | Now | Target | Source |
|---|---|---|---|
| Kutty v1 product base | **Qwen3-1.7B-Base** (Apache-2.0) | bake-off confirms Tamil quality | `docs/reports/base_decision_2026-11-03.md` |
| Teacher / reference | **Sarvam-M 24B** (Apache-2.0) | distillation data + eval ceiling | same |
| Tokenizer fertility, Qwen3 | **2.027 tok/akshara** (89.5% of aksharas split) | measure after SFT; no regression | `docs/reports/bases_tokenizer_bench.json` — 300 docs, 38,991 aksharas, seed `adhan-bases-2026-11-05` |
| Tokenizer fertility, Sarvam-1 | 0.584 | — (product-rejected) | same file |
| Tokenizer fertility, Sarvam-M | 0.795 | teacher efficiency | same file |
| **Best licence-clean Tamil tokenizer** | **Agaram 2.5B 0.418** (Apache-2.0) | proof that the tax is not a licence dead end | same file |
| Tokenizer fertility, Swaram v2 / v3 | 0.593 / 0.582 (0.0% clusters split) | research artifact | same file + `docs/reports/ab_swaram_vs_bpe.json` |
| Tamil-trained BPE fertility | 0.346 | research baseline | `docs/reports/ab_swaram_vs_bpe.json` |
| SFT examples | 11,232 (9,961 + 1,271) | 20–50k | `data/final/sft_*/stats.json` |
| bits/akshara (Adhan Stage 1b) | 1.820 | research only | `docs/reports/eval_report_nano_stage1b.json` |
| Serving `/generate` latency | p50 1,757 ms · p95 2,540 ms (Adhan CPU) | Qwen3 INT4 on phone | `docs/reports/serving_stage1b.json` |
| Aram Harness score | 0.13 · 7 CRITICAL (Adhan Stage 1b) | >0.90, 0 CRITICAL for Qwen3 | `docs/reports/aram_report_stage1b.json` |
| Tanglish neutral candidates | 171,449 (after keyword filter) | freeze 500 | `docs/reports/n3_tanglish_source_scope.md` |

## 4. In flight

| Change | What | Age | State | Needed to land |
|---|---|---|---|---|
| K1 base bake-off | Qwen3-1.7B vs Qwen3-4B on tamil-bench + 20-prompt read | new | partial CPU smoke only | GPU/full n + Qwen3-4B + generation read |
| K2 SFT v0 | Qwen3-1.7B + Sangam/QA SFT | new | not started | GPU window + regression gate |
| K3 distillation | Sarvam-M teacher prompts/filter | new | not started | GPU + Tamil verifier |
| K6 Tanglish freeze | gazetteer sanitizer + 500 sentences | new | not started | filter code + native review |
| Corpus hygiene wiring | MinHash + leak filter into `prepare_slm_corpus.py` | new | modules landed, unwired | one call site + a removed-doc count in the datasheet |

## 5. Blockers (oldest first)

| # | Blocker | Since | Blocks | Ask / who |
|---|---|---|---|---|
| B1 | **No GPU.** SFT, distillation, and Qwen3-4B/24B teacher runs need one | ongoing | K1–K3, 26 Dec | Book a T4/A10 weekend or platform GPU node (`K1`) |
| B2 | **Qwen3 Tamil quality unknown** and its tokenizer is 3.75× less efficient than Sarvam-1 | 3 Nov | product base confirmation | Run K1 bake-off now |
| B3 | **Export path for Qwen3 → INT4 GGUF/ONNX** is untested | 3 Nov | phone demo K4/K5 | Spike export on Qwen3-1.7B |
| B4 | **Tanglish political filter** has no human-labeled audit | 3 Nov | public eval table K6 | Gazetteer pass + 100-sentence native review |
| B5 | **Sarvam-M teacher access** (24B) not booked | 3 Nov | distillation K3 | GPU with ≥48 GB or quantised teacher |
| B6 | **Corpus hygiene is code-complete but unwired** — no run has recorded a dedup or leak-removal count | 5 Nov | any pretraining/continued-pretraining mix claim | Call the two modules from `prepare_slm_corpus.py`; put the counts in the datasheet |

## 6. Decisions needed

| # | Decision | Options | Recommendation | Decider | By |
|---|---|---|---|---|---|
| D1 | Kutty v1 base | Adhan-nano · Sarvam-1 · Qwen3-1.7B · Qwen3-4B | **Qwen3-1.7B-Base** chosen 3 Nov; revisit only if K1 fails | Founder | **recorded** |
| D2 | If Qwen3-1.7B Tamil quality fails K1 | Qwen3-4B · Gemma-3-4B-IT · negotiated Sarvam-1 commercial licence | Prefer Qwen3-4B first; do not silently change licence | Founder | 5 Nov |
| D3 | Teacher data policy | Sarvam-M outputs + human filter · Qwen3-4B outputs · no distillation | Sarvam-M + verifier/human sample; record licence/terms | Adhan DRI | 7 Nov |
| D4 | Reference phone + runtime | Android ₹10–15k · emulator · server demo | Phone + llama.cpp/ONNX; fall back to desktop demo only via cut list | Adhan DRI | 7 Nov |

## 7. Done last iteration

- **9 Nov — a benchmark corpus that can be cited.** `eval_manifests/tokenizer_bench.sources.json` now names four licensed sources and `tokenizer_bench.manifest.json` records per-source documents/candidates/sentences and SHA-256 (no raw text committed). Two crawl-derived pools are cut by `scripts/build_bench_sources.py` with a stated rule; Tanglish comes from the CC-BY-4.0 `vishnu-n/Tanglish-Corpus-185k` at the pinned revision. First real table: Agaram 0.364 tok/akshara, Sarvam-1 0.505, Swaram-v3 0.564, Sarvam-M 0.602, Qwen3 1.346, with Qwen3-1.7B and 4B sharing one tokenizer. Harness fix: `--swaram` is repeatable with `--swaram-name`, and `tokenizer.json` rows are labelled by file, not by parent directory.
- **9 Nov — one home for every script; Swaram and Aksharam hardened.** All Brahmic tables moved to `src/adhan_slm/common/scripts.py` (validated at import, read-only): 10 scripts with per-script layout offsets (Sinhala's virama is U+0DCA, not +0x4D), the Tamil eluttu inventory and the 48-column Dravidian matrix from *The Periodic Table of Dravidian Scripts*. `indic_akshara` is now a facade; morphology, punarchi and both tokenizers read the same table. Fixed: 69 reserved codepoints in the Tamil inventory (U+0BC3–U+0BC5, U+0BC9), Tamil anusvara U+0B82 and ௗ U+0BD7 not attaching, Grantha letters (ஜ ஶ ஷ ஸ ஹ) classified `OTHER` so sandhi on grantha-final words never fired, Devanagari ZWJ/Vedic/prishthamatra signs splitting, and puṇarcci reversal emitting candidates the forward engine cannot reproduce. **Existing vocab files are affected** — `data/final/tamil_slm/vocab.json` carries 148 reserved codepoints; regenerate rather than reuse (see `docs/DEV.md` §3). Tests: `tests/test_common_scripts.py` (45), `tests/test_tokenizer_hardening.py` (74), `tokenizer/punarchi_tests.py` (16).
- **5 Nov — Agaram 2.5B read + scored; base unchanged.** Tokenizer table is now rerunnable from a committed script and sample rule (`scripts/benchmark_bases_tokenizers.py` → `docs/reports/bases_tokenizer_bench.{json,md}`); quality smoke and honest caveats in `docs/reports/agaram_reading_2026-11-05.md`. `base_decision` §4 annotated (trigger partially fired, closed for v1).
- **5 Nov — corpus hygiene tooling.** MinHash LSH dedup (`adhan-dedup`, 128 hashes / 5-char shingles / 0.85 Jaccard default) and a 3-method benchmark-leak filter (`adhan-decontaminate`: exact, word 8-gram, char 5-gram) with JSON reports; 16 tests.
- **5 Nov — training diagnostics.** Per-layer gradient norms, activation stats, rolling-window spike detection (>3σ) and NaN/Inf alerts, wired into `train_jax` behind `train.diagnostics_every` (off by default so CPU runs are unchanged); 5 tests.
- **5 Nov — safety + chat surface.** Crisis-phrase detection and the Tele-MANAS 14416 / 112 footer (`adhan_slm.safety.crisis`), plus `adhan-chat` using Adhan's Tamil role protocol instead of ChatML; 6 tests.
- **5 Nov — packaging.** Console scripts `adhan-dedup`, `adhan-decontaminate`, `adhan-chat`; new `[export]` extra (torch/transformers/safetensors) kept out of the core env; `scripts/build_package.sh` runs the CI gates, resolves every entry point, builds sdist+wheel and asserts required modules are inside it.
- **3 Nov — base decision recorded.** Qwen3-1.7B student, Sarvam-M teacher, Sarvam-1 rejected; tokenizer fertility evidence and fallback documented (`docs/reports/base_decision_2026-11-03.md`).
- **3 Nov — MuRIL assessed.** Encoder-only, not a base; frozen XNLI 42.77%, Tanglish political weak-label CV 67.75%/AUC 0.754 → gazetteer-first sanitization (`docs/reports/muril_base_assessment.md`).
- **3 Nov — factory reset** around the new execution plan; old N1–N17 backlog removed.
- Earlier: Stage 1b eval 1.820 bits/akshara; packing fix 99.9997%; SFT v0 Sangam set; serving validated; first Aram baseline; structure-invariant CI.
