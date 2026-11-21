# K5 — Reference phone and on-device runtime

**Date:** 2026-11-22
**Task:** [#111](https://github.com/yazhi-lem/adhan/issues/111) — choose the device and runtime Kutty v1
ships against, so K4 ([#110](https://github.com/yazhi-lem/adhan/issues/110)) has a target.
**Status:** decision recorded; procurement is the founder's action.

> **Where this lives.** The issue asked for the decision in `factory/CURRENT.md` and the ADR. The
> factory planning docs were removed in `bb99d6f` ("GitHub issues are the single source of truth for
> tasks"), so the decision is recorded here and in
> [`docs/adr/0001-kutty-v1.md`](../adr/0001-kutty-v1.md) §2.4 instead.

---

## 1. What the device actually has to do

Kutty v1 is a **Qwen3-1.7B** fine-tune exported to **INT4** (K4). The phone must run that model
locally, offline, and answer a Tamil prompt in a time a person will tolerate. That sets the
requirements, in priority order:

| # | Requirement | Why it binds |
|---|---|---|
| R1 | **≥ 8 GB RAM** (6 GB is the floor, 8 GB is the target) | A 1.7B model at 4-bit is ~1.0–1.1 GB of weights; the KV cache, tokenizer, Android runtime and the app need the rest. On 6 GB devices the OS reclaims aggressively and the app gets killed mid-generation |
| R2 | **≥ 64 GB storage**, UFS not eMMC | Model + app + OS. eMMC read latency dominates first-token time |
| R3 | **Android 13+**, 64-bit only | Runtime ABI support and predictable memory limits |
| R4 | **A chipset with a usable CPU/GPU path** (ARMv8.2+, dot-product instructions) | INT4 matmul on CPU is the fallback that must work; a Vulkan/OpenCL GPU path is a bonus, not a requirement |
| R5 | **₹10–15k street price** | The product claim is a phone a Tamil family already owns, not a flagship |
| R6 | **Thermals that survive 20 prompts** | A 1.7B model will heat a mid-range phone; sustained tokens/s matters more than peak |
| R7 | **Available in India, in volume, with a locked-down OS** | Procurement and reproducibility: the same model must be buyable again in three months |

**Explicitly not required:** an NPU. Every current mobile NPU (Hexagon, Mali APU, Dimensity APU)
needs a vendor-specific compiler and a model conversion path that Qwen3 does not have today.
Chasing it would put the schedule behind a toolchain nobody on this project controls.

## 2. Device decision

**Decision: buy one mid-range Android phone in the ₹10–15k band with 8 GB RAM / 128 GB storage,
from a family that is still on sale, and treat the exact model as a procurement detail rather than
an architectural one.**

That framing is deliberate. The runtime and the export format are the architectural choices; the
phone is a *test rig* whose job is to prove the artifact runs on commodity hardware. Pinning a
specific model number in a document that outlives the model's sales window is how these decisions
rot.

### Shortlist to check at purchase

These families have consistently shipped 8 GB/128 GB configurations in this band in India. **Prices
and availability move weekly — confirm at purchase, do not trust this table as a quote.**

| Family | Why it is on the list | What to verify |
|---|---|---|
| Redmi Note series (base model of the current generation) | The volume leader in this band; 8 GB/128 GB is a standard SKU; long OS support | RAM/storage of the *specific* SKU, not the series |
| Poco X / M series | Same silicon tier, often cheaper for the same RAM | Whether the cheap SKU is 6 GB |
| Realme Narzo / number series | 8 GB SKUs common in this band | Bloatware load (affects available RAM) |
| Samsung Galaxy M series | Longest OS update commitment in the band | Exynos vs Snapdragon variant — measure both if possible |
| Moto G series | Near-stock Android, least bloat, best available RAM | Availability in India at the time of purchase |

### Hard filters (reject regardless of price)

- 6 GB RAM **and** a heavy OEM skin — the combination that gets the app killed.
- eMMC storage.
- 32-bit-only or Android 12 or older.
- A carrier-locked or region-locked unit: the artifact must be reproducible on a retail device.

### What to record when it is bought

Model number, RAM/storage SKU, SoC, Android version + security patch, purchase price and date, and
the exact build fingerprint (`adb shell getprop ro.build.fingerprint`). K4's report needs all of
these, because "tokens/s on a phone" is meaningless without them.

## 3. Runtime decision

**Decision: `llama.cpp` with a GGUF **Q4_K_M** build as the primary runtime, and ONNX Runtime GenAI
as the documented fallback.**

| Runtime | Verdict | Reasoning |
|---|---|---|
| **llama.cpp (GGUF Q4_K_M)** | **Primary** | Mature GGUF conversion for Qwen3, CPU path that works on any ARMv8 phone, optional Vulkan/OpenCL GPU offload, single static binary, no vendor toolchain. It is also the format the community will actually run, which matters for a public release |
| ONNX Runtime GenAI (INT4) | **Fallback** | Better if we later need a Java/Kotlin SDK inside an app, and it is the path the ADR originally named. Costs a second export pipeline and a second set of numbers |
| MLC-LLM | Not now | Best raw phone performance, but it compiles per-device and per-model; that is a research project, not a shipping path |
| MediaPipe LLM Inference | Not now | Convenient Android API, but its supported-model list is the constraint, and Qwen3 fine-tunes are not on it |
| ExecuTorch | Not now | Promising, still moving; revisit after v1 |
| Vendor NPU SDKs | Rejected | Per-vendor conversion, no Qwen3 path, unmaintainable by this team |

### Version pinning

The runtime version is part of the artifact, not an implementation detail: quantisation kernels
change between releases and tokens/s moves with them. K4 must record:

- the `llama.cpp` commit (not "latest"), built with the same flags on the same host;
- the exact quantisation command and the GGUF file's SHA-256;
- the runtime flags used for measurement (`-t`, `-ngl`, `-c`, `--temp`), because thread count alone
  can swing tokens/s by 30% on a big.LITTLE phone.

## 4. What K4 must measure on this device

The point of naming a device is that K4's numbers become comparable. Minimum set, per prompt:

| Metric | How | Why it matters |
|---|---|---|
| **Time to first token** | wall clock, prompt → first streamed token | The number a user feels |
| **Tokens/s (decode)** | tokens after the first / elapsed | The number that decides "usable" |
| **Peak RSS** | `adb shell dumpsys meminfo <pkg>` or `/proc/<pid>/status` VmHWM | Whether the OS will kill it |
| **Model file size** | `ls -l` on the GGUF | The download/storage claim |
| **Sustained tokens/s** | 20 prompts back to back, report first vs last | Thermal throttling is the real experience |
| **Quality delta** | the 20-prompt Tamil smoke test, INT4 vs unquantised | Quantisation damage is a product risk, not a footnote |

Report as `docs/reports/qwen3_int4_phone.json` (K4's exit criterion), with the device fingerprint
from §2 above.

## 5. Procurement

Not something this document can do. What it can do is make the ask concrete:

- **Buy:** one retail unit, 8 GB/128 GB, from the shortlist, in the ₹10–15k band.
- **Or confirm:** an existing test device that meets R1–R4, in which case record its fingerprint and
  skip the purchase.
- **Do not** buy a 6 GB unit to save ₹1,500. The failure mode (app killed mid-generation) looks like
  a model bug and costs a day to diagnose.

## 6. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| The chosen model goes out of stock | K4 numbers become unreproducible | Record the fingerprint; the *class* of device is the decision, so a same-tier replacement is acceptable and must be re-measured |
| 1.7B INT4 is too slow on a ₹12k CPU | Product claim fails | Measure early (K4); if decode is under ~3 tok/s, the cut list applies — smaller model or server-side demo, decided with data |
| Thermal throttling | First-run numbers flatter the device | Sustained measurement is mandatory, not optional |
| Runtime version drift | Numbers move between releases | Pin the commit and the flags in the report |
| ONNX fallback never gets built | Single-runtime risk | Accept for v1; the fallback is documented, not promised |

## 7. Decision summary

- **Device:** one retail Android phone, **8 GB RAM / 128 GB**, Android 13+, ₹10–15k, from a family
  still on sale; exact model recorded at purchase with its build fingerprint.
- **Runtime:** **`llama.cpp` + GGUF Q4_K_M**, commit-pinned, with ONNX Runtime GenAI as the
  documented fallback.
- **K4 targets this:** INT4 GGUF, measured for TTFT, decode tokens/s, peak RSS, file size, sustained
  throughput and INT4-vs-unquantised quality on 20 Tamil prompts.
