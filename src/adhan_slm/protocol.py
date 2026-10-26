"""Adhan response protocol v1 — Tamil textual role markers with structured reasoning.

The Swaram v2 tokenizer has no chat/role/thinking tokens (only ``<pad> <bos> <eos>
<unk> <mask>`` and the byte fallback). Adding them would resize the tied embedding
(57% of the 14.5M model) and invalidate every packed shard, so the protocol stays
*textual* until tokenizer v3:

    அமைப்பு: <system instruction>
    பயனர்: <turn 1>
    உதவியாளர்: <response 1>
    பயனர்: <turn 2>
    உதவியாளர்:

Structured Reasoning (new in v1)
--------------------------------
Adhan thinks, rationalizes, and reflects entirely in modern Tamil before answering.
The reasoning trace is a mandatory part of the target during SFT, but loss is
computed ONLY on the final answer — the reasoning tokens are masked.

    உதவியாளர்:
    சிந்தனை: <initial analysis of the question>
    ஆய்வு: <step-by-step investigation, evidence gathering>
    பரிசீலனை: <self-correction, checking assumptions, doubt>
    பதில்: <concise answer> [மேற்கோள்: <work> <reference>]

Why four stages?
  * ``சிந்தனை:`` (Thought) — decomposes the question, identifies what is being asked
  * ``ஆய்வு:`` (Rationalize/Investigate) — works through the logic, cites evidence
  * ``பரிசீலனை:`` (Reflect) — challenges own assumptions, admits uncertainty
  * ``பதில்:`` (Answer) — the final, confident output

Rules:
* markers are literal Tamil words at the start of a line, followed by ``:`` and a
  space. They tokenize into ordinary Swaram pieces;
* reasoning stages are optional in inference but mandatory in SFT training data;
* a grounded answer cites at least one source; when the source does not contain the
  answer, the target is :data:`REFUSAL`, with no citation;
* citations use ``[மேற்கோள்: work reference]`` and never invent a work/verse;
* all reasoning MUST be in modern Tamil — no English inside the trace;
* the model should admit uncertainty in ``பரிசீலனை:`` rather than hallucinate.

This module is dependency-free and import-safe (no JAX, no tokenizer), so data
builders, eval harnesses and serving code can all share one definition.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Role markers
SYSTEM = "அமைப்பு:"
USER = "பயனர்:"
ASSISTANT = "உதவியாளர்:"

# Reasoning markers (v1 — structured Tamil reasoning)
THOUGHT = "சிந்தனை:"  # Initial analysis
RATIONALIZE = "ஆய்வு:"  # Step-by-step investigation
REFLECT = "பரிசீலனை:"  # Self-correction, doubt, admission of uncertainty
ANSWER = "பதில்:"  # Final output

# Legacy marker for backward compatibility
THOUGHT_LEGACY = "சிந்தனை:"

CITATION_OPEN = "[மேற்கோள்:"
CITATION_CLOSE = "]"

#: All target-side markers that should be masked during SFT (loss = 0).
#: Only tokens after ANSWER contribute to loss.
REASONING_MARKERS = (THOUGHT, RATIONALIZE, REFLECT)

#: Canonical refusal for a question the provided source cannot answer.
REFUSAL = "இந்தக் கேள்விக்கான தகவல் கொடுக்கப்பட்ட மூலத்தில் இல்லை."

ROLE_MARKERS: Dict[str, str] = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}

_CITATION_RE = re.compile(r"\[மேற்கோள்:\s*([^\[\]]+?)\s*\]")

# Regex to identify reasoning sections in model output
_REASONING_RE = re.compile(
    rf"(?:{re.escape(THOUGHT)}|{re.escape(RATIONALIZE)}|{re.escape(REFLECT)})"
)


def citation(work: str, reference: str) -> str:
    """Render one citation. ``work`` is a text/work name, ``reference`` its locator."""
    work, reference = work.strip(), reference.strip()
    if not work or not reference:
        raise ValueError("citation needs both a work and a reference")
    return f"{CITATION_OPEN} {work} {reference}{CITATION_CLOSE}"


def render_prompt(
    messages: Sequence[dict],
    system: Optional[str] = None,
    *,
    add_assistant_cue: bool = True,
) -> str:
    """Render chat messages up to the assistant cue."""
    lines: List[str] = []
    if system:
        lines.append(f"{SYSTEM} {system.strip()}")
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"message {i}: role must be 'user' or 'assistant', got {role!r}")
        content = str(msg.get("content", "")).strip()
        if not content:
            raise ValueError(f"message {i}: empty content")
        lines.append(f"{ROLE_MARKERS[role]} {content}")
    if add_assistant_cue:
        lines.append(ASSISTANT)
    return "\n".join(lines)


def render_target(
    answer: str,
    thought: Optional[str] = None,
    rationalize: Optional[str] = None,
    reflect: Optional[str] = None,
    citations: Iterable[str] = (),
) -> str:
    """Render an assistant target with structured Tamil reasoning.

    Args:
        answer: The final answer (required).
        thought: Initial analysis ("இது ஒரு வரலாற்று கேள்வி...").
        rationalize: Step-by-step working ("முதலில், புறநானூற்றைப் பார்ப்போம்...").
        reflect: Self-correction ("எனினும், இந்த மூலம் நம்பத்தகுந்ததா என்று உறுதி செய்ய வேண்டும்").
        citations: List of citation strings.

    Returns:
        A target string with all reasoning stages + answer.
    """
    answer = answer.strip()
    if not answer:
        raise ValueError("answer must not be empty")
    cites = [c.strip() for c in citations if c and c.strip()]
    for c in cites:
        if not (c.startswith(CITATION_OPEN) and c.endswith(CITATION_CLOSE)):
            raise ValueError(f"citation must look like {CITATION_OPEN} work ref]: {c!r}")
    lines: List[str] = []
    if thought and thought.strip():
        lines.append(f"{THOUGHT} {thought.strip()}")
    if rationalize and rationalize.strip():
        lines.append(f"{RATIONALIZE} {rationalize.strip()}")
    if reflect and reflect.strip():
        lines.append(f"{REFLECT} {reflect.strip()}")
    tail = answer
    if cites:
        tail = f"{tail} {' '.join(cites)}"
    lines.append(f"{ANSWER} {tail}")
    return "\n".join(lines)


def render_example(
    messages: Sequence[dict],
    answer: str,
    thought: Optional[str] = None,
    rationalize: Optional[str] = None,
    reflect: Optional[str] = None,
    citations: Iterable[str] = (),
    system: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(prompt, target)`` for one supervised example."""
    return (
        render_prompt(messages, system),
        render_target(
            answer, thought=thought, rationalize=rationalize, reflect=reflect, citations=citations
        ),
    )


def parse_response(text: str) -> dict:
    """Best-effort parse of a model response into structured reasoning + answer.

    Handles both v0 (சிந்தனை/பதில் only) and v1 (four-stage) formats.
    Never raises on malformed output.
    """
    raw = (text or "").strip()
    cites = _CITATION_RE.findall(raw)
    body = _CITATION_RE.sub("", raw).strip()

    thought = rationalize = reflect = None
    answer = body
    found_answer = ANSWER in body

    if found_answer:
        before, _, after = body.partition(ANSWER)
        answer = after.strip()
        # Parse reasoning sections before the answer
        remaining = before.strip()
        for marker, key in [
            (REFLECT, "reflect"),
            (RATIONALIZE, "rationalize"),
            (THOUGHT, "thought"),
        ]:
            if marker in remaining:
                parts = remaining.split(marker, 1)
                remaining = parts[0].strip()
                val = parts[1].strip() if len(parts) > 1 else ""
                # Stop at the next reasoning marker
                for next_marker in (THOUGHT, RATIONALIZE, REFLECT, ANSWER):
                    if next_marker != marker and next_marker in val:
                        val = val.split(next_marker)[0].strip()
                if key == "thought":
                    thought = val or None
                elif key == "rationalize":
                    rationalize = val or None
                elif key == "reflect":
                    reflect = val or None
    elif THOUGHT_LEGACY in body:
        # v0 fallback
        thought, _, answer = body.partition(THOUGHT_LEGACY)
        thought = thought.strip() or None
        answer = answer.strip()

    return {
        "thought": thought,
        "rationalize": rationalize,
        "reflect": reflect,
        "answer": answer.strip(),
        "citations": [c.strip() for c in cites],
        "found_answer": found_answer,
        "raw": raw,
    }


def build_reasoning_mask(token_ids: List[int], tok) -> List[float]:
    """Build a loss mask for SFT: 0.0 on reasoning tokens, 1.0 on answer tokens.

    This is the KEY function for teaching Adhan to reason. The model sees the
    full reasoning trace in the target, but gradient only flows through the
    answer portion. This means:
      * The model learns to generate reasoning (autoregressive prediction)
      * But is only penalized for answer quality (supervised signal)

    Args:
        token_ids: The full target sequence (bos + prompt + reasoning + answer + eos).
        tok: Tokenizer with vocab dict.

    Returns:
        Float mask parallel to token_ids.
    """
    vocab = tok.vocab if hasattr(tok, "vocab") else {}
    unk_id = getattr(tok, "unk_id", vocab.get("<unk>", 1))
    eos_id = vocab.get("<eos>", unk_id)

    # Build marker token sequences
    def _marker_ids(marker_text: str) -> List[int]:
        return tok.encode(marker_text, add_special=False)

    ans_ids = _marker_ids(ANSWER)
    reason_ids = [_marker_ids(m) for m in REASONING_MARKERS]  # noqa: F841 (not yet used in mask)

    mask = [0.0] * len(token_ids)
    in_answer = False

    i = 0
    while i < len(token_ids):
        tid = token_ids[i]  # noqa: F841
        # Check if we're at the answer marker
        if not in_answer:
            if token_ids[i : i + len(ans_ids)] == ans_ids:
                in_answer = True
                i += len(ans_ids)
                continue
        # If in answer section, mask = 1.0 (until EOS)
        if in_answer:
            mask[i] = 1.0
        i += 1

    # Always mask EOS as part of answer
    if eos_id in token_ids:
        eos_idx = token_ids.index(eos_id)
        mask[eos_idx] = 1.0

    return mask


def is_refusal(answer: str) -> bool:
    """True when an answer is the canonical refusal (exact, after normalisation)."""
    norm = " ".join((answer or "").split())
    return norm == " ".join(REFUSAL.split())
