"""Model backends for the kernel — one client, five roles.

The orchestrator takes a **single** ``ModelClient``. That is not an accident of
convenience: it is Option A from §3 expressed structurally. Role separation
travels in the prompt's control token, not in the serving route, so there is no
place to accidentally grow four divergent adapters before the interference data
justifies it. Option B, if it ever lands, becomes a client that dispatches on
``role`` — a change confined to this file.

Backends, in the order the plan expects them to be used:

* :class:`EchoClient`     — no model at all; deterministic schema-valid output,
                            for validating plumbing and for tests.
* :class:`OllamaClient`   — **dev only** (Mac Mini, qwen3:4b), for iterating on
                            prompts before a training run (§6).
* :class:`VLLMClient`     — production. Adhan served on vLLM.
* :class:`AdhanCheckpointClient` — a local JAX checkpoint straight out of
                            ``adhan_slm.inference``, which is what Phase 1 wires
                            up "as-is" to prove the pipe works.

HTTP is done with the standard library on purpose: the kernel must stay
importable without pulling a new dependency into the serving image.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from domains.core.schemas import AgentRole

DEFAULT_TIMEOUT_S = 60.0


class ModelClientError(RuntimeError):
    """Any failure to get a completion out of a backend."""


class ModelClient(Protocol):
    """What the kernel needs from a model. Deliberately tiny."""

    def generate(
        self,
        prompt: str,
        *,
        role: AgentRole,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Sequence[str] | None = None,
    ) -> str:
        ...


# --- dev / test ------------------------------------------------------------


@dataclass
class EchoClient:
    """A model-shaped stub that answers each role with schema-valid output.

    Exists so the orchestrator, the parsers and the cost routing can all be
    exercised end-to-end with no checkpoint, no GPU and no network — which is
    most of what "validate the plumbing" means before there is anything trained
    to validate it against.

    ``requires_planning`` is what a caller usually wants to steer, so it is a
    constructor knob rather than something to fake with a scripted transcript.
    """

    requires_planning: bool = False
    verdict: str = "pass"
    calls: list[tuple[AgentRole, str]] = field(default_factory=list)

    def generate(
        self,
        prompt: str,
        *,
        role: AgentRole,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Sequence[str] | None = None,
    ) -> str:
        self.calls.append((role, prompt))
        question = prompt.rstrip().splitlines()[-1].strip()

        if role is AgentRole.CONTEXT:
            return "The user has spoken to us before; no unresolved threads."
        if role is AgentRole.REASON:
            return (
                f"INTENT: answer the user's question\n"
                f"SUB_PROBLEM: understand {question}\n"
                f"REQUIRES_PLANNING: {'yes' if self.requires_planning else 'no'}"
            )
        if role is AgentRole.PLAN:
            return (
                "STEP: gather what the user already told us | read the brief\n"
                "STEP: answer the question | in the user's language"
            )
        if role is AgentRole.RESPOND:
            return "இது ஒரு சோதனை பதில்."  # "this is a test reply"
        if role is AgentRole.REFLECT:
            if self.verdict == "pass":
                return "VERDICT: pass"
            return "VERDICT: fail\nREASON: the draft does not answer the question"
        raise ModelClientError(f"unhandled role: {role}")

    def count(self, role: AgentRole) -> int:
        return sum(1 for r, _ in self.calls if r is role)


# --- HTTP backends ---------------------------------------------------------


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise ModelClientError(f"{url} returned HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise ModelClientError(f"{url} failed: {e}") from e


@dataclass
class OllamaClient:
    """Local Ollama — **development only**.

    Per §6 this stays on the dev box for iterating on prompts before a training
    run. It must not end up in a serving path: it is here so that editing
    ``prompts.py`` does not require a GPU, nothing more.
    """

    model: str = "qwen3:4b"
    host: str = "http://localhost:11434"
    timeout_s: float = DEFAULT_TIMEOUT_S

    def generate(
        self,
        prompt: str,
        *,
        role: AgentRole,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Sequence[str] | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if stop:
            payload["options"]["stop"] = list(stop)
        data = _post_json(f"{self.host.rstrip('/')}/api/generate", payload, self.timeout_s)
        return (data.get("response") or "").strip()


@dataclass
class VLLMClient:
    """Adhan served on vLLM — the production path.

    Speaks the OpenAI-compatible completions API that vLLM exposes. Left as raw
    completions rather than chat completions because the role scaffold already
    lives in the prompt; wrapping it in a chat template would put a second,
    competing notion of "role" on top of the control token.
    """

    model: str = "adhan"
    base_url: str = "http://localhost:8000"
    timeout_s: float = DEFAULT_TIMEOUT_S
    api_key: str | None = None

    def generate(
        self,
        prompt: str,
        *,
        role: AgentRole,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Sequence[str] | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = list(stop)
        url = f"{self.base_url.rstrip('/')}/v1/completions"
        data = _post_json(url, payload, self.timeout_s)
        try:
            return data["choices"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            raise ModelClientError(f"unexpected vLLM response shape: {data}") from e


# --- local checkpoint ------------------------------------------------------


@dataclass
class AdhanCheckpointClient:
    """Drive a local Adhan SLM checkpoint through ``adhan_slm.inference``.

    This is the "wire it to the existing Adhan checkpoint as-is" client from the
    Phase 1 plan. The checkpoint has had no role-tagged training yet, so its
    structured output will not parse well — that is the expected result, and the
    structure-compliance number it produces is the Phase 1 baseline the first
    training run has to beat.

    JAX is imported lazily, and only on first use, so importing the kernel never
    drags in the JAX stack.
    """

    config_path: str
    checkpoint_dir: str
    tokenizer_dir: str
    tokenizer_kind: str = "swaram"
    _loaded: tuple = field(default=(), repr=False, compare=False)

    def _ensure_loaded(self) -> tuple:
        if not self._loaded:
            try:
                from adhan_slm.inference import load_model, load_tokenizer
            except ImportError as e:  # pragma: no cover - depends on env
                raise ModelClientError(
                    f"AdhanCheckpointClient needs the adhan_slm package on PYTHONPATH ({e})"
                ) from e
            tokenizer = load_tokenizer(self.tokenizer_dir, kind=self.tokenizer_kind)
            model, params, _cfg = load_model(
                self.config_path, self.checkpoint_dir, vocab_size=len(tokenizer)
            )
            self._loaded = (model, params, tokenizer)
        return self._loaded

    def generate(
        self,
        prompt: str,
        *,
        role: AgentRole,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Sequence[str] | None = None,
    ) -> str:
        model, params, tokenizer = self._ensure_loaded()
        from adhan_slm.inference import generate_text

        try:
            text = generate_text(
                model,
                params,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:  # pragma: no cover - depends on checkpoint
            raise ModelClientError(f"generation failed for role {role.value}: {e}") from e
        # generate_text returns prompt + continuation; keep only the continuation.
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
