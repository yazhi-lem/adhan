"""FastAPI inference server for Adhan SLM.

Provides REST endpoints for tokenization, decoding, and text generation.
Production features:
  * Async model loading with healthz differentiation (loading vs ready vs error)
  * Request validation via Pydantic v2
  * Rate limiting (token bucket) per client IP
  * Request timeouts on generation to prevent hung connections
  * Batch inference endpoint for throughput
  * Structured error responses with machine-readable codes
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when an inference operation is requested but no model is loaded."""


class ModelLoadingError(RuntimeError):
    """Raised when model loading fails terminally."""


# --------------------------------------------------------------------------- #
# Request / Response schemas
# --------------------------------------------------------------------------- #


class AdhanRequest(BaseModel):
    """Request schema for Adhan inference endpoints."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=16384,
        description="Input Tamil text to process (max 16K chars)",
    )
    max_length: Optional[int] = Field(None, ge=1, le=2048, description="Maximum length for generation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(50, ge=1, le=10000, description="Top-k sampling")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    repetition_penalty: Optional[float] = Field(
        1.0, ge=1.0, le=2.0, description="Repetition penalty"
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be non-empty")
        return v


class BatchGenerateRequest(BaseModel):
    """Batch generation request schema."""

    items: List[AdhanRequest] = Field(..., min_length=1, max_length=32)
    timeout_sec: Optional[float] = Field(30.0, ge=1.0, le=300.0)


class TokensResponse(BaseModel):
    """Response schema for tokenization."""

    tokens: List[int] = Field(..., description="List of token IDs")
    token_ids: List[int] = Field(..., description="Alias for tokens")
    num_tokens: int = Field(..., description="Number of tokens")
    text: str = Field(..., description="Original input text")


class TextResponse(BaseModel):
    """Response schema for text generation/decoding."""

    text: str = Field(..., description="Generated or decoded text")
    num_tokens: Optional[int] = Field(None, description="Number of tokens in response")


class ReasoningResponse(BaseModel):
    """Response schema with structured Tamil reasoning."""

    thought: Optional[str] = Field(None, description="சிந்தனை — initial analysis")
    rationalize: Optional[str] = Field(None, description="ஆய்வு — step-by-step investigation")
    reflect: Optional[str] = Field(None, description="பரிசீலனை — self-correction and doubt")
    answer: str = Field(..., description="பதில் — final answer")
    citations: List[str] = Field(default_factory=list, description="மேற்கோள்கள்")
    raw: str = Field(..., description="Raw model output")
    num_tokens: Optional[int] = Field(None, description="Number of tokens in response")


class BatchTextResponse(BaseModel):
    """Batch generation response schema."""

    results: List[TextResponse]
    total_tokens: int
    wall_ms: float


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Machine-readable error code")
    details: Optional[dict] = Field(None, description="Additional error details")


# --------------------------------------------------------------------------- #
# Rate limiter: token bucket per client IP
# --------------------------------------------------------------------------- #


class TokenBucket:
    """Thread-safe token bucket for per-IP rate limiting."""

    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self.rate = rate
        self.capacity = capacity
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        async with self._lock:
            last = self._last.get(key, now)
            bucket = min(
                self.capacity, self._tokens.get(key, self.capacity) + self.rate * (now - last)
            )
            self._last[key] = now
            if bucket >= tokens:
                self._tokens[key] = bucket - tokens
                return True
            self._tokens[key] = bucket
            return False


# --------------------------------------------------------------------------- #
# Inference API
# --------------------------------------------------------------------------- #


class AdhanInferenceAPI:
    """Inference API for Adhan SLM models.

    Handles tokenization, decoding, and text generation.
    Supports async background loading so the server starts immediately
    and reports /healthz status transitions: loading -> ok -> error.
    """

    def __init__(
        self,
        model_name: str = "adhan-nano",
        checkpoint_dir: Optional[Union[str, Path]] = None,
        tokenizer_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        model: Optional[Any] = None,
        params: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_concurrent_generations: int = 8,
        generation_timeout_sec: float = 60.0,
        defer_load: bool = False,
    ) -> None:
        """Initialize the inference API.

        Args:
            model_name: Name of the model to use (e.g., 'adhan-nano', 'adhan-tiny')
            checkpoint_dir: Path to directory containing Orbax checkpoint
            tokenizer_dir: Path to directory containing vocab.json and merges.txt
            config_path: Path to model config YAML
            model: Optional pre-loaded model instance
            params: Optional pre-loaded model parameters
            tokenizer: Optional pre-loaded tokenizer instance
            max_concurrent_generations: Semaphore to bound concurrent generation CPU
            generation_timeout_sec: Wall-clock timeout for a single generate() call
            defer_load: If True, do not attempt a synchronous load in the constructor;
                the first request (or the server's background task) loads instead.
                Default False preserves the eager, immediately-introspectable state.
        """
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir or os.environ.get("ADHAN_CHECKPOINT_DIR")
        self.tokenizer_dir = tokenizer_dir or os.environ.get("ADHAN_TOKENIZER_DIR")
        self.config_path = config_path or os.environ.get("ADHAN_CONFIG_PATH")

        self.model = model
        self.params = params
        self.tokenizer = tokenizer

        self._status = "loading"
        self.load_error: Optional[str] = None
        self._gen_semaphore = asyncio.Semaphore(max_concurrent_generations)
        self._generation_timeout_sec = generation_timeout_sec

        logger.info(f"Initializing Adhan Inference API for {model_name}")
        self._loaded = False
        self._load_lock = asyncio.Lock()

        if not defer_load:
            # Eager path (legacy behaviour): try to load now so callers can read
            # `is_loaded` / health immediately. Failures are captured, not raised,
            # so the server can still start in standby mode and report 503.
            try:
                self._initialize_pipeline()
                self._loaded = True
                self._status = "ok"
            except Exception as e:  # noqa: BLE001 - standby mode is intentional
                self._loaded = False
                self._status = "error"
                self.load_error = str(e)
                logger.warning(
                    f"Could not load real model '{self.model_name}': {e}. "
                    f"Server will operate in standby mode (health check returns 503)."
                )

    @property
    def is_loaded(self) -> bool:
        """True when a model+tokenizer are ready to serve."""
        return bool(self._loaded) and self._status == "ok"

    async def _ensure_loaded(self) -> None:
        """Idempotent async loader used only in deferred mode.

        Never raises: on failure it records ``_status = "error"`` and returns, so
        callers can raise the appropriate :class:`ModelNotLoadedError` with a
        service-specific message. This mirrors the eager path, which captures
        load failures instead of propagating them.
        """
        if self.is_loaded or self._status == "error":
            return
        async with self._load_lock:
            if self.is_loaded or self._status == "error":
                return
            try:
                await asyncio.to_thread(self._initialize_pipeline)
                self._loaded = True
                self._status = "ok"
            except Exception as e:  # noqa: BLE001 - standby mode is intentional
                self._status = "error"
                self.load_error = str(e)
                self._loaded = False
                logger.warning("Model loading failed: %s. Server stays in standby.", e)

    def _initialize_pipeline(self) -> None:
        """Attempt to load real tokenizer and model without crashing on failure."""
        if self.tokenizer is not None and self.model is not None and self.params is not None:
            self._status = "ok"
            logger.info(
                f"Inference pipeline initialized from injected instances for {self.model_name}"
            )
            return

        # Check default paths if not explicitly provided
        if not self.checkpoint_dir:
            default_ckpt = Path("checkpoints") / self.model_name
            if default_ckpt.exists():
                self.checkpoint_dir = default_ckpt

        if not self.tokenizer_dir:
            default_tok = Path("checkpoints") / self.model_name
            if default_tok.exists() and (default_tok / "vocab.json").exists():
                self.tokenizer_dir = default_tok
            elif Path("data/tokenizer").exists():
                self.tokenizer_dir = Path("data/tokenizer")

        from adhan_slm.inference import load_model, load_tokenizer

        if not self.tokenizer_dir or not Path(self.tokenizer_dir).exists():
            raise FileNotFoundError(
                "Tokenizer directory not found. Please provide a valid tokenizer directory with vocab.json and merges.txt."
            )

        if not self.checkpoint_dir or not Path(self.checkpoint_dir).exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found for model '{self.model_name}'. Please train or provide a model checkpoint."
            )

        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.tokenizer_dir)

        if self.model is None or self.params is None:
            self.model, self.params, _ = load_model(
                config_path=self.config_path,
                checkpoint_dir=self.checkpoint_dir,
            )

        self._status = "ok"
        logger.info(f"Successfully loaded model and tokenizer for {self.model_name}")

    async def tokenize(self, request: AdhanRequest) -> TokensResponse:
        """Tokenize Tamil text."""
        await self._ensure_loaded()
        if not self.is_loaded or self.tokenizer is None:
            raise ModelNotLoadedError(
                f"No model or tokenizer is loaded for '{self.model_name}'. "
                f"Cannot tokenize text without a loaded tokenizer."
            )
        try:
            text = request.text
            logger.info(f"Tokenizing text: {text[:50]}...")
            token_ids = self.tokenizer.encode(text, add_special=False)
            return TokensResponse(
                tokens=token_ids,
                token_ids=token_ids,
                num_tokens=len(token_ids),
                text=request.text,
            )
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise

    async def decode(self, request: AdhanRequest) -> TextResponse:
        """Decode token IDs back to text."""
        await self._ensure_loaded()
        if not self.is_loaded or self.tokenizer is None:
            raise ModelNotLoadedError(
                f"No model or tokenizer is loaded for '{self.model_name}'. "
                f"Cannot decode tokens without a loaded tokenizer."
            )
        try:
            try:
                token_ids = [int(x) for x in request.text.split()]
            except ValueError as e:
                raise ValueError(
                    f"Invalid token IDs format: expected space-separated integers, got '{request.text}'"
                ) from e

            decoded_text = self.tokenizer.decode(token_ids)
            return TextResponse(text=decoded_text, num_tokens=len(token_ids))
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise

    async def generate(self, request: AdhanRequest) -> TextResponse:
        """Generate text from a prompt with timeout and concurrency control."""
        await self._ensure_loaded()
        if (
            not self.is_loaded
            or self.model is None
            or self.params is None
            or self.tokenizer is None
        ):
            raise ModelNotLoadedError(
                f"No model is loaded for '{self.model_name}'. "
                f"Cannot generate text without a loaded model and tokenizer."
            )

        async with self._gen_semaphore:
            try:
                from adhan_slm.inference import generate_text

                gen_kw = {}
                if request.max_length is not None:
                    gen_kw["max_new_tokens"] = request.max_length
                if request.temperature is not None:
                    gen_kw["temperature"] = request.temperature
                if request.top_k is not None:
                    gen_kw["top_k"] = request.top_k
                if request.top_p is not None:
                    gen_kw["top_p"] = request.top_p
                if request.repetition_penalty is not None:
                    gen_kw["repetition_penalty"] = request.repetition_penalty

                # Run generation in thread pool with timeout
                loop = asyncio.get_event_loop()
                generated_text = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: generate_text(
                            self.model,
                            self.params,
                            self.tokenizer,
                            prompt=request.text,
                            **gen_kw,
                        ),
                    ),
                    timeout=self._generation_timeout_sec,
                )

                return TextResponse(
                    text=generated_text,
                    num_tokens=len(generated_text.split()),
                )
            except asyncio.TimeoutError:
                logger.error(f"Generation timed out after {self._generation_timeout_sec}s")
                raise asyncio.TimeoutError("Generation request timed out")
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise

    async def batch_generate(self, request: BatchGenerateRequest) -> BatchTextResponse:
        """Batch generation with bounded concurrency and per-item timeout."""
        await self._ensure_loaded()
        t0 = time.perf_counter()
        results: List[TextResponse] = []
        total_tokens = 0

        async def _one(req: AdhanRequest) -> TextResponse:
            try:
                r = await asyncio.wait_for(
                    self.generate(req),
                    timeout=request.timeout_sec,
                )
                return r
            except asyncio.TimeoutError:
                return TextResponse(text="[timeout]", num_tokens=0)
            except Exception as e:
                return TextResponse(text=f"[error: {e}]", num_tokens=0)

        # Limit concurrency within the batch to avoid thundering herd
        sem = asyncio.Semmaphore(4)

        async def _bounded(req: AdhanRequest) -> TextResponse:
            async with sem:
                return await _one(req)

        results = await asyncio.gather(*[_bounded(r) for r in request.items])
        total_tokens = sum(r.num_tokens or 0 for r in results)
        return BatchTextResponse(
            results=results,
            total_tokens=total_tokens,
            wall_ms=(time.perf_counter() - t0) * 1000.0,
        )

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> dict:
        """Protocol-aware chat completion (OpenAI message list -> Tamil answer)."""
        await self._ensure_loaded()
        if (
            not self.is_loaded
            or self.model is None
            or self.params is None
            or self.tokenizer is None
        ):
            raise ModelNotLoadedError(
                f"No model is loaded for '{self.model_name}'. Cannot chat without a model."
            )

        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        system = None
        convo = []
        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role == "system":
                system = content
            elif role in ("user", "assistant"):
                convo.append({"role": role, "content": content})
            else:
                raise ValueError(f"unsupported role {role!r}")
        if not convo:
            raise ValueError("messages must contain at least one user/assistant turn")

        from adhan_slm import protocol as p
        from adhan_slm.inference import generate_text

        prompt = p.render_prompt(convo, system=system)

        async with self._gen_semaphore:
            loop = asyncio.get_event_loop()
            text = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: generate_text(
                        self.model,
                        self.params,
                        self.tokenizer,
                        prompt=prompt,
                        max_new_tokens=max(1, int(max_tokens)),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                        repetition_penalty=1.0,
                        include_prompt=False,
                    ),
                ),
                timeout=self._generation_timeout_sec,
            )
        return {
            "text": text,
            "prompt_tokens": len(self.tokenizer.encode(prompt)),
            "completion_tokens": len(self.tokenizer.encode(text)),
        }

    def health_check(self) -> dict:
        """Health check endpoint with granular status."""
        if self._status == "error" or (not self.is_loaded and self._status != "loading"):
            return {
                "status": "unavailable",
                "model": self.model_name,
                "error": self.load_error or "No model loaded",
            }
        if self._status == "loading":
            return {
                "status": "loading",
                "model": self.model_name,
                "note": "Model is being loaded in the background",
            }
        return {"status": "ok", "model": self.model_name}


# --------------------------------------------------------------------------- #
# FastAPI app factory
# --------------------------------------------------------------------------- #


def create_app(
    model_name: str = "adhan-nano",
    checkpoint_dir: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    rate_limit_rate: float = 10.0,
    rate_limit_capacity: float = 20.0,
) -> Any:
    """Create a configured FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException, Request, status
    except ImportError as e:
        raise ImportError(
            f"FastAPI is required for serving: {e}. Install: pip install fastapi uvicorn"
        )

    api = AdhanInferenceAPI(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        tokenizer_dir=tokenizer_dir,
        config_path=config_path,
    )
    limiter = TokenBucket(rate=rate_limit_rate, capacity=rate_limit_capacity)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Trigger background load on startup
        asyncio.create_task(api._ensure_loaded())
        yield
        # Cleanup handled by process exit

    app = FastAPI(
        title="Adhan SLM Inference API",
        description="Tamil-native small language model inference",
        version="1.0.0",
        lifespan=lifespan,
    )

    async def _check_rate(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if not await limiter.acquire(client):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Slow down.",
            )

    @app.get("/healthz")
    async def healthz():
        return api.health_check()

    @app.post("/tokenize", response_model=TokensResponse)
    async def tokenize_endpoint(request: AdhanRequest, req: Request):
        await _check_rate(req)
        try:
            return await api.tokenize(request)
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("tokenize error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/decode", response_model=TextResponse)
    async def decode_endpoint(request: AdhanRequest, req: Request):
        await _check_rate(req)
        try:
            return await api.decode(request)
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("decode error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate", response_model=TextResponse)
    async def generate_endpoint(request: AdhanRequest, req: Request):
        await _check_rate(req)
        try:
            return await api.generate(request)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Generation timed out")
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("generate error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch_generate", response_model=BatchTextResponse)
    async def batch_generate_endpoint(request: BatchGenerateRequest, req: Request):
        await _check_rate(req)
        try:
            return await api.batch_generate(request)
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("batch_generate error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate/reasoning", response_model=ReasoningResponse)
    async def generate_reasoning_endpoint(request: AdhanRequest, req: Request):
        """Generate with structured Tamil reasoning (சிந்தனை → ஆய்வு → பரிசீலனை → பதில்)."""
        await _check_rate(req)
        try:
            from adhan_slm import protocol as proto
            from adhan_slm.inference import generate_text

            await api._ensure_loaded()
            gen_kw = {}
            if request.max_length is not None:
                gen_kw["max_new_tokens"] = request.max_length
            if request.temperature is not None:
                gen_kw["temperature"] = request.temperature
            if request.top_k is not None:
                gen_kw["top_k"] = request.top_k
            if request.top_p is not None:
                gen_kw["top_p"] = request.top_p
            if request.repetition_penalty is not None:
                gen_kw["repetition_penalty"] = request.repetition_penalty

            loop = asyncio.get_event_loop()
            raw_text = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: generate_text(
                        api.model,
                        api.params,
                        api.tokenizer,
                        prompt=request.text,
                        **gen_kw,
                    ),
                ),
                timeout=api._generation_timeout_sec,
            )
            parsed = proto.parse_response(raw_text)
            return ReasoningResponse(
                thought=parsed.get("thought"),
                rationalize=parsed.get("rationalize"),
                reflect=parsed.get("reflect"),
                answer=parsed.get("answer", raw_text),
                citations=parsed.get("citations", []),
                raw=raw_text,
                num_tokens=len(raw_text.split()),
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Generation timed out")
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("generate_reasoning error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/chat/completions")
    async def chat_endpoint(raw: dict, req: Request):
        """OpenAI-compatible chat completions endpoint."""
        await _check_rate(req)
        messages = raw.get("messages", [])
        max_tokens = raw.get("max_tokens", 256)
        temperature = raw.get("temperature", 0.0)
        top_p = raw.get("top_p", 0.95)
        top_k = raw.get("top_k", 40)
        try:
            result = await api.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            return {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": api.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
            }
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Generation timed out")
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception("chat error")
            raise HTTPException(status_code=500, detail=str(e))

    return app
