"""FastAPI server for Adhan SLM inference.

Usage:
    python scripts/run_api_server.py --port 8000 --model adhan-nano
"""

import argparse
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("FastAPI and uvicorn required. Install with: pip install fastapi uvicorn")
    exit(1)

from adhan_slm.core.logging import configure_root_logger, get_logger
from adhan_slm.serving.api import (
    AdhanInferenceAPI,
    AdhanRequest,
    ModelNotLoadedError,
    TextResponse,
    TokensResponse,
)

logger = get_logger(__name__)


# --- OpenAI-compatible schemas ---------------------------------------------
# Adhan's native API (/generate, /tokenize, /decode) is richer, but every
# evaluation harness in the ecosystem (tamil-bench, lm-eval, vLLM clients,
# Open WebUI) speaks the OpenAI chat-completions dialect. These models are the
# thin translation layer.
class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "adhan"
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = Field(default=256, ge=1)
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = False


def render_chat_prompt(messages: List[ChatMessage]) -> str:
    """Flatten an OpenAI message list into the plain-text prompt Adhan expects.

    Adhan is a base (not chat-tuned) SLM, so no special chat tokens are used;
    system and user turns are newline-separated and the answer is cued with a
    trailing newline. Once SFT alignment lands this becomes a real template.
    """
    parts: List[str] = []
    for m in messages:
        content = (m.content or "").strip()
        if not content:
            continue
        if m.role == "system":
            parts.append(content)
        elif m.role == "assistant":
            parts.append(content)
        else:
            parts.append(content)
    return "\n\n".join(parts) + "\n"


def create_app(
    model_name: str = "adhan-nano",
    api: Optional[AdhanInferenceAPI] = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        model_name: Name of the model to use
        api: Optional pre-configured AdhanInferenceAPI instance

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Adhan SLM Inference API",
        description="REST API for Adhan Tamil Small Language Model inference",
        version="0.1.0",
    )

    # Initialize inference API if not provided
    if api is None:
        api = AdhanInferenceAPI(model_name=model_name)

    # ModelNotLoadedError handler -> 503 Service Unavailable
    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        """Handle ModelNotLoadedError by returning HTTP 503 Service Unavailable."""
        logger.warning(f"Inference request rejected because model is not loaded: {exc}")
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "error_code": "MODEL_NOT_LOADED",
                "details": {
                    "model": api.model_name,
                    "status": "unavailable",
                    "message": "Inference server is running without a loaded model.",
                },
            },
        )

    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check(response: Response):
        """Health check endpoint.

        Returns 200 with model name when loaded, or 503 when no model is loaded.
        """
        health = api.health_check()
        if not api.is_loaded:
            response.status_code = 503
        return health

    # Tokenization endpoint
    @app.post("/tokenize", response_model=TokensResponse, tags=["Tokenization"])
    async def tokenize(request: AdhanRequest) -> TokensResponse:
        """Tokenize Tamil text.

        Args:
            request: Tokenization request with Tamil text

        Returns:
            TokensResponse with token IDs
        """
        try:
            return await api.tokenize(request)
        except ModelNotLoadedError:
            raise
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

    # Decoding endpoint
    @app.post("/decode", response_model=TextResponse, tags=["Tokenization"])
    async def decode(request: AdhanRequest) -> TextResponse:
        """Decode token IDs back to Tamil text.

        Args:
            request: Decode request with space-separated token IDs in text field

        Returns:
            TextResponse with decoded text
        """
        try:
            return await api.decode(request)
        except ModelNotLoadedError:
            raise
        except ValueError as e:
            logger.warning(f"Decoding input error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")

    # Generation endpoint
    @app.post("/generate", response_model=TextResponse, tags=["Generation"])
    async def generate(request: AdhanRequest) -> TextResponse:
        """Generate Tamil text from a prompt.

        Args:
            request: Generation request with prompt and sampling parameters

        Returns:
            TextResponse with generated text
        """
        try:
            return await api.generate(request)
        except ModelNotLoadedError:
            raise
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # --- OpenAI-compatible surface (tamil-bench / lm-eval / Open WebUI) -----
    @app.get("/v1/models", tags=["OpenAI-compatible"])
    async def list_models() -> Dict[str, Any]:
        """List the single served model in OpenAI format."""
        return {
            "object": "list",
            "data": [
                {
                    "id": api.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "yazhi",
                }
            ],
        }

    @app.post("/v1/chat/completions", tags=["OpenAI-compatible"])
    async def chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]:
        """Map an OpenAI chat-completion call onto Adhan generation."""
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported yet")

        prompt = render_chat_prompt(request.messages)
        adhan_request = AdhanRequest(
            text=prompt,
            max_length=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k or 50,
            top_p=request.top_p if request.top_p is not None else 0.9,
        )
        try:
            result = await api.generate(adhan_request)
        except ModelNotLoadedError:
            raise
        except Exception as e:
            logger.error(f"chat/completions error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or api.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": result.num_tokens or 0,
                "total_tokens": len(prompt.split()) + (result.num_tokens or 0),
            },
        }

    # Global error handler
    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "details": str(exc),
            },
        )

    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Adhan SLM inference API server")
    parser.add_argument(
        "--model",
        type=str,
        default="adhan-nano",
        help="Model name (default: adhan-nano)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on file changes",
    )

    args = parser.parse_args()

    # Configure logging
    configure_root_logger(level=args.log_level.upper())
    logger.info(f"Starting Adhan SLM API server (model: {args.model})")

    # Create app
    app = create_app(model_name=args.model)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
