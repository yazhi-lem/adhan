"""FastAPI server for Adhan SLM inference.

Usage:
    python scripts/run_api_server.py --port 8000 --model adhan-nano

``fastapi`` and ``uvicorn`` are optional (the ``serving`` extra). Importing this module must
never fail or exit when they are absent: tests import it, and an import-time ``exit()`` used to
kill pytest collection outright. The dependency is checked when the app is actually built.
"""

import argparse
import sys
from typing import Any, Optional

try:  # pragma: no cover - presence of the optional extra
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse

    SERVING_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - exercised by the regression test
    uvicorn = None  # type: ignore[assignment]
    FastAPI = HTTPException = Request = Response = JSONResponse = None  # type: ignore[assignment]
    SERVING_IMPORT_ERROR = str(exc)

from adhan_slm.core.logging import configure_root_logger, get_logger
from adhan_slm.serving.api import (
    AdhanInferenceAPI,
    AdhanRequest,
    ModelNotLoadedError,
    TextResponse,
    TokensResponse,
)

logger = get_logger(__name__)

SERVING_HINT = 'pip install -e ".[serving]"'


def require_serving_dependencies() -> None:
    """Raise a clear error when the optional ``serving`` extra is missing.

    Called from :func:`create_app` and :func:`main`, never at import time, so importing this
    module stays side-effect free.
    """
    if SERVING_IMPORT_ERROR is None:
        return
    raise ImportError(
        f"fastapi and uvicorn are required to serve the API ({SERVING_IMPORT_ERROR}). "
        f"Install them with: {SERVING_HINT}"
    )


def create_app(
    model_name: str = "adhan-nano",
    api: Optional[AdhanInferenceAPI] = None,
) -> Any:
    """Create FastAPI application.

    Args:
        model_name: Name of the model to use
        api: Optional pre-configured AdhanInferenceAPI instance

    Returns:
        FastAPI application instance

    Raises:
        ImportError: when the ``serving`` extra is not installed.
    """
    require_serving_dependencies()
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


def main() -> int:
    """Main entry point. Returns a process exit code instead of calling exit()."""
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

    if SERVING_IMPORT_ERROR is not None:
        print(
            f"FastAPI and uvicorn are required to run the API server: {SERVING_IMPORT_ERROR}\n"
            f"Install them with: {SERVING_HINT}",
            file=sys.stderr,
        )
        return 2

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
