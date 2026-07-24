"""FastAPI server for Adhan SLM inference.

Usage:
    python scripts/run_api_server.py --port 8000 --model adhan-nano
"""

import argparse
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("FastAPI and uvicorn required. Install with: pip install fastapi uvicorn")
    exit(1)

from adhan_slm.core.logging import configure_root_logger, get_logger
from adhan_slm.serving.api import (
    AdhanInferenceAPI,
    AdhanRequest,
    ErrorResponse,
    TextResponse,
    TokensResponse,
)

logger = get_logger(__name__)


def create_app(model_name: str = "adhan-nano") -> FastAPI:
    """Create FastAPI application.

    Args:
        model_name: Name of the model to use

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Adhan SLM Inference API",
        description="REST API for Adhan Tamil Small Language Model inference",
        version="0.1.0",
    )

    # Initialize inference API
    api = AdhanInferenceAPI(model_name=model_name)

    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return api.health_check()

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
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Tokenization failed: {str(e)}"
            )

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
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Error handler
    @app.exception_handler(Exception)
    async def exception_handler(request, exc):
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
    parser = argparse.ArgumentParser(
        description="Run Adhan SLM inference API server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="adhan-nano",
        help="Model name (default: adhan-nano)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
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
