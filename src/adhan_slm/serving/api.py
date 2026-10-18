"""FastAPI inference server for Adhan SLM.

Provides REST endpoints for tokenization, decoding, and text generation.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when an inference operation is requested but no model is loaded."""


class AdhanRequest(BaseModel):
    """Request schema for Adhan inference endpoints."""

    text: str = Field(..., description="Input Tamil text to process")
    max_length: Optional[int] = Field(None, description="Maximum length for generation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(50, ge=1, description="Top-k sampling")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    repetition_penalty: Optional[float] = Field(1.0, ge=1.0, description="Repetition penalty")


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


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Machine-readable error code")
    details: Optional[dict] = Field(None, description="Additional error details")


class AdhanInferenceAPI:
    """Inference API for Adhan SLM models.

    Handles tokenization, decoding, and text generation.
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
        """
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir or os.environ.get("ADHAN_CHECKPOINT_DIR")
        self.tokenizer_dir = tokenizer_dir or os.environ.get("ADHAN_TOKENIZER_DIR")
        self.config_path = config_path or os.environ.get("ADHAN_CONFIG_PATH")

        self.model = model
        self.params = params
        self.tokenizer = tokenizer

        self.is_loaded = False
        self.load_error: Optional[str] = None

        logger.info(f"Initializing Adhan Inference API for {model_name}")
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Attempt to load real tokenizer and model without crashing on failure."""
        if self.tokenizer is not None and self.model is not None and self.params is not None:
            self.is_loaded = True
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

        try:
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

            self.is_loaded = True
            logger.info(f"Successfully loaded model and tokenizer for {self.model_name}")
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            logger.warning(
                f"Could not load real model '{self.model_name}': {e}. "
                f"Server will operate in standby mode (health check will return 503)."
            )

    async def tokenize(self, request: AdhanRequest) -> TokensResponse:
        """Tokenize Tamil text.

        Args:
            request: Tokenization request with Tamil text

        Returns:
            TokensResponse with token IDs and metadata
        """
        if not self.is_loaded or self.tokenizer is None:
            raise ModelNotLoadedError(
                f"No model or tokenizer is loaded for '{self.model_name}'. "
                f"Cannot tokenize text without a loaded tokenizer."
            )
        try:
            text = request.text
            logger.info(f"Tokenizing text: {text[:50]}...")

            token_ids = self.tokenizer.encode(text, add_special=False)

            response = TokensResponse(
                tokens=token_ids,
                token_ids=token_ids,
                num_tokens=len(token_ids),
                text=request.text,
            )

            logger.info(f"Successfully tokenized {len(token_ids)} tokens")
            return response

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise

    async def decode(self, request: AdhanRequest) -> TextResponse:
        """Decode token IDs back to text.

        Args:
            request: Decode request with token IDs (as space-separated integers in text)

        Returns:
            TextResponse with decoded text
        """
        if not self.is_loaded or self.tokenizer is None:
            raise ModelNotLoadedError(
                f"No model or tokenizer is loaded for '{self.model_name}'. "
                f"Cannot decode tokens without a loaded tokenizer."
            )
        try:
            try:
                token_ids = [int(x) for x in request.text.split()]
            except ValueError as e:
                logger.warning(f"Failed to parse token IDs from text '{request.text[:50]}': {e}")
                raise ValueError(
                    f"Invalid token IDs format: expected space-separated integers, got '{request.text}'"
                ) from e

            logger.info(f"Decoding {len(token_ids)} tokens")

            decoded_text = self.tokenizer.decode(token_ids)

            response = TextResponse(text=decoded_text, num_tokens=len(token_ids))

            logger.info(f"Successfully decoded tokens to: {decoded_text}")
            return response

        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise

    async def generate(self, request: AdhanRequest) -> TextResponse:
        """Generate text from a prompt.

        Args:
            request: Generation request with prompt and sampling parameters

        Returns:
            TextResponse with generated text
        """
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
        try:
            from adhan_slm.inference import generate_text

            logger.info(f"Generating text from prompt: {request.text[:50]}...")
            logger.info(
                f"Parameters: temp={request.temperature}, top_k={request.top_k}, "
                f"top_p={request.top_p}, repetition_penalty={request.repetition_penalty}"
            )

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

            generated_text = generate_text(
                self.model,
                self.params,
                self.tokenizer,
                prompt=request.text,
                **gen_kw,
            )

            response = TextResponse(
                text=generated_text,
                num_tokens=len(generated_text.split()),
            )

            logger.info(f"Successfully generated {len(generated_text)} characters")
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def health_check(self) -> dict:
        """Health check endpoint.

        Returns:
            Status dictionary reflecting whether real model is loaded
        """
        if not self.is_loaded:
            logger.warning(f"Health check: {self.model_name} is not loaded ({self.load_error})")
            return {
                "status": "unavailable",
                "model": self.model_name,
                "error": self.load_error or "No model loaded",
            }

        logger.debug(f"Health check: {self.model_name} is running")
        return {"status": "ok", "model": self.model_name}
