"""FastAPI inference server for Adhan SLM.

Provides REST endpoints for tokenization, decoding, and text generation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)


class AdhanRequest(BaseModel):
    """Request schema for Adhan inference endpoints."""

    text: str = Field(..., description="Input Tamil text to process")
    max_length: Optional[int] = Field(
        None, description="Maximum length for generation"
    )
    temperature: Optional[float] = Field(
        0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_k: Optional[int] = Field(50, ge=1, description="Top-k sampling")
    top_p: Optional[float] = Field(
        0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling"
    )
    repetition_penalty: Optional[float] = Field(
        1.0, ge=1.0, description="Repetition penalty"
    )


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

    def __init__(self, model_name: str = "adhan-nano") -> None:
        """Initialize the inference API.

        Args:
            model_name: Name of the model to use (e.g., 'adhan-nano', 'adhan-tiny')
        """
        self.model_name = model_name
        logger.info(f"Initialized Adhan Inference API for {model_name}")

    async def tokenize(self, request: AdhanRequest) -> TokensResponse:
        """Tokenize Tamil text.

        Args:
            request: Tokenization request with Tamil text

        Returns:
            TokensResponse with token IDs and metadata
        """
        try:
            text = request.text
            logger.info(f"Tokenizing text: {text[:50]}...")

            # Placeholder: would load actual tokenizer here
            token_ids = [] if text == "" else [1, 2, 3]  # Placeholder

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
        try:
            # Parse space-separated token IDs from text field
            token_ids = [int(x) for x in request.text.split()]
            logger.info(f"Decoding {len(token_ids)} tokens")

            # Placeholder: would load actual tokenizer here
            # decoded_text = self.tokenizer.decode(token_ids)
            decoded_text = "தமிழ் மொழி"  # Placeholder

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
        try:
            logger.info(f"Generating text from prompt: {request.text[:50]}...")
            logger.info(
                f"Parameters: temp={request.temperature}, top_k={request.top_k}, "
                f"top_p={request.top_p}, repetition_penalty={request.repetition_penalty}"
            )

            # Placeholder: would load model and tokenizer here
            # tokens = self.tokenizer.encode(request.text)
            # generated_ids = self.model.generate(tokens, max_length=request.max_length, ...)
            # generated_text = self.tokenizer.decode(generated_ids)

            generated_text = request.text + " சொல்வதாக விளக்கினார்."  # Placeholder

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
            Status dictionary
        """
        logger.debug(f"Health check: {self.model_name} is running")
        return {"status": "ok", "model": self.model_name}
