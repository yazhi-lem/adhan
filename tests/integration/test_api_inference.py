"""Integration tests for Adhan SLM inference API."""

import pytest

from adhan_slm.serving.api import (
    AdhanInferenceAPI,
    AdhanRequest,
    TextResponse,
    TokensResponse,
)


@pytest.mark.integration
class TestAdhanInferenceAPI:
    """Test Adhan Inference API."""

    @pytest.fixture
    def api(self) -> AdhanInferenceAPI:
        """Create API instance."""
        return AdhanInferenceAPI(model_name="adhan-nano")

    @pytest.mark.asyncio
    async def test_tokenize(self, api: AdhanInferenceAPI) -> None:
        """Test tokenization endpoint."""
        request = AdhanRequest(text="தமிழ் மொழி")
        response = await api.tokenize(request)

        assert isinstance(response, TokensResponse)
        assert response.text == "தமிழ் மொழி"
        assert len(response.tokens) > 0
        assert response.num_tokens == len(response.tokens)

    @pytest.mark.asyncio
    async def test_decode(self, api: AdhanInferenceAPI) -> None:
        """Test decoding endpoint."""
        request = AdhanRequest(text="1 2 3")  # Space-separated token IDs
        response = await api.decode(request)

        assert isinstance(response, TextResponse)
        assert len(response.text) > 0

    @pytest.mark.asyncio
    async def test_generate(self, api: AdhanInferenceAPI) -> None:
        """Test generation endpoint."""
        request = AdhanRequest(
            text="சொல், உனக்கு பிடித்த உணவு என்ன?",
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        response = await api.generate(request)

        assert isinstance(response, TextResponse)
        assert len(response.text) > 0
        assert "சொல், உனக்கு பிடித்த உணவு என்ன?" in response.text

    def test_health_check(self, api: AdhanInferenceAPI) -> None:
        """Test health check."""
        status = api.health_check()

        assert status["status"] == "ok"
        assert status["model"] == "adhan-nano"

    @pytest.mark.asyncio
    async def test_generate_with_params(self, api: AdhanInferenceAPI) -> None:
        """Test generation with various parameters."""
        request = AdhanRequest(
            text="நாம்",
            max_length=100,
            temperature=0.5,
            top_k=40,
            top_p=0.85,
            repetition_penalty=1.2,
        )
        response = await api.generate(request)

        assert isinstance(response, TextResponse)
        assert len(response.text) > 0

    @pytest.mark.asyncio
    async def test_tokenize_empty_text(self, api: AdhanInferenceAPI) -> None:
        """Test tokenization with empty text."""
        request = AdhanRequest(text="")
        response = await api.tokenize(request)

        assert response.text == ""
        assert response.num_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
