"""Integration tests for Adhan SLM inference API and FastAPI serving."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.serving.api import (  # noqa: E402
    AdhanInferenceAPI,
    AdhanRequest,
    ModelNotLoadedError,
    TextResponse,
    TokensResponse,
)
from adhan_slm.tokenizer import SwaramTokenizer  # noqa: E402
from scripts.run_api_server import create_app  # noqa: E402

try:
    from fastapi.testclient import TestClient

    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False


@pytest.fixture(scope="module")
def sample_tokenizer() -> SwaramTokenizer:
    """Provide a real, fast-trained in-memory SwaramTokenizer for testing."""
    sample_corpus = [
        "தமிழ் மொழி மிகவும் தொன்மையானது.",
        "சொல், உனக்கு பிடித்த உணவு என்ன?",
        "நாம் அனைவரும் ஒன்றாக வாழ்வோம்.",
    ]
    return SwaramTokenizer.train(sample_corpus, vocab_size=64, min_freq=1)


@pytest.mark.integration
class TestAdhanInferenceAPIUnloaded:
    """Test API behavior when no model or checkpoint is loaded (standby mode)."""

    @pytest.fixture
    def unloaded_api(self) -> AdhanInferenceAPI:
        """Create API instance without valid checkpoint/tokenizer paths."""
        return AdhanInferenceAPI(
            model_name="adhan-nano",
            checkpoint_dir="/nonexistent/checkpoints/adhan-nano",
            tokenizer_dir="/nonexistent/data/tokenizer",
        )

    def test_health_check_returns_unavailable(self, unloaded_api: AdhanInferenceAPI) -> None:
        """Health check must reflect that no model is loaded."""
        assert not unloaded_api.is_loaded
        status = unloaded_api.health_check()
        assert status["status"] == "unavailable"
        assert status["model"] == "adhan-nano"
        assert "error" in status

    @pytest.mark.asyncio
    async def test_tokenize_raises_model_not_loaded(self, unloaded_api: AdhanInferenceAPI) -> None:
        """Tokenize must reject calls with ModelNotLoadedError."""
        with pytest.raises(ModelNotLoadedError, match="No model or tokenizer is loaded"):
            await unloaded_api.tokenize(AdhanRequest(text="தமிழ் மொழி"))

    @pytest.mark.asyncio
    async def test_decode_raises_model_not_loaded(self, unloaded_api: AdhanInferenceAPI) -> None:
        """Decode must reject calls with ModelNotLoadedError."""
        with pytest.raises(ModelNotLoadedError, match="No model or tokenizer is loaded"):
            await unloaded_api.decode(AdhanRequest(text="1 2 3"))

    @pytest.mark.asyncio
    async def test_generate_raises_model_not_loaded(self, unloaded_api: AdhanInferenceAPI) -> None:
        """Generate must reject calls with ModelNotLoadedError, never return canned output."""
        with pytest.raises(ModelNotLoadedError, match="No model is loaded"):
            await unloaded_api.generate(AdhanRequest(text="சொல், உனக்கு பிடித்த உணவு என்ன?"))


@pytest.mark.integration
class TestAdhanInferenceAPILoaded:
    """Test API behavior when real model and tokenizer are loaded."""

    @pytest.fixture
    def loaded_api(self, sample_tokenizer: SwaramTokenizer) -> AdhanInferenceAPI:
        """Create API instance with injected real tokenizer and mock model/params."""
        return AdhanInferenceAPI(
            model_name="adhan-nano",
            tokenizer=sample_tokenizer,
            model=object(),
            params={"params": {}},
        )

    def test_health_check_returns_ok_when_loaded(self, loaded_api: AdhanInferenceAPI) -> None:
        """Health check returns status ok when real components are loaded."""
        assert loaded_api.is_loaded
        status = loaded_api.health_check()
        assert status == {"status": "ok", "model": "adhan-nano"}

    @pytest.mark.asyncio
    async def test_tokenize_real(
        self, loaded_api: AdhanInferenceAPI, sample_tokenizer: SwaramTokenizer
    ) -> None:
        """Tokenize invokes the real tokenizer."""
        request = AdhanRequest(text="தமிழ் மொழி")
        response = await loaded_api.tokenize(request)

        expected_ids = sample_tokenizer.encode("தமிழ் மொழி", add_special=False)
        assert isinstance(response, TokensResponse)
        assert response.text == "தமிழ் மொழி"
        assert response.tokens == expected_ids
        assert response.num_tokens == len(expected_ids)

    @pytest.mark.asyncio
    async def test_decode_real(
        self, loaded_api: AdhanInferenceAPI, sample_tokenizer: SwaramTokenizer
    ) -> None:
        """Decode invokes the real tokenizer decode."""
        token_ids = sample_tokenizer.encode("தமிழ் மொழி", add_special=False)
        request = AdhanRequest(text=" ".join(str(x) for x in token_ids))
        response = await loaded_api.decode(request)

        assert isinstance(response, TextResponse)
        assert "தமிழ்" in response.text

    @pytest.mark.asyncio
    async def test_generate_calls_real_inference(self, loaded_api: AdhanInferenceAPI) -> None:
        """Generate calls the real generate_text pipeline."""
        with patch("adhan_slm.inference.generate_text", return_value="தமிழ் வாழ்க"):
            request = AdhanRequest(text="தமிழ்", temperature=0.8, top_k=40)
            response = await loaded_api.generate(request)

            assert isinstance(response, TextResponse)
            assert response.text == "தமிழ் வாழ்க"

    @pytest.mark.asyncio
    async def test_decode_malformed_input_raises_value_error(
        self, loaded_api: AdhanInferenceAPI
    ) -> None:
        """Decode rejects non-integer token strings with ValueError."""
        with pytest.raises(ValueError, match="Invalid token IDs format"):
            await loaded_api.decode(AdhanRequest(text="not integer tokens"))


@pytest.mark.integration
class TestFastAPIServerEndpoints:
    """Test HTTP status codes and error bodies served by FastAPI."""

    @pytest.fixture(autouse=True)
    def check_testclient(self):
        if not HAS_TESTCLIENT:
            pytest.skip("fastapi.testclient (httpx) is not installed")

    def test_http_health_unloaded_returns_503(self) -> None:
        """GET /health must return HTTP 503 when no model is loaded."""
        api = AdhanInferenceAPI(
            model_name="adhan-nano",
            checkpoint_dir="/nonexistent/checkpoints/adhan-nano",
            tokenizer_dir="/nonexistent/data/tokenizer",
        )
        app = create_app(model_name="adhan-nano", api=api)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["model"] == "adhan-nano"

    def test_http_generate_unloaded_returns_503(self) -> None:
        """POST /generate must return HTTP 503 with MODEL_NOT_LOADED error code."""
        api = AdhanInferenceAPI(
            model_name="adhan-nano",
            checkpoint_dir="/nonexistent/checkpoints/adhan-nano",
            tokenizer_dir="/nonexistent/data/tokenizer",
        )
        app = create_app(model_name="adhan-nano", api=api)
        client = TestClient(app)

        response = client.post("/generate", json={"text": "சொல்"})
        assert response.status_code == 503
        data = response.json()
        assert data["error_code"] == "MODEL_NOT_LOADED"
        assert "adhan-nano" in data["error"]

    def test_http_tokenize_and_decode_unloaded_returns_503(self) -> None:
        """POST /tokenize and /decode must return HTTP 503 when uninitialized."""
        api = AdhanInferenceAPI(
            model_name="adhan-nano",
            checkpoint_dir="/nonexistent/checkpoints/adhan-nano",
            tokenizer_dir="/nonexistent/data/tokenizer",
        )
        app = create_app(model_name="adhan-nano", api=api)
        client = TestClient(app)

        res_tok = client.post("/tokenize", json={"text": "தமிழ்"})
        assert res_tok.status_code == 503
        assert res_tok.json()["error_code"] == "MODEL_NOT_LOADED"

        res_dec = client.post("/decode", json={"text": "1 2 3"})
        assert res_dec.status_code == 503
        assert res_dec.json()["error_code"] == "MODEL_NOT_LOADED"

    def test_http_health_loaded_returns_200(self, sample_tokenizer: SwaramTokenizer) -> None:
        """GET /health must return HTTP 200 when model is loaded."""
        loaded_api = AdhanInferenceAPI(
            model_name="adhan-nano",
            tokenizer=sample_tokenizer,
            model=object(),
            params={"params": {}},
        )
        app = create_app(model_name="adhan-nano", api=loaded_api)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model": "adhan-nano"}

    def test_http_decode_malformed_returns_400(self, sample_tokenizer: SwaramTokenizer) -> None:
        """POST /decode with non-integer token IDs returns HTTP 400 Bad Request."""
        loaded_api = AdhanInferenceAPI(
            model_name="adhan-nano",
            tokenizer=sample_tokenizer,
            model=object(),
            params={"params": {}},
        )
        app = create_app(model_name="adhan-nano", api=loaded_api)
        client = TestClient(app)

        res = client.post("/decode", json={"text": "abc def"})
        assert res.status_code == 400
        assert "Invalid token IDs format" in res.json()["detail"]
