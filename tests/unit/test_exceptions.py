import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.exceptions import (
    EXCEPTION_HANDLERS,
    AnalysisError,
    InsufficientReadingsError,
    ModelNotLoadedError,
    analysis_error_handler,
    insufficient_readings_handler,
    model_not_loaded_handler,
)

pytestmark = pytest.mark.unit


class TestExceptionClasses:
    def test_insufficient_readings_error(self) -> None:
        exc = InsufficientReadingsError(required=12, actual=5)
        assert exc.required == 12
        assert exc.actual == 5
        assert "12 required" in str(exc)
        assert "5 provided" in str(exc)

    def test_model_not_loaded_error(self) -> None:
        exc = ModelNotLoadedError()
        assert "ML model is not loaded" in str(exc)

    def test_analysis_error_default(self) -> None:
        exc = AnalysisError()
        assert exc.detail == "Analysis failed"

    def test_analysis_error_custom_detail(self) -> None:
        exc = AnalysisError(detail="Custom error")
        assert exc.detail == "Custom error"


class TestExceptionHandlers:
    @pytest.mark.asyncio
    async def test_insufficient_readings_handler(self) -> None:
        request = Request(scope={"type": "http", "method": "GET", "path": "/"})
        exc = InsufficientReadingsError(required=12, actual=5)
        response = await insufficient_readings_handler(request, exc)
        assert response.status_code == 400
        body = response.body.decode()
        assert "12" in body
        assert "5" in body

    @pytest.mark.asyncio
    async def test_insufficient_readings_handler_wrong_type(self) -> None:
        request = Request(scope={"type": "http", "method": "GET", "path": "/"})
        with pytest.raises(TypeError, match="Expected InsufficientReadingsError"):
            await insufficient_readings_handler(request, ValueError("bad"))

    @pytest.mark.asyncio
    async def test_model_not_loaded_handler(self) -> None:
        request = Request(scope={"type": "http", "method": "GET", "path": "/"})
        exc = ModelNotLoadedError()
        response = await model_not_loaded_handler(request, exc)
        assert response.status_code == 503
        assert "not available" in response.body.decode()

    @pytest.mark.asyncio
    async def test_analysis_error_handler(self) -> None:
        request = Request(scope={"type": "http", "method": "GET", "path": "/"})
        exc = AnalysisError()
        response = await analysis_error_handler(request, exc)
        assert response.status_code == 500
        assert "Internal analysis error" in response.body.decode()


class TestExceptionHandlersRegistered:
    def setup_method(self) -> None:
        self.app = FastAPI()
        for exc_type, handler in EXCEPTION_HANDLERS.items():
            self.app.add_exception_handler(exc_type, handler)

        @self.app.get("/trigger-insufficient")
        async def trigger_insufficient() -> None:
            raise InsufficientReadingsError(required=12, actual=3)

        @self.app.get("/trigger-model-not-loaded")
        async def trigger_model_not_loaded() -> None:
            raise ModelNotLoadedError()

        @self.app.get("/trigger-analysis-error")
        async def trigger_analysis_error() -> None:
            raise AnalysisError()

        self.client = TestClient(self.app)

    def test_insufficient_readings_returns_400(self) -> None:
        response = self.client.get("/trigger-insufficient")
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_model_not_loaded_returns_503(self) -> None:
        response = self.client.get("/trigger-model-not-loaded")
        assert response.status_code == 503
        assert response.json()["detail"] == "ML model is not available"

    def test_analysis_error_returns_500(self) -> None:
        response = self.client.get("/trigger-analysis-error")
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal analysis error"
