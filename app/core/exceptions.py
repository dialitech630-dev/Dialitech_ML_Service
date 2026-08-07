from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse


class InsufficientReadingsError(Exception):
    def __init__(self, required: int, actual: int) -> None:
        self.required = required
        self.actual = actual
        super().__init__(
            f"Insufficient readings: {required} required, {actual} provided"
        )


class ModelNotLoadedError(Exception):
    def __init__(self) -> None:
        super().__init__("ML model is not loaded")


class AnalysisError(Exception):
    def __init__(self, detail: str = "Analysis failed") -> None:
        self.detail = detail
        super().__init__(detail)


async def insufficient_readings_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    assert isinstance(exc, InsufficientReadingsError)
    return JSONResponse(
        status_code=400,
        content={
            "detail": f"Requires at least {exc.required} readings, got {exc.actual}"
        },
    )


async def model_not_loaded_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "ML model is not available"},
    )


async def analysis_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal analysis error"},
    )


ExceptionHandler = Callable[[Request, Exception], Awaitable[JSONResponse]]

EXCEPTION_HANDLERS: dict[type[Exception], ExceptionHandler] = {
    InsufficientReadingsError: insufficient_readings_handler,
    ModelNotLoadedError: model_not_loaded_handler,
    AnalysisError: analysis_error_handler,
}
