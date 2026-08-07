from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.ml.model_loader import get_model_loader

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, object]:
    return {"status": "healthy"}


@router.get("/health/ready", response_model=None)
async def readiness_check() -> dict[str, object] | JSONResponse:
    loader = get_model_loader()
    if not loader.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "Model not loaded"},
        )
    return {"status": "ready", "modelVersion": loader.version}
