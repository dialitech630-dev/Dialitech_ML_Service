from fastapi import APIRouter, Depends

from app.core.security import verify_api_key
from app.ml.model_loader import get_model_loader

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model-info")
async def model_info(
    _api_key: str = Depends(verify_api_key),
) -> dict[str, object]:
    loader = get_model_loader()
    return {
        "version": loader.version,
        "isLoaded": loader.is_loaded,
        "metadata": loader.metadata,
    }
