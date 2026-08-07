import logging

import numpy as np
from fastapi import APIRouter, Depends

from app.api.v1.dependencies import get_orchestrator
from app.core.exceptions import InsufficientReadingsError
from app.core.security import verify_api_key
from app.ml.model_loader import get_model_loader
from app.schemas.analyze_request import AnalyzeRequest
from app.services.clinical_analysis_orchestrator import ClinicalAnalysisOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    _api_key: str = Depends(verify_api_key),
    orchestrator: ClinicalAnalysisOrchestrator = Depends(get_orchestrator),
) -> dict[str, object]:
    loader = get_model_loader()
    window_size = request.windowSize or 12

    if len(request.readings) < window_size:
        raise InsufficientReadingsError(
            required=window_size, actual=len(request.readings)
        )

    heart_rates = np.array([r.heartRate for r in request.readings])
    oxygens = np.array([r.oxygen for r in request.readings])
    activities = np.array([r.activity for r in request.readings])

    result = await orchestrator.analyze(
        heart_rates=heart_rates,
        oxygens=oxygens,
        activities=activities,
        patient_id=request.patientId,
        window_size=window_size,
    )

    return {
        "patientId": request.patientId,
        "modelVersion": loader.version,
        **result,
    }
