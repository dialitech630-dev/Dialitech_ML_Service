import logging

import numpy as np

from app.core.config import settings
from app.ml.model_loader import get_model_loader

logger = logging.getLogger(__name__)


class RiskPredictionService:
    def predict(self, features: np.ndarray, patient_id: str) -> dict[str, object]:
        loader = get_model_loader()

        if not loader.is_loaded:
            from app.core.exceptions import ModelNotLoadedError

            raise ModelNotLoadedError()

        score = loader.predict_proba(features)

        if score >= settings.RISK_THRESHOLD_HIGH:
            level = "HIGH"
            recommendation = (
                "High risk detected. Consider immediate clinical review."
            )
        elif score >= settings.RISK_THRESHOLD_MEDIUM:
            level = "MEDIUM"
            recommendation = (
                "Moderate risk. Schedule follow-up within 24 hours."
            )
        else:
            level = "LOW"
            recommendation = "Risk within acceptable range."

        logger.info(
            "Risk prediction for patient %s: score=%.3f, level=%s",
            patient_id,
            score,
            level,
        )

        return {
            "riskScore": round(score, 3),
            "riskLevel": level,
            "recommendation": recommendation,
        }
