import asyncio
import logging

import numpy as np

from app.ml.feature_engineering import build_feature_vector
from app.services.anomaly_detection_service import AnomalyDetectionService
from app.services.interfaces.anomaly_detector import AnomalyDetector
from app.services.interfaces.pattern_detector import PatternDetector
from app.services.interfaces.risk_predictor import RiskPredictor
from app.services.interfaces.trend_analyzer import TrendAnalyzer
from app.services.pattern_detection_service import PatternDetectionService
from app.services.risk_prediction_service import RiskPredictionService
from app.services.trend_analysis_service import TrendAnalysisService

logger = logging.getLogger(__name__)


class ClinicalAnalysisOrchestrator:
    def __init__(
        self,
        risk_predictor: RiskPredictor | None = None,
        trend_analyzer: TrendAnalyzer | None = None,
        pattern_detector: PatternDetector | None = None,
        anomaly_detector: AnomalyDetector | None = None,
    ) -> None:
        self._risk_predictor = risk_predictor or RiskPredictionService()
        self._trend_analyzer = trend_analyzer or TrendAnalysisService()
        self._pattern_detector = pattern_detector or PatternDetectionService()
        self._anomaly_detector = anomaly_detector or AnomalyDetectionService()

    async def analyze(
        self,
        heart_rates: np.ndarray,
        oxygens: np.ndarray,
        activities: np.ndarray,
        patient_id: str,
        window_size: int,
    ) -> dict[str, object]:
        features = build_feature_vector(heart_rates, oxygens, activities, window_size)

        readings_matrix = np.column_stack([heart_rates, oxygens, activities])

        (
            risk_result,
            hr_trend,
            o2_trend,
            act_trend,
            pattern_result,
            anomaly_result,
        ) = await asyncio.gather(
            asyncio.to_thread(self._risk_predictor.predict, features, patient_id),
            asyncio.to_thread(self._trend_analyzer.analyze, heart_rates),
            asyncio.to_thread(self._trend_analyzer.analyze, oxygens),
            asyncio.to_thread(self._trend_analyzer.analyze, activities),
            asyncio.to_thread(self._pattern_detector.detect, heart_rates),
            asyncio.to_thread(self._anomaly_detector.detect, readings_matrix),
            return_exceptions=True,
        )

        results: dict[str, object] = {}

        for name, result in [
            ("risk", risk_result),
            ("hr_trend", hr_trend),
            ("o2_trend", o2_trend),
            ("act_trend", act_trend),
            ("pattern", pattern_result),
            ("anomaly", anomaly_result),
        ]:
            if isinstance(result, Exception):
                logger.error("Capability %s failed: %s", name, result)
            else:
                results[name] = result

        return {
            "riskPrediction": results.get(
                "risk",
                {
                    "riskScore": 0.0,
                    "riskLevel": "LOW",
                    "recommendation": "Analysis unavailable",
                },
            ),
            "trendAnalysis": {
                "heartRate": results.get(
                    "hr_trend",
                    {
                        "direction": "STABLE",
                        "slope": 0.0,
                        "confidence": 0.0,
                    },
                ),
                "oxygen": results.get(
                    "o2_trend",
                    {
                        "direction": "STABLE",
                        "slope": 0.0,
                        "confidence": 0.0,
                    },
                ),
                "activity": results.get(
                    "act_trend",
                    {
                        "direction": "STABLE",
                        "slope": 0.0,
                        "confidence": 0.0,
                    },
                ),
            },
            "patternDetection": results.get(
                "pattern",
                {
                    "patternsFound": False,
                    "patterns": [],
                },
            ),
            "anomalyDetection": results.get(
                "anomaly",
                {
                    "anomalyDetected": False,
                    "anomalyScore": 0.0,
                    "affectedReadingsIndexes": [],
                },
            ),
        }
