import numpy as np
import pytest

from app.services.clinical_analysis_orchestrator import ClinicalAnalysisOrchestrator

pytestmark = pytest.mark.unit


class TestClinicalAnalysisOrchestrator:
    def setup_method(self) -> None:
        self.orchestrator = ClinicalAnalysisOrchestrator()

    @pytest.mark.asyncio
    async def test_analyze_returns_all_sections(self) -> None:
        rng = np.random.default_rng(42)
        hr = rng.uniform(60, 100, 12)
        o2 = rng.uniform(94, 99, 12)
        act = rng.uniform(20, 80, 12)

        result = await self.orchestrator.analyze(
            heart_rates=hr,
            oxygens=o2,
            activities=act,
            patient_id="test-patient",
            window_size=12,
        )

        assert "riskPrediction" in result
        assert "trendAnalysis" in result
        assert "patternDetection" in result
        assert "anomalyDetection" in result
