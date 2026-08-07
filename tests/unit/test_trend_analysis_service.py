import numpy as np

from app.services.trend_analysis_service import TrendAnalysisService


class TestTrendAnalysisService:
    def setup_method(self) -> None:
        self.service = TrendAnalysisService()

    def test_ascending_trend(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.service.analyze(values)
        assert result["direction"] == "ASCENDING"
        assert result["slope"] > 0

    def test_descending_trend(self) -> None:
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = self.service.analyze(values)
        assert result["direction"] == "DESCENDING"
        assert result["slope"] < 0

    def test_stable_trend(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = self.service.analyze(values)
        assert result["direction"] == "STABLE"

    def test_single_value(self) -> None:
        result = self.service.analyze(np.array([5.0]))
        assert result["direction"] == "STABLE"
