import numpy as np
import pytest

from app.services.trend_analysis_service import TrendAnalysisService

pytestmark = pytest.mark.unit


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
        assert result["slope"] == 0.0
        assert result["confidence"] == 0.0

    def test_confidence_increases_with_slope(self) -> None:
        mild = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        steep = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result_mild = self.service.analyze(mild)
        result_steep = self.service.analyze(steep)
        assert result_steep["confidence"] >= result_mild["confidence"]

    def test_slope_rounded_to_three_decimals(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.service.analyze(values)
        slope_str = str(result["slope"])
        decimal_part = slope_str.split(".")[-1] if "." in slope_str else ""
        assert len(decimal_part) <= 3

    def test_confidence_range(self) -> None:
        values = np.array([1.0, 5.0, 2.0, 8.0, 1.0])
        result = self.service.analyze(values)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_nearly_stable_trend(self) -> None:
        values = np.array([5.0, 5.01, 5.02, 5.03, 5.04])
        result = self.service.analyze(values)
        assert result["direction"] == "STABLE"
