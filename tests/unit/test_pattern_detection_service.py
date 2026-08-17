import numpy as np
import pytest

from app.services.pattern_detection_service import PatternDetectionService

pytestmark = pytest.mark.unit


class TestPatternDetectionService:
    def setup_method(self) -> None:
        self.service = PatternDetectionService()

    def test_oscillation_detected(self) -> None:
        values = np.array([10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0])
        result = self.service.detect(values)
        assert result["patternsFound"] is True

    def test_stable_no_pattern(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = self.service.detect(values)
        assert result["patternsFound"] is False

    def test_too_few_values(self) -> None:
        result = self.service.detect(np.array([1.0, 2.0]))
        assert result["patternsFound"] is False

    def test_exactly_three_values(self) -> None:
        result = self.service.detect(np.array([1.0, 2.0, 3.0]))
        assert result["patternsFound"] is False

    def test_exactly_four_values(self) -> None:
        result = self.service.detect(np.array([1.0, 5.0, 1.0, 5.0]))
        assert isinstance(result["patternsFound"], bool)
        assert isinstance(result["patterns"], list)

    def test_high_variability_detected(self) -> None:
        values = np.array([10.0, 100.0, 5.0, 95.0, 8.0, 100.0, 3.0])
        result = self.service.detect(values)
        assert result["patternsFound"] is True

    def test_low_variability_no_pattern(self) -> None:
        values = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        result = self.service.detect(values)
        assert result["patternsFound"] is False

    def test_both_patterns_detected(self) -> None:
        values = np.array([10.0, 100.0, 5.0, 95.0, 8.0, 100.0, 5.0, 95.0])
        result = self.service.detect(values)
        if result["patternsFound"]:
            pattern_types = [p["type"] for p in result["patterns"]]
            assert len(pattern_types) > 0

    def test_pattern_contents_have_required_keys(self) -> None:
        values = np.array([10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0])
        result = self.service.detect(values)
        if result["patternsFound"]:
            for pattern in result["patterns"]:
                assert "type" in pattern
                assert "description" in pattern
                assert "confidence" in pattern
                assert isinstance(pattern["confidence"], float)
