import numpy as np

from app.services.pattern_detection_service import PatternDetectionService


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
