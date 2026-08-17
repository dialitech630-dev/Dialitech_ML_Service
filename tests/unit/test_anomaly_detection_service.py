import numpy as np
import pytest

from app.services.anomaly_detection_service import AnomalyDetectionService

pytestmark = pytest.mark.unit


class TestAnomalyDetectionService:
    def setup_method(self) -> None:
        self.service = AnomalyDetectionService()

    def test_fit_and_detect(self) -> None:
        rng = np.random.default_rng(42)
        training_data = rng.uniform(0, 100, (100, 3))
        self.service.fit(training_data)

        normal = np.array([[70.0, 97.0, 40.0]])
        result = self.service.detect(normal)
        assert "anomalyDetected" in result
        assert "anomalyScore" in result
        assert "affectedReadingsIndexes" in result

    def test_unfitted_model_fits_on_first_call(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, (20, 3))
        result = self.service.detect(data)
        assert isinstance(result["anomalyDetected"], bool)

    def test_anomaly_score_in_valid_range(self) -> None:
        rng = np.random.default_rng(42)
        training_data = rng.uniform(0, 100, (100, 3))
        self.service.fit(training_data)

        normal = np.array([[70.0, 97.0, 40.0]])
        result = self.service.detect(normal)
        assert 0.0 <= result["anomalyScore"] <= 1.0

    def test_detect_with_clear_outlier(self) -> None:
        rng = np.random.default_rng(42)
        training_data = rng.uniform(50, 60, (100, 3))
        self.service.fit(training_data)

        outlier = np.array([[500.0, 500.0, 500.0]])
        result = self.service.detect(outlier)
        assert result["anomalyDetected"] is True
        assert len(result["affectedReadingsIndexes"]) > 0

    def test_detect_multiple_readings(self) -> None:
        rng = np.random.default_rng(42)
        training_data = rng.uniform(50, 60, (100, 3))
        self.service.fit(training_data)

        readings = np.array(
            [
                [55.0, 55.0, 55.0],
                [56.0, 56.0, 56.0],
                [500.0, 500.0, 500.0],
            ]
        )
        result = self.service.detect(readings)
        assert isinstance(result["affectedReadingsIndexes"], list)

    def test_affected_indexes_are_valid_indices(self) -> None:
        rng = np.random.default_rng(42)
        training_data = rng.uniform(50, 60, (100, 3))
        self.service.fit(training_data)

        readings = np.array(
            [
                [55.0, 55.0, 55.0],
                [500.0, 500.0, 500.0],
                [56.0, 56.0, 56.0],
            ]
        )
        result = self.service.detect(readings)
        for idx in result["affectedReadingsIndexes"]:
            assert 0 <= idx < len(readings)
