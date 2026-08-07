import numpy as np

from app.services.anomaly_detection_service import AnomalyDetectionService


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
