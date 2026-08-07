from unittest.mock import patch

import numpy as np

from app.services.risk_prediction_service import RiskPredictionService


class TestRiskPredictionService:
    def setup_method(self) -> None:
        self.service = RiskPredictionService()

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_predict_returns_expected_keys(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.75

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert "riskScore" in result
        assert "riskLevel" in result
        assert "recommendation" in result
        assert result["riskLevel"] in ("LOW", "MEDIUM", "HIGH")

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_predict_high_risk(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.85

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert result["riskLevel"] == "HIGH"

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_predict_low_risk(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.2

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert result["riskLevel"] == "LOW"

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_predict_model_not_loaded(self, mock_get_loader) -> None:
        from app.core.exceptions import ModelNotLoadedError

        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = False

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        try:
            self.service.predict(features, "test-patient")
            assert False, "Should have raised ModelNotLoadedError"
        except ModelNotLoadedError:
            pass
