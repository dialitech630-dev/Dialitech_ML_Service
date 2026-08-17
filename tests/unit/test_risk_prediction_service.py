from unittest.mock import patch

import numpy as np
import pytest

from app.core.exceptions import ModelNotLoadedError
from app.services.risk_prediction_service import RiskPredictionService

pytestmark = pytest.mark.unit


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
    def test_predict_medium_risk(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.55

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert result["riskLevel"] == "MEDIUM"

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_predict_low_risk(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.2

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert result["riskLevel"] == "LOW"

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_predict_boundary_high_threshold(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.7

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert result["riskLevel"] == "HIGH"

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_predict_boundary_medium_threshold(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.4

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert result["riskLevel"] == "MEDIUM"

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_predict_model_not_loaded(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = False

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        with pytest.raises(ModelNotLoadedError):
            self.service.predict(features, "test-patient")

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_risk_score_rounded_to_three_decimals(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.123456789

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert result["riskScore"] == 0.123

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_recommendation_matches_risk_level_high(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.85

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert "immediate" in result["recommendation"].lower()

    @patch("app.services.risk_prediction_service.get_model_loader")
    def test_recommendation_matches_risk_level_low(self, mock_get_loader) -> None:
        mock_loader = mock_get_loader.return_value
        mock_loader.is_loaded = True
        mock_loader.predict_proba.return_value = 0.1

        features = np.array([75.0, 5.0, 0.5, 96.0, 1.0, -0.1, 40.0, 10.0, -1.0])
        result = self.service.predict(features, "test-patient")
        assert "acceptable" in result["recommendation"].lower()
