import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_readings() -> list[dict]:
    rng = np.random.default_rng(42)
    return [
        {
            "heartRate": round(float(rng.uniform(60, 100)), 1),
            "oxygen": round(float(rng.uniform(94, 99)), 1),
            "activity": round(float(rng.uniform(20, 80)), 1),
            "timestamp": "2026-08-06T10:00:00Z",
        }
        for _ in range(15)
    ]


class TestAnalyzeEndpointValidRequests:
    def test_valid_request(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "integration-test",
                "windowSize": 12,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "REPLACED_API_KEY"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["patientId"] == "integration-test"
        assert "riskPrediction" in data
        assert "trendAnalysis" in data
        assert "patternDetection" in data
        assert "anomalyDetection" in data

    def test_valid_request_returns_model_version(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 200
        assert "modelVersion" in response.json()

    def test_valid_request_risk_prediction_structure(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "test-key"},
        )
        data = response.json()
        risk = data["riskPrediction"]
        assert "riskScore" in risk
        assert "riskLevel" in risk
        assert "recommendation" in risk
        assert risk["riskLevel"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(risk["riskScore"], float)

    def test_valid_request_trend_analysis_structure(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "test-key"},
        )
        data = response.json()
        trend = data["trendAnalysis"]
        for metric in ("heartRate", "oxygen", "activity"):
            assert metric in trend
            assert "direction" in trend[metric]
            assert "slope" in trend[metric]
            assert "confidence" in trend[metric]
            assert trend[metric]["direction"] in ("ASCENDING", "DESCENDING", "STABLE")

    def test_valid_request_pattern_detection_structure(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "test-key"},
        )
        data = response.json()
        pattern = data["patternDetection"]
        assert "patternsFound" in pattern
        assert "patterns" in pattern
        assert isinstance(pattern["patternsFound"], bool)
        assert isinstance(pattern["patterns"], list)

    def test_valid_request_anomaly_detection_structure(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "test-key"},
        )
        data = response.json()
        anomaly = data["anomalyDetection"]
        assert "anomalyDetected" in anomaly
        assert "anomalyScore" in anomaly
        assert "affectedReadingsIndexes" in anomaly
        assert isinstance(anomaly["anomalyDetected"], bool)
        assert isinstance(anomaly["anomalyScore"], float)
        assert isinstance(anomaly["affectedReadingsIndexes"], list)


class TestAnalyzeEndpointErrors:
    def test_insufficient_readings(self, client: TestClient) -> None:
        readings = [
            {
                "heartRate": 70.0,
                "oxygen": 97.0,
                "activity": 40.0,
                "timestamp": "2026-08-06T10:00:00Z",
            }
        ]

        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": readings,
            },
            headers={"X-API-Key": "REPLACED_API_KEY"},
        )

        assert response.status_code == 400
        assert "detail" in response.json()

    def test_invalid_api_key(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

    def test_missing_api_key(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": valid_readings,
            },
        )
        assert response.status_code == 422

    def test_missing_patient_id(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "windowSize": 12,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 422

    def test_missing_readings(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
            },
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 422

    def test_empty_readings(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": [],
            },
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 422

    def test_malformed_json(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyze",
            content="not json",
            headers={"X-API-Key": "test-key", "Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_window_size_zero(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 0,
                "readings": valid_readings,
            },
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 422

    def test_window_size_default_12(
        self, client: TestClient, valid_readings: list[dict]
    ) -> None:
        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "readings": valid_readings,
            },
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 200


class TestHealthEndpointExpanded:
    def test_liveness(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_returns_correct_structure(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        if response.status_code == 200:
            assert "modelVersion" in data


class TestModelInfoEndpoint:
    def test_model_info_requires_api_key(self, client: TestClient) -> None:
        response = client.get("/api/v1/model-info")
        assert response.status_code == 422

    def test_model_info_invalid_key(self, client: TestClient) -> None:
        response = client.get("/api/v1/model-info", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_model_info_valid_key(self, client: TestClient) -> None:
        response = client.get("/api/v1/model-info", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "isLoaded" in data
        assert "metadata" in data
        assert isinstance(data["isLoaded"], bool)


class TestCORSHeaders:
    def test_cors_preflight_analyze(self, client: TestClient) -> None:
        response = client.options(
            "/api/v1/analyze",
            headers={
                "Origin": "https://dialitech.netlify.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-API-Key",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert (
            response.headers["access-control-allow-origin"]
            == "https://dialitech.netlify.app"
        )

    def test_cors_preflight_health(self, client: TestClient) -> None:
        response = client.options(
            "/health",
            headers={
                "Origin": "https://dialitech.netlify.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_rejects_unknown_origin(self, client: TestClient) -> None:
        response = client.options(
            "/api/v1/analyze",
            headers={
                "Origin": "https://evil-site.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_origin = response.headers.get("access-control-allow-origin")
        assert allow_origin is None or allow_origin != "https://evil-site.com"
