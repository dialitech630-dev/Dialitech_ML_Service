import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestAnalyzeEndpoint:
    def test_valid_request(self, client: TestClient) -> None:
        rng = np.random.default_rng(42)
        readings = []
        for _ in range(15):
            readings.append({
                "heartRate": round(float(rng.uniform(60, 100)), 1),
                "oxygen": round(float(rng.uniform(94, 99)), 1),
                "activity": round(float(rng.uniform(20, 80)), 1),
                "timestamp": "2026-08-06T10:00:00Z",
            })

        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "integration-test",
                "windowSize": 12,
                "readings": readings,
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

    def test_missing_api_key(self, client: TestClient) -> None:
        rng = np.random.default_rng(42)
        readings = [
            {
                "heartRate": round(float(rng.uniform(60, 100)), 1),
                "oxygen": round(float(rng.uniform(94, 99)), 1),
                "activity": round(float(rng.uniform(20, 80)), 1),
                "timestamp": "2026-08-06T10:00:00Z",
            }
            for _ in range(15)
        ]

        response = client.post(
            "/api/v1/analyze",
            json={
                "patientId": "test",
                "windowSize": 12,
                "readings": readings,
            },
        )

        assert response.status_code == 422
