from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.security import verify_api_key

pytestmark = pytest.mark.security

app = FastAPI()


@app.get("/test-protected")
async def protected_route(_api_key: str = Depends(verify_api_key)) -> dict:
    return {"status": "ok", "key": _api_key}


client = TestClient(app)


class TestVerifyApiKey:
    def test_valid_api_key(self) -> None:
        response = client.get("/test-protected", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["key"] == "test-key"

    def test_invalid_api_key(self) -> None:
        response = client.get("/test-protected", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API key"

    def test_missing_api_key_header(self) -> None:
        response = client.get("/test-protected")
        assert response.status_code == 422

    def test_empty_api_key(self) -> None:
        response = client.get("/test-protected", headers={"X-API-Key": ""})
        assert response.status_code == 401

    def test_api_key_case_sensitive(self) -> None:
        response = client.get("/test-protected", headers={"X-API-Key": "TEST-KEY"})
        assert response.status_code == 401


class TestTimingSafeComparison:
    def test_uses_hmac_compare_digest(self) -> None:
        with patch("app.core.security.hmac.compare_digest") as mock_cmp:
            mock_cmp.return_value = True
            response = client.get("/test-protected", headers={"X-API-Key": "test-key"})
            assert response.status_code == 200
            mock_cmp.assert_called_once_with("test-key", "test-key")

    def test_rejects_with_hmac_compare_digest(self) -> None:
        with patch("app.core.security.hmac.compare_digest") as mock_cmp:
            mock_cmp.return_value = False
            response = client.get("/test-protected", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 401

    def test_hmac_is_imported(self) -> None:
        import app.core.security as sec

        assert hasattr(sec, "hmac")
