import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security_headers import SecurityHeadersMiddleware

pytestmark = pytest.mark.unit

app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)


@app.get("/test")
async def test_route() -> dict:
    return {"status": "ok"}


client = TestClient(app)


class TestSecurityHeadersMiddleware:
    def test_x_content_type_options(self) -> None:
        response = client.get("/test")
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_x_frame_options(self) -> None:
        response = client.get("/test")
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_content_security_policy(self) -> None:
        response = client.get("/test")
        assert "Content-Security-Policy" in response.headers
        assert "default-src 'none'" in response.headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]

    def test_referrer_policy(self) -> None:
        response = client.get("/test")
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_permissions_policy(self) -> None:
        response = client.get("/test")
        assert "Permissions-Policy" in response.headers
        assert "geolocation=()" in response.headers["Permissions-Policy"]

    def test_all_headers_present(self) -> None:
        response = client.get("/test")
        required = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy",
        ]
        for header in required:
            assert header in response.headers, f"Missing header: {header}"
