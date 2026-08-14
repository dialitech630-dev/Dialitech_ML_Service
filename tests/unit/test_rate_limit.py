import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.rate_limit import RateLimitMiddleware

pytestmark = pytest.mark.unit

app = FastAPI()
app.add_middleware(RateLimitMiddleware, max_requests=3, window_seconds=1)


@app.get("/test")
async def test_route() -> dict:
    return {"status": "ok"}


client = TestClient(app)


class TestRateLimitMiddleware:
    def test_allows_requests_under_limit(self) -> None:
        for _ in range(3):
            response = client.get("/test")
            assert response.status_code == 200

    def test_blocks_requests_over_limit(self) -> None:
        for _ in range(3):
            client.get("/test")

        response = client.get("/test")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_resets_after_window(self) -> None:
        for _ in range(3):
            client.get("/test")

        response = client.get("/test")
        assert response.status_code == 429

        time.sleep(1.1)

        response = client.get("/test")
        assert response.status_code == 200
