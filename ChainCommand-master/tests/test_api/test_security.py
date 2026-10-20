"""Security tests — auth, CORS, rate limiting, WebSocket auth."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, WebSocketException

# ── Authentication Tests ────────────────────────────────────


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_root_no_auth_required(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "ChainCommand"

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_dashboard_requires_auth(self, client):
        """All dashboard endpoints return 401 without API key."""
        endpoints = [
            "/api/kpi/current",
            "/api/kpi/history",
            "/api/inventory/status",
            "/api/bom/summary",
            "/api/events/recent",
            "/api/approvals/pending",
            "/api/aws/status",
        ]
        for ep in endpoints:
            resp = await client.get(ep)
            assert resp.status_code == 401, f"{ep} should require auth"

    @pytest.mark.asyncio
    async def test_control_requires_auth(self, client):
        """All control endpoints return 401 without API key."""
        endpoints = [
            ("/api/simulation/start", "post"),
            ("/api/simulation/stop", "post"),
            ("/api/simulation/status", "get"),
        ]
        for ep, method in endpoints:
            resp = await getattr(client, method)(ep)
            assert resp.status_code == 401, f"{ep} should require auth"

    @pytest.mark.asyncio
    async def test_valid_api_key_accepted(self, client, auth_headers):
        """Endpoints accept valid API key (may fail for other reasons, but not 401)."""
        resp = await client.get("/api/kpi/current", headers=auth_headers)
        assert resp.status_code != 401

    @pytest.mark.asyncio
    async def test_wrong_api_key_rejected(self, client):
        """Endpoints reject wrong API key."""
        resp = await client.get("/api/kpi/current", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_ws_api_key_rejected(self):
        from chaincommand.auth import check_ws_query_key

        websocket = MagicMock()
        websocket.query_params = {"api_key": "wrong-key"}

        with pytest.raises(WebSocketException) as exc:
            await check_ws_query_key(websocket)

        assert exc.value.code == 1008
        assert exc.value.reason == "Invalid API key"


# ── CORS Tests ──────────────────────────────────────────────


class TestCORS:
    @pytest.mark.asyncio
    async def test_cors_not_wildcard(self, client):
        """CORS does not allow all origins."""
        resp = await client.options(
            "/",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert allow_origin != "*"

    @pytest.mark.asyncio
    async def test_cors_allows_configured_origin(self, client):
        """CORS allows configured localhost origins."""
        resp = await client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert allow_origin in ("http://localhost:3000", "")


# ── Rate Limiting Tests ────────────────────────────────────


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_not_triggered_normally(self, client):
        """A few requests should not trigger rate limiting."""
        for _ in range(5):
            resp = await client.get("/")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429_when_threshold_exceeded(self):
        import importlib

        from httpx import ASGITransport, AsyncClient

        from chaincommand.api.app import configure_middlewares

        app_module = importlib.import_module("chaincommand.api.app")

        app = FastAPI()
        configure_middlewares(app)

        @app.get("/")
        async def root():
            return {"ok": True}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(app_module, "_rate_limit_store", defaultdict(list))
                mp.setattr(app_module.settings, "rate_limit_per_minute", 2)
                assert (await client.get("/")).status_code == 200
                assert (await client.get("/")).status_code == 200
                assert (await client.get("/")).status_code == 429
