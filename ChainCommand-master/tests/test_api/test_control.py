"""Control endpoint tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSimulationControl:
    @pytest.mark.asyncio
    async def test_simulation_status(self, client, auth_headers, mock_runtime):
        mock_orch = MagicMock()
        mock_orch.running = False
        mock_orch.cycle_count = 0

        with patch("chaincommand.orchestrator._runtime", mock_runtime), \
             patch("chaincommand.orchestrator.get_orchestrator", return_value=mock_orch):
            resp = await client.get("/api/simulation/status", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["running"] is False

    @pytest.mark.asyncio
    async def test_simulation_start(self, client, auth_headers):
        mock_orch = MagicMock()
        mock_orch.start_loop = AsyncMock(return_value=True)

        with patch("chaincommand.orchestrator.get_orchestrator", return_value=mock_orch):
            resp = await client.post("/api/simulation/start", headers=auth_headers)
            assert resp.status_code == 200
            assert resp.json()["status"] == "started"
            mock_orch.start_loop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_simulation_start_already_running(self, client, auth_headers):
        mock_orch = MagicMock()
        mock_orch.start_loop = AsyncMock(return_value=False)

        with patch("chaincommand.orchestrator.get_orchestrator", return_value=mock_orch):
            resp = await client.post("/api/simulation/start", headers=auth_headers)
            assert resp.status_code == 200
            assert resp.json()["status"] == "already_running"

    @pytest.mark.asyncio
    async def test_simulation_speed_invalid(self, client, auth_headers):
        resp = await client.post(
            "/api/simulation/speed?speed=0.01",
            headers=auth_headers,
        )
        assert resp.status_code == 422
        assert "Speed must be between" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_simulation_speed_valid(self, client, auth_headers):
        resp = await client.post(
            "/api/simulation/speed?speed=2.0",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["speed"] == 2.0

    @pytest.mark.asyncio
    async def test_simulation_stop(self, client, auth_headers):
        mock_orch = MagicMock()
        mock_orch.stop_loop = AsyncMock(return_value=None)

        with patch("chaincommand.orchestrator.get_orchestrator", return_value=mock_orch):
            resp = await client.post("/api/simulation/stop", headers=auth_headers)
            assert resp.status_code == 200
            mock_orch.stop_loop.assert_awaited_once()


class TestOrchestratorLoopLifecycle:
    @pytest.mark.asyncio
    async def test_start_loop_only_creates_one_task(self):
        from chaincommand.orchestrator import ChainCommandOrchestrator

        orch = ChainCommandOrchestrator()
        orch._initialized = True

        async def fake_run_loop():
            await asyncio.sleep(0.05)

        orch.run_loop = fake_run_loop  # type: ignore[method-assign]

        first = await orch.start_loop()
        second = await orch.start_loop()

        assert first is True
        assert second is False

        await orch.stop_loop()

    @pytest.mark.asyncio
    async def test_stop_loop_cancels_running_task(self):
        from chaincommand.orchestrator import ChainCommandOrchestrator

        orch = ChainCommandOrchestrator()
        orch._initialized = True
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fake_run_loop():
            started.set()
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        orch.run_loop = fake_run_loop  # type: ignore[method-assign]

        assert await orch.start_loop() is True
        await asyncio.wait_for(started.wait(), timeout=1)
        await orch.stop_loop()
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        assert orch.running is False
