"""Lifecycle tests for orchestrator initialization and shutdown."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chaincommand.data.schemas import KPISnapshot


@pytest.fixture(autouse=True)
def _reset_runtime():
    """Ensure _runtime is clean before and after every test in this module."""
    from chaincommand.orchestrator import _reset_runtime_state

    _reset_runtime_state()
    yield
    _reset_runtime_state()


class TestOrchestratorInitializeShutdown:
    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, sample_products, sample_suppliers, sample_demand_df):
        from chaincommand.orchestrator import ChainCommandOrchestrator

        mock_forecaster = MagicMock()
        mock_anomaly = MagicMock()
        mock_kpi_engine = MagicMock()
        mock_kpi_engine.calculate_snapshot.return_value = KPISnapshot()
        mock_bom = MagicMock()
        mock_bom.get_summary.return_value = {"assemblies": 1}
        mock_rl_policy = MagicMock()
        mock_rl_policy.train.return_value = SimpleNamespace(
            method="mock",
            mean_reward=1.0,
            improvement_pct=5.0,
        )
        mock_risk = MagicMock()
        mock_risk.generate_synthetic_history.return_value = []
        mock_backend = MagicMock()
        mock_backend.setup = AsyncMock()
        mock_backend.persist_demand_history = AsyncMock()
        mock_backend.teardown = AsyncMock()
        mock_event_bus = MagicMock()
        mock_event_bus.start = AsyncMock()
        mock_event_bus.stop = AsyncMock()
        mock_monitor = MagicMock()
        mock_monitor.start = AsyncMock()
        mock_monitor.stop = AsyncMock()

        with patch(
            "chaincommand.data.generator.generate_all",
            return_value=(sample_products, sample_suppliers, sample_demand_df),
        ), \
             patch("chaincommand.models.forecaster.EnsembleForecaster", return_value=mock_forecaster), \
             patch("chaincommand.models.anomaly_detector.AnomalyDetector", return_value=mock_anomaly), \
             patch("chaincommand.models.optimizer.HybridOptimizer", return_value=MagicMock()), \
             patch("chaincommand.events.bus.EventBus", return_value=mock_event_bus), \
             patch("chaincommand.events.monitor.ProactiveMonitor", return_value=mock_monitor), \
             patch("chaincommand.kpi.engine.KPIEngine", return_value=mock_kpi_engine), \
             patch("chaincommand.bom.BOMManager", return_value=mock_bom), \
             patch("chaincommand.rl.RLInventoryPolicy", return_value=mock_rl_policy), \
             patch("chaincommand.risk.SupplierRiskScorer", return_value=mock_risk), \
             patch("chaincommand.aws.get_backend", return_value=mock_backend):
            orch = ChainCommandOrchestrator()
            await orch.initialize()
            await orch.initialize()

        assert orch._initialized is True
        mock_event_bus.start.assert_awaited_once()
        mock_backend.setup.assert_awaited_once()
        mock_backend.persist_demand_history.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_resets_runtime_state(self):
        import chaincommand.orchestrator as orchestrator_module
        from chaincommand.orchestrator import ChainCommandOrchestrator

        mock_backend = MagicMock()
        mock_backend.teardown = AsyncMock()
        mock_monitor = MagicMock()
        mock_monitor.stop = AsyncMock()
        mock_event_bus = MagicMock()
        mock_event_bus.stop = AsyncMock()

        orchestrator_module._runtime.products = [MagicMock()]
        orchestrator_module._runtime.purchase_orders.append(MagicMock())
        orchestrator_module._runtime.pending_approvals["x"] = MagicMock()
        orchestrator_module._runtime.last_cycle_results["k"] = "v"
        orchestrator_module._runtime.backend = mock_backend
        orchestrator_module._runtime.monitor = mock_monitor
        orchestrator_module._runtime.event_bus = mock_event_bus

        orch = ChainCommandOrchestrator()
        orch._initialized = True
        await orch.shutdown()

        assert orchestrator_module._runtime.products is None
        assert orchestrator_module._runtime.purchase_orders == []
        assert orchestrator_module._runtime.pending_approvals == {}
        assert orchestrator_module._runtime.last_cycle_results == {}
        assert orch._initialized is False
        mock_backend.teardown.assert_awaited_once()
        mock_monitor.stop.assert_awaited_once()
        mock_event_bus.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_loop_requires_initialization(self):
        from chaincommand.orchestrator import ChainCommandOrchestrator

        orch = ChainCommandOrchestrator()
        assert await orch.start_loop() is False
