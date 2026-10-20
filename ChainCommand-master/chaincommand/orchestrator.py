"""ChainCommand Orchestrator — system coordinator and runtime state."""

from __future__ import annotations

import asyncio
import random
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import settings
from .data.schemas import (
    HumanApprovalRequest,
    OrderStatus,
    Product,
    PurchaseOrder,
    Supplier,
)
from .utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)


# ── Runtime state (singleton) ───────────────────────────────

@dataclass
class _RuntimeState:
    """Mutable global state shared across modules and API."""

    products: Optional[List[Product]] = None
    suppliers: Optional[List[Supplier]] = None
    demand_df: Optional[pd.DataFrame] = None

    # ML models  (typed as Optional for clarity; actual types are module-local)
    forecaster: Optional[Any] = None
    anomaly_detector: Optional[Any] = None
    optimizer: Optional[Any] = None

    # Engines
    kpi_engine: Optional[Any] = None
    event_bus: Optional[Any] = None
    monitor: Optional[Any] = None

    # New modules
    bom_manager: Optional[Any] = None
    rl_policy: Optional[Any] = None
    risk_scorer: Optional[Any] = None
    ctb_analyzer: Optional[Any] = None

    # Transaction state
    purchase_orders: List[PurchaseOrder] = field(default_factory=list)
    pending_approvals: Dict[str, HumanApprovalRequest] = field(default_factory=dict)

    # Results cache
    last_cycle_results: Dict[str, Any] = field(default_factory=dict)

    # Mutable runtime config — values that can be changed at runtime
    # without mutating the frozen Pydantic Settings object.
    runtime_config: Dict[str, Any] = field(default_factory=lambda: {
        "simulation_speed": settings.simulation_speed,
    })

    # Persistence backend
    backend: Any = None


_runtime = _RuntimeState()
_runtime_lock: asyncio.Lock | None = None


def _get_runtime_lock() -> asyncio.Lock:
    """Lazy-init the runtime lock inside an event loop (Python 3.12+ safe)."""
    global _runtime_lock
    if _runtime_lock is None:
        _runtime_lock = asyncio.Lock()
    return _runtime_lock


def _reset_runtime_state() -> None:
    """Reset shared runtime state to a clean baseline."""
    _runtime.products = None
    _runtime.suppliers = None
    _runtime.demand_df = None
    _runtime.forecaster = None
    _runtime.anomaly_detector = None
    _runtime.optimizer = None
    _runtime.kpi_engine = None
    _runtime.event_bus = None
    _runtime.monitor = None
    _runtime.bom_manager = None
    _runtime.rl_policy = None
    _runtime.risk_scorer = None
    _runtime.ctb_analyzer = None
    _runtime.purchase_orders.clear()
    _runtime.pending_approvals.clear()
    _runtime.last_cycle_results.clear()
    _runtime.runtime_config = {"simulation_speed": settings.simulation_speed}
    _runtime.backend = None


# ── Orchestrator ────────────────────────────────────────────

class ChainCommandOrchestrator:
    """Main orchestrator: initializes modules, runs optimization cycles."""

    STAGES = ["data", "ml", "engines", "bom", "rl", "risk", "kpi"]

    def __init__(self, on_progress: Any = None) -> None:
        self._running = False
        self._initialized = False
        self._cycle_count = 0
        self._loop_task: Optional[asyncio.Task] = None
        self._loop_lock = asyncio.Lock()
        self._on_progress = on_progress or (lambda *a, **kw: None)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    async def initialize(self) -> None:
        """Bootstrap the entire system."""
        async with _get_runtime_lock():
            await self._initialize_locked()

    async def _initialize_locked(self) -> None:
        """Bootstrap the entire system (caller must hold _runtime_lock)."""
        if self._initialized:
            log.info("initialize_skipped_already_initialized")
            return

        setup_logging()
        self._rng = random.Random(settings.random_seed)
        self._running = False
        self._cycle_count = 0
        _reset_runtime_state()
        log.info("initializing")

        # Phase 0: Generate synthetic data
        self._on_progress("data", "running", {})
        from .data.generator import generate_all

        products, suppliers, demand_df = generate_all(rng=self._rng)
        _runtime.products = products
        _runtime.suppliers = suppliers
        _runtime.demand_df = demand_df
        log.info("data_generated", products=len(products), suppliers=len(suppliers))
        self._on_progress("data", "completed", {"products": len(products), "suppliers": len(suppliers)})

        # Phase 1: Train ML models
        self._on_progress("ml", "running", {})
        from .models.anomaly_detector import AnomalyDetector
        from .models.forecaster import EnsembleForecaster
        from .models.optimizer import HybridOptimizer

        _runtime.forecaster = EnsembleForecaster()
        product_ids = [p.product_id for p in products[:settings.max_train_products]]
        _runtime.forecaster.train_all(demand_df, product_ids)
        _runtime.anomaly_detector = AnomalyDetector()
        _runtime.anomaly_detector.train(demand_df)
        _runtime.optimizer = HybridOptimizer()
        log.info("ml_models_trained")
        self._on_progress("ml", "completed", {})

        # Phase 2: Initialize engines
        self._on_progress("engines", "running", {})
        from .events.bus import EventBus
        from .events.monitor import ProactiveMonitor
        from .kpi.engine import KPIEngine

        _runtime.kpi_engine = KPIEngine()
        _runtime.event_bus = EventBus()
        await _runtime.event_bus.start()
        _runtime.monitor = ProactiveMonitor(
            _runtime.event_bus, _runtime.kpi_engine, _runtime.anomaly_detector
        )
        self._on_progress("engines", "completed", {})

        # Phase 3: BOM Management
        self._on_progress("bom", "running", {})
        from .bom import BOMManager

        _runtime.bom_manager = BOMManager()
        _runtime.bom_manager.generate_synthetic_boms(
            n_assemblies=settings.bom_default_assemblies,
            seed=settings.random_seed,
        )
        bom_summary = _runtime.bom_manager.get_summary()
        log.info("bom_initialized", **bom_summary)
        self._on_progress("bom", "completed", bom_summary)

        # Phase 4: RL Inventory Policy
        self._on_progress("rl", "running", {})
        from .rl import RLInventoryPolicy
        from .rl.environment import InventoryEnvConfig

        avg_demand = float(demand_df["quantity"].mean()) if "quantity" in demand_df.columns else 100.0
        rl_config = InventoryEnvConfig(
            demand_mean=avg_demand,
            demand_std=avg_demand * 0.3,
            episode_length=settings.rl_episode_length,
            holding_cost_per_unit=settings.rl_holding_cost,
            stockout_cost_per_unit=settings.rl_stockout_cost,
            ordering_cost_fixed=settings.rl_ordering_cost_fixed,
        )
        _runtime.rl_policy = RLInventoryPolicy(rl_config)
        rl_result = _runtime.rl_policy.train(
            total_timesteps=settings.rl_total_timesteps,
            seed=settings.random_seed,
        )
        log.info(
            "rl_trained",
            method=rl_result.method,
            improvement_pct=rl_result.improvement_pct,
        )
        self._on_progress("rl", "completed", {
            "method": rl_result.method,
            "mean_reward": rl_result.mean_reward,
            "improvement_pct": rl_result.improvement_pct,
        })

        # Phase 5: Risk Scoring
        self._on_progress("risk", "running", {})
        from .risk import SupplierRiskScorer

        _runtime.risk_scorer = SupplierRiskScorer()
        # Train ML risk model on synthetic history
        history = _runtime.risk_scorer.generate_synthetic_history(n_suppliers=100, seed=settings.random_seed)
        _runtime.risk_scorer.train_ml_model(history, seed=settings.random_seed)
        log.info("risk_scorer_initialized")
        self._on_progress("risk", "completed", {})

        # Phase 6: Initial KPI snapshot
        self._on_progress("kpi", "running", {})
        _runtime.kpi_engine.calculate_snapshot(
            products, _runtime.purchase_orders, suppliers,
            forecaster=_runtime.forecaster,
        )
        log.info("initial_kpi_computed")
        self._on_progress("kpi", "completed", {})

        # Phase 7: AWS backend
        from .aws import get_backend

        _runtime.backend = get_backend()
        await _runtime.backend.setup()
        if _runtime.demand_df is not None:
            await _runtime.backend.persist_demand_history(_runtime.demand_df)

        self._initialized = True
        log.info("system_ready")

    async def run_cycle(self) -> Dict[str, Any]:
        """Execute one optimization cycle."""
        async with _get_runtime_lock():
            return await self._run_cycle_locked()

    async def _run_cycle_locked(self) -> Dict[str, Any]:
        """Execute one optimization cycle (caller must hold _runtime_lock)."""
        self._cycle_count += 1
        log.info("cycle_start", cycle=self._cycle_count)

        products = _runtime.products or []
        suppliers = _runtime.suppliers or []
        results: Dict[str, Any] = {"cycle": self._cycle_count}

        # Step 1: Anomaly detection
        if _runtime.anomaly_detector and _runtime.demand_df is not None:
            anomalies = _runtime.anomaly_detector.detect_batch(products[:10])
            results["anomalies"] = len(anomalies) if anomalies else 0

        # Step 2: Risk scoring for suppliers
        if _runtime.risk_scorer and suppliers:
            from .risk.scorer import SupplierMetrics

            risk_scores = []
            for s in suppliers[:10]:
                metrics = SupplierMetrics(
                    supplier_id=s.supplier_id,
                    on_time_rate=s.on_time_rate,
                    defect_rate=s.defect_rate,
                    lead_time_mean=s.lead_time_mean,
                    lead_time_std=s.lead_time_std,
                )
                score = _runtime.risk_scorer.score_supplier(metrics)
                risk_scores.append(score)

            high_risk = sum(1 for r in risk_scores if r.risk_level in ("high", "critical"))
            results["risk"] = {
                "scored": len(risk_scores),
                "high_risk_count": high_risk,
            }

        # Step 3: CP-SAT supplier allocation for low-stock products
        low_stock = [p for p in products if p.current_stock < p.reorder_point]
        if low_stock:
            from .optimization.cpsat_optimizer import SupplierAllocationOptimizer, SupplierCandidate

            allocator = SupplierAllocationOptimizer()
            for product in low_stock[:5]:
                candidates = []
                for s in suppliers:
                    if product.product_id in s.products:
                        candidates.append(SupplierCandidate(
                            supplier_id=s.supplier_id,
                            unit_cost=product.unit_cost * s.cost_multiplier,
                            risk_score=1.0 - s.reliability_score,
                            capacity=s.capacity,
                            min_order_qty=float(product.min_order_qty),
                            lead_time_days=s.lead_time_mean,
                        ))
                if candidates:
                    alloc = allocator.optimize(candidates, product.daily_demand_avg * 30)
                    results.setdefault("allocations", []).append({
                        "product_id": product.product_id,
                        "status": alloc.solver_status,
                        "total_cost": alloc.total_cost,
                    })

        # Step 4: RL inventory decisions
        if _runtime.rl_policy:
            rl_decisions = []
            for p in products[:10]:
                decision = _runtime.rl_policy.decide(
                    current_stock=p.current_stock,
                    avg_demand=p.daily_demand_avg,
                )
                rl_decisions.append({
                    "product_id": p.product_id,
                    "action": decision.action,
                    "order_qty": decision.order_quantity,
                    "method": decision.method,
                })
            results["rl_decisions"] = len(rl_decisions)

        # Step 5: CTB analysis
        if _runtime.bom_manager and _runtime.ctb_analyzer is None:
            from .ctb import CTBAnalyzer
            _runtime.ctb_analyzer = CTBAnalyzer()

        if _runtime.ctb_analyzer and _runtime.bom_manager:
            ctb_reports = []
            for assembly_id, tree in list(_runtime.bom_manager.assemblies.items())[:3]:
                # Build inventory from current product stock
                inventory = {p.product_id: p.current_stock for p in products}
                roots = tree.root_items
                for root in roots:
                    report = _runtime.ctb_analyzer.analyze(
                        tree, root.part_id, settings.ctb_default_build_qty, inventory,
                    )
                    ctb_reports.append({
                        "assembly_id": assembly_id,
                        "is_clear": report.is_clear,
                        "clear_pct": report.clear_percentage,
                        "shortages": len(report.shortages),
                    })
            results["ctb"] = ctb_reports

        # Step 6: KPI update
        snapshot = _runtime.kpi_engine.calculate_snapshot(
            products, _runtime.purchase_orders, suppliers,
            forecaster=_runtime.forecaster,
        )
        violations = _runtime.kpi_engine.check_thresholds(snapshot)
        for event in violations:
            if _runtime.event_bus:
                await _runtime.event_bus.publish(event)

        # Persist cycle data
        if _runtime.backend:
            await _runtime.backend.persist_cycle(
                cycle=self._cycle_count,
                kpi=snapshot,
                events=list(_runtime.event_bus.recent_events[-50:]) if _runtime.event_bus else [],
                pos=_runtime.purchase_orders,
                products=products,
                suppliers=suppliers,
            )

        # Fulfill purchase orders whose lead time has elapsed
        now = datetime.now(UTC)
        product_map = {p.product_id: p for p in products}
        fulfilled_count = 0
        for po in _runtime.purchase_orders:
            if po.status in (OrderStatus.PENDING, OrderStatus.APPROVED, OrderStatus.SHIPPED):
                # Determine if enough time has passed since PO creation
                created = po.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                # Use expected_delivery if set, otherwise estimate from lead_time
                if po.expected_delivery:
                    delivery = po.expected_delivery
                    if delivery.tzinfo is None:
                        delivery = delivery.replace(tzinfo=UTC)
                else:
                    # Estimate delivery based on supplier lead time
                    supplier = next(
                        (s for s in suppliers if s.supplier_id == po.supplier_id), None
                    )
                    lead_days = supplier.lead_time_mean if supplier else 7.0
                    delivery = created + timedelta(days=lead_days)

                if now >= delivery:
                    product = product_map.get(po.product_id)
                    if product:
                        product.current_stock += po.quantity
                        fulfilled_count += 1
                    po.status = OrderStatus.DELIVERED

        if fulfilled_count:
            log.info("po_fulfilled", count=fulfilled_count)

        # Simulate demand consumption
        for p in products:
            consumed = max(0, self._rng.gauss(p.daily_demand_avg, p.daily_demand_std))
            p.current_stock = max(0, p.current_stock - consumed)

        results["kpi"] = snapshot.model_dump()
        results["violations"] = len(violations)
        _runtime.last_cycle_results = results

        log.info("cycle_complete", cycle=self._cycle_count, violations=len(violations))
        return results

    async def start_loop(self) -> bool:
        """Start the simulation loop once."""
        async with self._loop_lock:
            if not self._initialized:
                return False
            if self._loop_task and not self._loop_task.done():
                return False
            if self._running:
                return False
            self._running = True
            self._loop_task = asyncio.create_task(self.run_loop())
            return True

    async def run_loop(self) -> None:
        """Run continuous optimization cycles."""
        current_task = asyncio.current_task()
        if current_task and self._loop_task is None:
            self._loop_task = current_task
        self._running = True
        if _runtime.monitor:
            await _runtime.monitor.start()

        log.info("simulation_loop_started")
        try:
            while self._running:
                try:
                    await self.run_cycle()
                    speed = _runtime.runtime_config.get("simulation_speed", settings.simulation_speed)
                    speed = max(settings.simulation_speed_min, min(speed, settings.simulation_speed_max))
                    interval = settings.event_tick_seconds * 2 / speed
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.error("cycle_error", error=str(exc), exc_type=type(exc).__name__)
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            log.info("simulation_loop_cancelled")
            raise
        finally:
            self._running = False
            if _runtime.monitor:
                await _runtime.monitor.stop()
            if current_task is None or self._loop_task is current_task:
                self._loop_task = None

    async def stop_loop(self) -> None:
        """Stop the simulation loop."""
        async with self._loop_lock:
            self._running = False
            task = self._loop_task
            if task and not task.done():
                task.cancel()
        if task and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass
        log.info("simulation_loop_stopped")

    async def run_demo(self) -> Dict[str, Any]:
        """Run a single demo cycle and return results."""
        await self.initialize()
        result = await self.run_cycle()
        await self.shutdown()
        return result

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        async with _get_runtime_lock():
            await self._shutdown_locked()

    async def _shutdown_locked(self) -> None:
        """Clean shutdown of all components (caller must hold _runtime_lock)."""
        if not self._initialized and not _runtime.backend and not _runtime.monitor and not _runtime.event_bus:
            log.info("shutdown_skipped_not_initialized")
            return

        self._running = False
        if _runtime.backend:
            await _runtime.backend.teardown()
        if _runtime.monitor:
            await _runtime.monitor.stop()
        if _runtime.event_bus:
            await _runtime.event_bus.stop()
        self._loop_task = None
        self._initialized = False
        _reset_runtime_state()
        log.info("system_shutdown")


# ── Singleton ───────────────────────────────────────────────

_orchestrator: Optional[ChainCommandOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> ChainCommandOrchestrator:
    """Get or create the orchestrator singleton (thread-safe)."""
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = ChainCommandOrchestrator()
    return _orchestrator
