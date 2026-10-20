"""Proactive monitoring engine — periodic health checks and alerting."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..config import settings
from ..data.schemas import AlertSeverity, SupplyChainEvent, ensure_utc
from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..events.bus import EventBus
    from ..kpi.engine import KPIEngine
    from ..models.anomaly_detector import AnomalyDetector

log = get_logger(__name__)

# Use the canonical ensure_utc from schemas
_ensure_utc = ensure_utc

_ALERT_COOLDOWN_SECONDS = 15 * 60  # 15 minutes


class ProactiveMonitor:
    """Scans system state every tick and emits alerts.

    Checks:
    1. Inventory water levels → low stock alerts
    2. KPI deviations → threshold violation events
    3. Pending POs → delivery delay alerts
    4. Anomaly detection → anomaly events

    Alert deduplication: each (product_id/po_id, alert_type) pair is subject
    to a cooldown period so the same alert is not emitted every tick.
    """

    def __init__(
        self,
        event_bus: EventBus,
        kpi_engine: KPIEngine,
        anomaly_detector: AnomalyDetector | None = None,
        alert_cooldown_seconds: float = _ALERT_COOLDOWN_SECONDS,
    ) -> None:
        self._bus = event_bus
        self._kpi = kpi_engine
        self._anomaly = anomaly_detector
        self._running = False
        self._task: asyncio.Task | None = None
        self._tick_count = 0
        self._alert_cooldown_seconds = alert_cooldown_seconds
        # (entity_id, alert_type) -> last fire timestamp
        self._recent_alerts: dict[tuple[str, str], datetime] = {}

    def _should_fire(self, entity_id: str, alert_type: str) -> bool:
        """Return True if this alert hasn't been fired within the cooldown window."""
        now = datetime.now(UTC)
        key = (entity_id, alert_type)
        last = self._recent_alerts.get(key)
        if last is not None:
            elapsed = (now - _ensure_utc(last)).total_seconds()
            if elapsed < self._alert_cooldown_seconds:
                return False
        self._recent_alerts[key] = now
        # Periodically clean up entries older than the cooldown to prevent unbounded growth
        if len(self._recent_alerts) > 500:
            cutoff = now
            stale_keys = [
                k for k, v in self._recent_alerts.items()
                if (cutoff - _ensure_utc(v)).total_seconds() > self._alert_cooldown_seconds
            ]
            for k in stale_keys:
                del self._recent_alerts[k]
        return True

    async def start(self) -> None:
        if not settings.enable_proactive_monitoring:
            log.info("proactive_monitoring_disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("proactive_monitor_started", tick_seconds=settings.event_tick_seconds)

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.tick()
                # TODO: _runtime import creates coupling between monitor and
                # orchestrator modules.  Ideally the runtime config (e.g.
                # simulation_speed) would be injected via constructor or a
                # shared config object rather than importing the module-level
                # singleton directly.  Tracked for future refactor.
                from ..orchestrator import _runtime
                speed = _runtime.runtime_config.get("simulation_speed", settings.simulation_speed)
                speed = max(settings.simulation_speed_min, min(speed, settings.simulation_speed_max))
                interval = settings.event_tick_seconds / speed
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("monitor_tick_error", error=str(exc))

    async def tick(self) -> None:
        """One monitoring cycle."""
        from ..orchestrator import _runtime

        self._tick_count += 1
        products = _runtime.products or []
        purchase_orders = _runtime.purchase_orders or []

        # ── 1. Low inventory alerts ──────────────────────────
        for p in products:
            if p.current_stock <= 0:
                if self._should_fire(p.product_id, "stockout_alert"):
                    await self._bus.publish(SupplyChainEvent(
                        event_type="stockout_alert",
                        severity=AlertSeverity.CRITICAL,
                        source_agent="monitor",
                        description=f"STOCKOUT: {p.name} ({p.product_id}) has zero stock",
                        data={"product_id": p.product_id, "current_stock": p.current_stock},
                    ))
            elif p.current_stock < p.safety_stock:
                if self._should_fire(p.product_id, "low_stock_alert"):
                    await self._bus.publish(SupplyChainEvent(
                        event_type="low_stock_alert",
                        severity=AlertSeverity.HIGH,
                        source_agent="monitor",
                        description=(
                            f"Low stock: {p.name} ({p.product_id}) "
                            f"at {p.current_stock:.0f} (safety={p.safety_stock:.0f})"
                        ),
                        data={
                            "product_id": p.product_id,
                            "current_stock": p.current_stock,
                            "safety_stock": p.safety_stock,
                        },
                    ))
            elif p.current_stock > p.reorder_point * 3:
                if self._should_fire(p.product_id, "overstock_alert"):
                    await self._bus.publish(SupplyChainEvent(
                        event_type="overstock_alert",
                        severity=AlertSeverity.MEDIUM,
                        source_agent="monitor",
                        description=(
                            f"Overstock: {p.name} ({p.product_id}) "
                            f"at {p.current_stock:.0f} (3x ROP={p.reorder_point * 3:.0f})"
                        ),
                        data={
                            "product_id": p.product_id,
                            "current_stock": p.current_stock,
                            "reorder_point": p.reorder_point,
                        },
                    ))

        # ── 2. KPI threshold checks ─────────────────────────
        if self._tick_count % 5 == 0:  # every 5th tick
            snapshot = self._kpi.calculate_snapshot(
                products, purchase_orders, _runtime.suppliers or []
            )
            violations = self._kpi.check_thresholds(snapshot)
            for event in violations:
                await self._bus.publish(event)

            await self._bus.publish(SupplyChainEvent(
                event_type="kpi_snapshot_created",
                severity=AlertSeverity.LOW,
                source_agent="monitor",
                description="KPI snapshot calculated",
                data=snapshot.model_dump(),
            ))

        # ── 3. Delivery delay alerts ─────────────────────────
        now = datetime.now(UTC)
        for po in purchase_orders:
            if (
                po.status.value in ("pending", "approved", "shipped")
                and po.expected_delivery
                and _ensure_utc(po.expected_delivery) < now
            ):
                delay_days = (now - _ensure_utc(po.expected_delivery)).days
                if self._should_fire(po.po_id, "delivery_delayed"):
                    await self._bus.publish(SupplyChainEvent(
                        event_type="delivery_delayed",
                        severity=AlertSeverity.HIGH if delay_days > 3 else AlertSeverity.MEDIUM,
                        source_agent="monitor",
                        description=(
                            f"PO {po.po_id} delayed by {delay_days} days "
                            f"(product={po.product_id}, supplier={po.supplier_id})"
                        ),
                        data={
                            "po_id": po.po_id,
                            "delay_days": delay_days,
                            "product_id": po.product_id,
                            "supplier_id": po.supplier_id,
                        },
                    ))

        # ── 4. Anomaly detection ─────────────────────────────
        if self._anomaly and self._tick_count % 3 == 0:
            anomalies = self._anomaly.detect_batch(products[:10])
            for anomaly in anomalies:
                entity = getattr(anomaly, "product_id", None) or anomaly.anomaly_id
                if self._should_fire(entity, "anomaly_detected"):
                    await self._bus.publish(SupplyChainEvent(
                        event_type="anomaly_detected",
                        severity=anomaly.severity,
                        source_agent="monitor",
                        description=anomaly.description,
                        data=anomaly.model_dump(),
                    ))

        # ── Tick event (for agents that act every tick) ──────
        await self._bus.publish(SupplyChainEvent(
            event_type="tick",
            severity=AlertSeverity.LOW,
            source_agent="monitor",
            description=f"Monitor tick #{self._tick_count}",
            data={"tick": self._tick_count},
        ))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("proactive_monitor_stopped", total_ticks=self._tick_count)
