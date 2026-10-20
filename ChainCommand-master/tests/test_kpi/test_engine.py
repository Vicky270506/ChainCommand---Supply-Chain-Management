"""Tests for KPI calculation engine."""

from __future__ import annotations

from datetime import UTC, datetime

from chaincommand.data.schemas import (
    KPISnapshot,
    OrderStatus,
    PurchaseOrder,
)


class TestKPICalculation:
    def test_snapshot_basic(self, kpi_engine, sample_products, sample_suppliers):
        snapshot = kpi_engine.calculate_snapshot(sample_products, [], sample_suppliers)
        assert isinstance(snapshot, KPISnapshot)
        assert 0 <= snapshot.fill_rate <= 1
        assert snapshot.dsi >= 0

    def test_snapshot_with_orders(self, kpi_engine, sample_products, sample_suppliers):
        po = PurchaseOrder(
            supplier_id="SUP-0000",
            product_id="PRD-0000",
            quantity=100,
            unit_cost=10.0,
            total_cost=1000.0,
            status=OrderStatus.DELIVERED,
            expected_delivery=datetime.now(UTC),
        )
        snapshot = kpi_engine.calculate_snapshot(sample_products, [po], sample_suppliers)
        assert snapshot.otif >= 0

    def test_stockout_count(self, kpi_engine, sample_suppliers):
        from chaincommand.data.schemas import Product, ProductCategory

        products = [
            Product(
                product_id="P1",
                name="Empty",
                category=ProductCategory.FOOD,
                unit_cost=5,
                selling_price=10,
                current_stock=0.0,
                daily_demand_avg=10.0,
            ),
        ]
        snapshot = kpi_engine.calculate_snapshot(products, [], sample_suppliers)
        assert snapshot.stockout_count == 1

    def test_inventory_value(self, kpi_engine, sample_products, sample_suppliers):
        snapshot = kpi_engine.calculate_snapshot(sample_products, [], sample_suppliers)
        expected = sum(p.current_stock * p.unit_cost for p in sample_products)
        assert abs(snapshot.total_inventory_value - round(expected, 2)) < 1.0

    def test_history_appended(self, kpi_engine, sample_products, sample_suppliers):
        kpi_engine.calculate_snapshot(sample_products, [], sample_suppliers)
        kpi_engine.calculate_snapshot(sample_products, [], sample_suppliers)
        assert len(kpi_engine.history) == 2


class TestKPIThresholds:
    def test_no_violations_good_kpi(self, kpi_engine):
        snapshot = KPISnapshot(
            otif=0.98, fill_rate=0.99, mape=5.0, dsi=30.0,
            stockout_count=0,
        )
        events = kpi_engine.check_thresholds(snapshot)
        assert len(events) == 0

    def test_otif_violation(self, kpi_engine):
        snapshot = KPISnapshot(otif=0.80, fill_rate=0.99, mape=5.0, dsi=30.0)
        events = kpi_engine.check_thresholds(snapshot)
        otif_events = [e for e in events if "OTIF" in e.description]
        assert len(otif_events) == 1

    def test_stockout_violation(self, kpi_engine):
        snapshot = KPISnapshot(
            otif=0.99, fill_rate=0.99, mape=5.0, dsi=30.0,
            stockout_count=10,
        )
        events = kpi_engine.check_thresholds(snapshot)
        stockout_events = [e for e in events if "Stockout" in e.description]
        assert len(stockout_events) == 1
