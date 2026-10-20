"""Property-based tests using Hypothesis.

Tests actual business-logic invariants rather than mere Pydantic assignment:
  - GA optimization always produces valid, bounded results
  - BOM explosion quantities are always positive
  - Supplier allocation always meets demand
  - Risk scores stay within [0, 1]
"""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

pytestmark = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")


if HAS_HYPOTHESIS:

    class TestGAOptimizationInvariants:
        """GA optimizer must always return physically valid inventory parameters."""

        @given(
            daily_demand_avg=st.floats(min_value=1.0, max_value=500.0),
            daily_demand_std=st.floats(min_value=0.1, max_value=50.0),
            lead_time_days=st.integers(min_value=1, max_value=30),
            min_order_qty=st.integers(min_value=10, max_value=500),
        )
        @settings(max_examples=30)
        def test_ga_results_are_physically_valid(
            self, daily_demand_avg, daily_demand_std, lead_time_days, min_order_qty,
        ):
            from chaincommand.data.schemas import Product, ProductCategory
            from chaincommand.models.optimizer import GeneticOptimizer

            product = Product(
                name="HypTest",
                category=ProductCategory.ELECTRONICS,
                unit_cost=10.0,
                selling_price=25.0,
                daily_demand_avg=daily_demand_avg,
                daily_demand_std=daily_demand_std,
                lead_time_days=lead_time_days,
                min_order_qty=min_order_qty,
            )
            opt = GeneticOptimizer()
            result = opt.optimize(product, demand_forecast=[])

            # Reorder point must be positive (avg_demand * lead_time + safety_stock)
            assert result.recommended_reorder_point > 0, (
                f"reorder_point must be > 0, got {result.recommended_reorder_point}"
            )
            # Safety stock must be non-negative
            assert result.recommended_safety_stock >= 0, (
                f"safety_stock must be >= 0, got {result.recommended_safety_stock}"
            )
            # Order qty must be at least min_order_qty
            assert result.recommended_order_qty >= min_order_qty, (
                f"order_qty ({result.recommended_order_qty}) < min_order_qty ({min_order_qty})"
            )
            # Expected cost saving is non-negative
            assert result.expected_cost_saving >= 0

    class TestBOMExplosionInvariants:
        """BOM explosion quantities must always be positive."""

        @given(n=st.integers(min_value=1, max_value=10))
        @settings(max_examples=10)
        def test_bom_explosion_quantities_positive(self, n):
            from chaincommand.bom.manager import BOMManager

            mgr = BOMManager()
            trees = mgr.generate_synthetic_boms(n_assemblies=n)
            assert len(trees) == n
            for tree in trees:
                assert len(tree.root_items) >= 1
                # Explode every root and verify quantities
                for root in tree.root_items:
                    rows = tree.explode(root.part_id, parent_qty=1.0)
                    for row in rows:
                        assert row.extended_quantity > 0, (
                            f"Part {row.part_id}: extended_quantity must be > 0, "
                            f"got {row.extended_quantity}"
                        )
                        assert row.unit_cost >= 0, (
                            f"Part {row.part_id}: unit_cost must be >= 0"
                        )
                        assert row.extended_cost >= 0, (
                            f"Part {row.part_id}: extended_cost must be >= 0"
                        )

    class TestSupplierAllocationInvariants:
        """CP-SAT supplier allocation must always meet demand."""

        @given(demand=st.floats(min_value=100, max_value=10000))
        @settings(max_examples=20)
        def test_optimizer_satisfies_demand(self, demand):
            from chaincommand.optimization.cpsat_optimizer import (
                SupplierAllocationOptimizer,
                SupplierCandidate,
            )

            candidates = [
                SupplierCandidate(
                    supplier_id="S1", unit_cost=10.0, risk_score=0.1, capacity=50000,
                ),
                SupplierCandidate(
                    supplier_id="S2", unit_cost=12.0, risk_score=0.05, capacity=50000,
                ),
            ]
            opt = SupplierAllocationOptimizer()
            result = opt.optimize(candidates, demand=demand)
            total = sum(result.allocations.values())
            assert total >= demand * 0.99  # allow tiny rounding

    class TestRiskScoreInvariants:
        """Risk scores must remain in [0, 1] with valid risk levels."""

        @given(
            on_time=st.floats(min_value=0.5, max_value=0.99),
            defect=st.floats(min_value=0.001, max_value=0.1),
        )
        @settings(max_examples=20)
        def test_risk_score_in_range(self, on_time, defect):
            from chaincommand.risk.scorer import SupplierMetrics, SupplierRiskScorer

            scorer = SupplierRiskScorer()
            metrics = SupplierMetrics(supplier_id="TEST", on_time_rate=on_time, defect_rate=defect)
            score = scorer.score_supplier(metrics)
            assert 0 <= score.overall_score <= 1
            assert score.risk_level in ("low", "medium", "high", "critical")
