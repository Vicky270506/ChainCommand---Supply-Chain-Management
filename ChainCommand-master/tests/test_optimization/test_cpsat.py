"""Tests for CP-SAT supplier allocation optimizer."""

from __future__ import annotations

import pytest

from chaincommand.optimization.cpsat_optimizer import (
    AllocationResult,
    SensitivityResult,
    SupplierAllocationOptimizer,
    SupplierCandidate,
)


@pytest.fixture
def candidates():
    return [
        SupplierCandidate(
            supplier_id="S1", unit_cost=10.0, risk_score=0.1,
            capacity=5000, min_order_qty=100, lead_time_days=5,
        ),
        SupplierCandidate(
            supplier_id="S2", unit_cost=12.0, risk_score=0.05,
            capacity=8000, min_order_qty=200, lead_time_days=3,
        ),
        SupplierCandidate(
            supplier_id="S3", unit_cost=8.0, risk_score=0.3,
            capacity=3000, min_order_qty=50, lead_time_days=10,
        ),
        SupplierCandidate(
            supplier_id="S4", unit_cost=15.0, risk_score=0.02,
            capacity=10000, min_order_qty=500, lead_time_days=7,
        ),
    ]


@pytest.fixture
def optimizer():
    return SupplierAllocationOptimizer()


class TestSupplierAllocationOptimizer:
    def test_optimize_returns_result(self, optimizer, candidates):
        result = optimizer.optimize(candidates, demand=1000)
        assert isinstance(result, AllocationResult)
        assert result.total_cost > 0

    def test_demand_satisfied(self, optimizer, candidates):
        result = optimizer.optimize(candidates, demand=1000)
        total_allocated = sum(result.allocations.values())
        assert total_allocated >= 999  # allow small rounding

    def test_max_suppliers_constraint(self, optimizer, candidates):
        result = optimizer.optimize(candidates, demand=1000, max_suppliers=2)
        assert len(result.allocations) <= 2

    def test_capacity_respected(self, optimizer, candidates):
        result = optimizer.optimize(candidates, demand=1000)
        for sid, qty in result.allocations.items():
            cand = next(c for c in candidates if c.supplier_id == sid)
            assert qty <= cand.capacity + 1  # small rounding tolerance

    def test_lead_time_filter(self, optimizer, candidates):
        result = optimizer.optimize(candidates, demand=1000, max_lead_time=6.0)
        for sid in result.allocations:
            cand = next(c for c in candidates if c.supplier_id == sid)
            assert cand.lead_time_days <= 6.0

    def test_risk_lambda_zero(self, optimizer, candidates):
        result = optimizer.optimize(candidates, demand=1000, risk_lambda=0.0)
        assert isinstance(result, AllocationResult)

    def test_risk_lambda_one(self, optimizer, candidates):
        result = optimizer.optimize(candidates, demand=1000, risk_lambda=1.0)
        assert isinstance(result, AllocationResult)

    def test_empty_candidates(self, optimizer):
        result = optimizer.optimize([], demand=1000)
        assert result.total_cost == 0 or result.solver_status in ("infeasible", "greedy_fallback")


class TestSensitivityAnalysis:
    def test_sweep_returns_result(self, optimizer, candidates):
        result = optimizer.sensitivity_analysis(candidates, demand=1000, steps=5)
        assert isinstance(result, SensitivityResult)
        assert len(result.lambda_values) == 5
        assert len(result.costs) == 5
        assert len(result.risks) == 5
        assert 0 <= result.elbow_lambda <= 1

    def test_elbow_within_range(self, optimizer, candidates):
        result = optimizer.sensitivity_analysis(candidates, demand=1000)
        assert result.elbow_lambda in result.lambda_values
