"""Benchmark CP-SAT vs GA optimizer on same problem instances."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from ..data.schemas import ForecastResult, Product
from ..models.optimizer import GeneticOptimizer
from ..utils.logging_config import get_logger
from .cpsat_optimizer import SupplierAllocationOptimizer, SupplierCandidate

log = get_logger(__name__)


class OptimizerBenchmark:
    """Run CP-SAT vs GA on the same instances and compare."""

    def __init__(self) -> None:
        self._cpsat = SupplierAllocationOptimizer()
        self._ga = GeneticOptimizer()

    def run(
        self,
        candidates: List[SupplierCandidate],
        demand: float,
        product: Product,
        forecast: List[ForecastResult] | None = None,
    ) -> Dict[str, Any]:
        """Compare both optimizers on the same problem."""
        forecast = forecast or []

        # CP-SAT
        t0 = time.monotonic()
        cpsat_result = self._cpsat.optimize(candidates, demand)
        cpsat_ms = (time.monotonic() - t0) * 1000

        # GA
        t0 = time.monotonic()
        ga_result = self._ga.optimize(product, forecast)
        ga_ms = (time.monotonic() - t0) * 1000

        # NOTE: CP-SAT and GA solve fundamentally different problems:
        #   CP-SAT  -> multi-supplier allocation  -> reports procurement cost
        #   GA      -> inventory policy (EOQ/ROP)  -> reports holding/ordering cost savings
        # A direct cost comparison is not meaningful, so we report each
        # optimizer's native metrics separately and estimate a GA procurement
        # cost only for rough reference (not a true optimality gap).
        unit_cost = product.unit_cost
        holding_cost_pct = 0.25
        ga_monthly_holding = (
            (ga_result.recommended_safety_stock + ga_result.recommended_order_qty / 2)
            * unit_cost * holding_cost_pct / 365
        ) * 30  # monthly holding cost estimate

        # Estimate GA procurement cost (order_qty * unit_cost) for rough comparison
        ga_procurement_cost = ga_result.recommended_order_qty * unit_cost

        report = {
            "cpsat": {
                "procurement_cost": cpsat_result.total_cost,
                "risk": cpsat_result.total_risk,
                "status": cpsat_result.solver_status,
                "time_ms": round(cpsat_ms, 1),
                "suppliers_used": len(cpsat_result.allocations),
            },
            "ga": {
                "procurement_cost_estimate": round(ga_procurement_cost, 2),
                "monthly_holding_cost": round(ga_monthly_holding, 2),
                "reorder_point": ga_result.recommended_reorder_point,
                "safety_stock": ga_result.recommended_safety_stock,
                "order_qty": ga_result.recommended_order_qty,
                "expected_saving": ga_result.expected_cost_saving,
                "time_ms": round(ga_ms, 1),
            },
            "note": (
                "CP-SAT solves supplier allocation (procurement cost); "
                "GA solves inventory policy (holding/ordering). "
                "Costs are not directly comparable."
            ),
        }

        log.info("benchmark_complete", cpsat_ms=cpsat_ms, ga_ms=ga_ms)
        return report
