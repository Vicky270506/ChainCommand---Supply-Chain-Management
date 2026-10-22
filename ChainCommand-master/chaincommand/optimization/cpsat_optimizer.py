"""CP-SAT MILP optimizer for supplier allocation (Section 4.3)."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..config import settings
from ..utils.logging_config import get_logger

log = get_logger(__name__)


# ── Data models ───────────────────────────────────────────

class SupplierCandidate(BaseModel):
    """A candidate supplier for allocation."""

    supplier_id: str
    unit_cost: float
    risk_score: float = 0.0  # 0-1, higher = riskier
    capacity: float = 10_000.0
    min_order_qty: float = 0.0
    lead_time_days: float = 7.0


class AllocationResult(BaseModel):
    """Result of a CP-SAT supplier allocation optimization."""

    allocations: Dict[str, float] = Field(default_factory=dict)  # supplier_id -> qty
    total_cost: float = 0.0
    total_risk: float = 0.0
    objective_value: float = 0.0
    solver_status: str = "unknown"
    solve_time_ms: float = 0.0
    method: str = "cpsat"


class SensitivityResult(BaseModel):
    """Result of sensitivity analysis over risk-cost trade-off."""

    lambda_values: List[float] = Field(default_factory=list)
    costs: List[float] = Field(default_factory=list)
    risks: List[float] = Field(default_factory=list)
    elbow_lambda: float = 0.0
    elbow_cost: float = 0.0
    elbow_risk: float = 0.0


# ── CP-SAT Optimizer ─────────────────────────────────────

class SupplierAllocationOptimizer:
    """CP-SAT MILP optimizer for supplier allocation.

    Objective: min Σ(c_i · x_i) + λ · Σ(r_i · x_i)

    Subject to:
        - Σ x_i >= demand               (demand satisfaction)
        - x_i >= MOQ_i · y_i            (minimum order quantities)
        - x_i <= cap_i · y_i            (capacity limits)
        - Σ y_i <= max_suppliers         (max active suppliers)
        - lead_time_i <= max_lead_time   (lead-time constraint)

    Falls back to a greedy heuristic if ortools is not installed.
    """

    def __init__(self) -> None:
        self._has_ortools = False
        try:
            from ortools.sat.python import cp_model  # noqa: F401
            self._has_ortools = True
        except ImportError:
            log.info("ortools_unavailable", fallback="greedy_heuristic")

    def optimize(
        self,
        candidates: List[SupplierCandidate],
        demand: float,
        risk_lambda: Optional[float] = None,
        max_suppliers: Optional[int] = None,
        max_lead_time: Optional[float] = None,
        time_limit_ms: Optional[int] = None,
    ) -> AllocationResult:
        lam = risk_lambda if risk_lambda is not None else settings.ortools_risk_lambda
        max_sup = max_suppliers if max_suppliers is not None else settings.ortools_max_suppliers
        tl = time_limit_ms if time_limit_ms is not None else settings.ortools_time_limit_ms

        if not candidates:
            log.warning("optimize_empty_candidates", msg="No supplier candidates provided")
            return AllocationResult(solver_status="no_candidates")

        if self._has_ortools:
            return self._solve_cpsat(candidates, demand, lam, max_sup, max_lead_time, tl)
        return self._solve_greedy(candidates, demand, lam, max_sup, max_lead_time)

    def _solve_cpsat(
        self,
        candidates: List[SupplierCandidate],
        demand: float,
        risk_lambda: float,
        max_suppliers: int,
        max_lead_time: Optional[float],
        time_limit_ms: int,
    ) -> AllocationResult:
        import time as _time

        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        n = len(candidates)
        scale = 100  # scale float -> int for CP-SAT (use round() to avoid truncation bias)

        # Decision variables
        x = [model.new_int_var(0, round(c.capacity * scale), f"x_{i}") for i, c in enumerate(candidates)]
        y = [model.new_bool_var(f"y_{i}") for i in range(n)]

        # Demand constraint: Σ x_i >= demand * scale
        model.add(sum(x) >= round(demand * scale))

        for i, c in enumerate(candidates):
            # Link x and y: x_i <= cap_i * y_i
            model.add(x[i] <= round(c.capacity * scale) * y[i])

            # MOQ: x_i >= MOQ_i * y_i (when selected)
            if c.min_order_qty > 0:
                model.add(x[i] >= round(c.min_order_qty * scale) * y[i])

            # Lead-time filter
            if max_lead_time is not None and c.lead_time_days > max_lead_time:
                model.add(y[i] == 0)

        # Max suppliers
        model.add(sum(y) <= max_suppliers)

        # Objective: min Σ(c_i · x_i) + λ · Σ(r_i · x_i) (all scaled to int)
        cost_terms = []
        risk_terms = []
        for i, c in enumerate(candidates):
            cost_terms.append(round(c.unit_cost * 1000) * x[i])
            risk_terms.append(round(c.risk_score * risk_lambda * 1000) * x[i])

        model.minimize(sum(cost_terms) + sum(risk_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_ms / 1000.0

        t0 = _time.monotonic()
        status = solver.solve(model)
        solve_ms = (_time.monotonic() - t0) * 1000

        status_name = {
            cp_model.OPTIMAL: "optimal",
            cp_model.FEASIBLE: "feasible",
            cp_model.INFEASIBLE: "infeasible",
            cp_model.MODEL_INVALID: "invalid",
        }.get(status, "unknown")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            allocations = {}
            total_cost = 0.0
            total_risk = 0.0
            for i, c in enumerate(candidates):
                qty = solver.value(x[i]) / scale
                if qty > 0:
                    allocations[c.supplier_id] = round(qty, 2)
                    total_cost += qty * c.unit_cost
                    total_risk += qty * c.risk_score

            return AllocationResult(
                allocations=allocations,
                total_cost=round(total_cost, 2),
                total_risk=round(total_risk, 4),
                objective_value=round(solver.objective_value / (scale * 1000), 2),
                solver_status=status_name,
                solve_time_ms=round(solve_ms, 1),
                method="cpsat",
            )

        log.warning("cpsat_infeasible", status=status_name)
        return AllocationResult(solver_status=status_name, solve_time_ms=round(solve_ms, 1))

    def _solve_greedy(
        self,
        candidates: List[SupplierCandidate],
        demand: float,
        risk_lambda: float,
        max_suppliers: int,
        max_lead_time: Optional[float],
    ) -> AllocationResult:
        """Greedy heuristic fallback when OR-Tools is unavailable."""
        filtered = candidates
        if max_lead_time is not None:
            filtered = [c for c in candidates if c.lead_time_days <= max_lead_time]

        # Sort by composite score: cost + lambda * risk
        scored = sorted(filtered, key=lambda c: c.unit_cost + risk_lambda * c.risk_score)

        allocations: Dict[str, float] = {}
        remaining = demand
        total_cost = 0.0
        total_risk = 0.0

        for c in scored[:max_suppliers]:
            if remaining <= 0:
                break

            # Skip this supplier if remaining demand is below its MOQ:
            # ordering MOQ would overshoot demand, and capping to remaining
            # would silently violate MOQ.
            if c.min_order_qty > 0 and remaining < c.min_order_qty:
                continue

            qty = min(remaining, c.capacity)
            if c.min_order_qty > 0:
                qty = max(qty, c.min_order_qty)
            qty = min(qty, c.capacity)
            allocations[c.supplier_id] = round(qty, 2)
            total_cost += qty * c.unit_cost
            total_risk += qty * c.risk_score
            remaining -= qty
            if remaining <= 0:
                break

        solver_status = "greedy_partial" if remaining > 0 else "greedy_fallback"

        return AllocationResult(
            allocations=allocations,
            total_cost=round(total_cost, 2),
            total_risk=round(total_risk, 4),
            objective_value=round(total_cost + risk_lambda * total_risk, 2),
            solver_status=solver_status,
            method="greedy",
        )

    def sensitivity_analysis(
        self,
        candidates: List[SupplierCandidate],
        demand: float,
        steps: int = 11,
    ) -> SensitivityResult:
        """Sweep λ from 0→1, find the cost-risk elbow point."""
        lambdas = [i / max(1, steps - 1) for i in range(steps)]
        costs: List[float] = []
        risks: List[float] = []

        for lam in lambdas:
            result = self.optimize(candidates, demand, risk_lambda=lam)
            costs.append(result.total_cost)
            risks.append(result.total_risk)

        # Find elbow using a combined approach:
        # 1. Compute normalized second derivative of (cost + risk) curve
        # 2. Compute percentage-change in the composite objective
        # 3. Pick the point where both signals agree the trade-off is sharpest
        elbow_idx = 0
        if len(lambdas) >= 3:
            composite = [c + r for c, r in zip(costs, risks, strict=False)]

            # ── Second-derivative analysis (normalized) ───────────
            second_derivs = []
            for i in range(1, len(composite) - 1):
                d2 = composite[i + 1] - 2 * composite[i] + composite[i - 1]
                second_derivs.append(abs(d2))

            # Normalize to [0, 1] to make it comparable
            sd_max = max(second_derivs) if second_derivs else 1.0
            sd_norm = [v / sd_max if sd_max > 0 else 0.0 for v in second_derivs]

            # ── Percentage-change analysis ────────────────────────
            pct_changes = []
            for i in range(1, len(composite) - 1):
                prev_delta = composite[i] - composite[i - 1]
                next_delta = composite[i + 1] - composite[i]
                denom = abs(prev_delta) if abs(prev_delta) > 1e-9 else 1e-9
                pct_change = abs((next_delta - prev_delta) / denom)
                pct_changes.append(pct_change)

            pc_max = max(pct_changes) if pct_changes else 1.0
            pc_norm = [v / pc_max if pc_max > 0 else 0.0 for v in pct_changes]

            # ── Combined score (equal weight) ─────────────────────
            combined = [0.5 * s + 0.5 * p for s, p in zip(sd_norm, pc_norm, strict=False)]
            if combined:
                elbow_idx = combined.index(max(combined)) + 1

        return SensitivityResult(
            lambda_values=lambdas,
            costs=costs,
            risks=risks,
            elbow_lambda=lambdas[elbow_idx],
            elbow_cost=costs[elbow_idx],
            elbow_risk=risks[elbow_idx],
        )
