"""Dashboard routes — KPI, inventory, BOM, risk, CTB, forecast, approvals, AWS."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...auth import require_api_key
from ...config import settings
from ...data.schemas import ApprovalStatus
from ...kpi.engine import ALLOWED_KPI_METRICS
from ...utils.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["dashboard"], dependencies=[Depends(require_api_key)])


# ── KPI ──────────────────────────────────────────────────────

@router.get("/kpi/current")
async def get_current_kpi():
    """Get the latest KPI snapshot."""
    from ...orchestrator import _runtime

    if not _runtime.kpi_engine or not _runtime.kpi_engine.history:
        raise HTTPException(status_code=503, detail="No KPI data available yet")
    snapshot = _runtime.kpi_engine.history[-1]
    return snapshot.model_dump()


@router.get("/kpi/history")
async def get_kpi_history(periods: int = Query(default=30, ge=1, le=365)):
    """Get KPI history for trend analysis."""
    from ...orchestrator import _runtime

    if not _runtime.kpi_engine:
        return {"snapshots": [], "count": 0}
    snapshots = _runtime.kpi_engine.history[-periods:]
    return {
        "snapshots": [s.model_dump() for s in snapshots],
        "count": len(snapshots),
    }


# ── Inventory ────────────────────────────────────────────────

@router.get("/inventory/status")
async def get_inventory_status(product_id: Optional[str] = None):
    """Get inventory status for all or a specific product."""
    from ...orchestrator import _runtime

    products = _runtime.products or []
    if product_id:
        products = [p for p in products if p.product_id == product_id]
        if not products:
            raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")

    items = []
    for p in products:
        dsi = p.current_stock / p.daily_demand_avg if p.daily_demand_avg > 0 else None
        items.append({
            "product_id": p.product_id,
            "name": p.name,
            "category": p.category.value,
            "current_stock": p.current_stock,
            "reorder_point": p.reorder_point,
            "safety_stock": p.safety_stock,
            "daily_demand_avg": p.daily_demand_avg,
            "days_of_supply": round(dsi, 1) if dsi is not None else None,
            "unit_cost": p.unit_cost,
            "status": (
                "critical" if p.current_stock < p.safety_stock
                else "low" if p.current_stock < p.reorder_point
                else "healthy"
            ),
        })

    return {"products": items, "count": len(items)}


# ── BOM ──────────────────────────────────────────────────────

@router.get("/bom/summary")
async def get_bom_summary():
    """Get BOM management summary."""
    from ...orchestrator import _runtime

    if not _runtime.bom_manager:
        raise HTTPException(status_code=503, detail="BOM manager not initialized")
    return _runtime.bom_manager.get_summary()


@router.get("/bom/risks")
async def get_bom_risks():
    """Get single-source and long-lead-time risks."""
    from ...orchestrator import _runtime

    if not _runtime.bom_manager:
        raise HTTPException(status_code=503, detail="BOM manager not initialized")
    return {
        "single_source": _runtime.bom_manager.find_single_source_risks(),
        "long_lead": _runtime.bom_manager.find_long_lead_items(settings.bom_long_lead_threshold_days),
    }


# ── Risk Scoring ─────────────────────────────────────────────

@router.get("/risk/scores")
async def get_risk_scores(limit: int = Query(default=20, ge=1, le=100)):
    """Get supplier risk scores."""
    from ...orchestrator import _runtime

    if not _runtime.risk_scorer or not _runtime.suppliers:
        return {"scores": [], "count": 0}

    from ...risk.scorer import SupplierMetrics

    scores = []
    for s in _runtime.suppliers[:limit]:
        metrics = SupplierMetrics(
            supplier_id=s.supplier_id,
            on_time_rate=s.on_time_rate,
            defect_rate=s.defect_rate,
            lead_time_mean=s.lead_time_mean,
            lead_time_std=s.lead_time_std,
        )
        score = _runtime.risk_scorer.score_supplier(metrics)
        scores.append({
            "supplier_id": s.supplier_id,
            "name": s.name,
            "overall_score": score.overall_score,
            "risk_level": score.risk_level,
            "delivery_risk": score.delivery_risk,
            "quality_risk": score.quality_risk,
            "recommendations": score.recommendations,
        })
    scores.sort(key=lambda x: -x["overall_score"])
    return {"scores": scores, "count": len(scores)}


# ── CTB ──────────────────────────────────────────────────────

@router.get("/ctb/status")
async def get_ctb_status():
    """Get Clear-to-Build status for all assemblies."""
    from ...orchestrator import _runtime

    results = _runtime.last_cycle_results.get("ctb", [])
    return {"reports": results, "count": len(results)}


# ── Events ───────────────────────────────────────────────────

@router.get("/events/recent")
async def get_recent_events(limit: int = Query(default=50, ge=1, le=200)):
    """Get recent supply chain events."""
    from ...orchestrator import _runtime

    if not _runtime.event_bus:
        return {"events": [], "count": 0}
    events = _runtime.event_bus.recent_events[-limit:]
    return {
        "events": [e.model_dump() for e in reversed(events)],
        "count": len(events),
    }


# ── Forecast ─────────────────────────────────────────────────

@router.get("/forecast/{product_id}")
async def get_forecast(product_id: str, horizon: int = Query(default=30, ge=1, le=90)):
    """Get demand forecast for a product."""
    from ...orchestrator import _runtime

    if not _runtime.forecaster:
        raise HTTPException(status_code=503, detail="Forecaster not initialized")

    import asyncio
    results = await asyncio.to_thread(_runtime.forecaster.predict, product_id, horizon)
    accuracy = (
        _runtime.forecaster.get_accuracy(product_id)
        if hasattr(_runtime.forecaster, "get_accuracy")
        else {}
    )
    return {
        "product_id": product_id,
        "horizon": horizon,
        "forecasts": [r.model_dump() for r in results],
        "accuracy": accuracy,
    }


# ── Human Approval ───────────────────────────────────────────

@router.get("/approvals/pending")
async def get_pending_approvals():
    """List all pending approval requests."""
    from ...orchestrator import _runtime

    pending = {
        k: v.model_dump()
        for k, v in _runtime.pending_approvals.items()
        if v.status == ApprovalStatus.PENDING
    }
    return {"approvals": pending, "count": len(pending)}


@router.post("/approval/{request_id}/decide")
async def decide_approval(request_id: str, approved: bool, reason: str = ""):
    """Human decision on an approval request.

    TODO: Production deployments must add RBAC — only authorized roles
    (e.g., supply-chain-manager) should be allowed to approve/reject.
    """
    from ...orchestrator import _runtime

    approval = _runtime.pending_approvals.get(request_id)
    if not approval:
        raise HTTPException(status_code=404, detail=f"Approval request {request_id} not found")

    approval.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
    approval.decided_at = datetime.now(timezone.utc)
    approval.decided_by = "human"
    approval.reason = reason

    log.info("approval_decided", request_id=request_id, approved=approved)
    return {"request_id": request_id, "status": approval.status.value, "reason": reason}


# ── AWS Integration ──────────────────────────────────────

@router.get("/aws/status")
async def get_aws_status():
    """Return AWS connection status and configuration.

    TODO: This endpoint exposes internal config (region, bucket name).
    In production, restrict to admin roles or redact sensitive fields.
    """
    from ...orchestrator import _runtime

    backend_type = type(_runtime.backend).__name__ if _runtime.backend else "None"
    return {
        "enabled": settings.aws_enabled,
        "backend": backend_type,
        "region": settings.aws_region,
        "s3_bucket": settings.aws_s3_bucket,
    }


@router.get("/aws/kpi-trend/{metric}")
async def get_aws_kpi_trend(
    metric: str,
    days: int = Query(default=30, ge=1, le=365),
):
    """Query KPI trend from Redshift via the persistence backend."""
    # Allowlist validation to prevent SQL injection via metric path param
    if metric not in ALLOWED_KPI_METRICS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid metric '{metric}'. Allowed: {sorted(ALLOWED_KPI_METRICS)}",
        )

    from ...orchestrator import _runtime

    if not _runtime.backend or not settings.aws_enabled:
        raise HTTPException(status_code=503, detail="AWS backend not enabled")
    data = await _runtime.backend.query_kpi_trend(metric, days)
    return {"metric": metric, "days": days, "data": data}
