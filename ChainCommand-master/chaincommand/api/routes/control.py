"""Control routes — simulation start/stop/speed."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...auth import require_api_key
from ...config import settings
from ...utils.logging_config import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["control"], dependencies=[Depends(require_api_key)])


@router.post("/simulation/start")
async def start_simulation():
    """Start the simulation loop."""
    from ...orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    started = await orchestrator.start_loop()
    if not started:
        return {"status": "already_running"}
    from ...orchestrator import _runtime
    return {"status": "started", "speed": _runtime.runtime_config.get("simulation_speed", settings.simulation_speed)}


@router.post("/simulation/stop")
async def stop_simulation():
    """Stop the simulation loop."""
    from ...orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    was_running = orchestrator.running
    await orchestrator.stop_loop()
    return {"status": "stopped" if was_running else "already_stopped"}


@router.post("/simulation/speed")
async def set_speed(speed: float):
    """Adjust simulation speed multiplier."""
    if speed < settings.simulation_speed_min or speed > settings.simulation_speed_max:
        raise HTTPException(
            status_code=422,
            detail=f"Speed must be between {settings.simulation_speed_min} and {settings.simulation_speed_max}",
        )

    from ...orchestrator import _runtime
    _runtime.runtime_config["simulation_speed"] = speed
    log.info("simulation_speed_changed", speed=speed)
    return {"status": "ok", "speed": speed}


@router.get("/simulation/status")
async def simulation_status():
    """Get current simulation status."""
    from ...orchestrator import _runtime, get_orchestrator

    orchestrator = get_orchestrator()
    return {
        "running": orchestrator.running,
        "cycle_count": orchestrator.cycle_count,
        "speed": _runtime.runtime_config.get("simulation_speed", settings.simulation_speed),
        "products": len(_runtime.products) if _runtime.products else 0,
        "suppliers": len(_runtime.suppliers) if _runtime.suppliers else 0,
        "purchase_orders": len(_runtime.purchase_orders),
        "pending_approvals": len(_runtime.pending_approvals),
        "events": _runtime.event_bus.event_count if _runtime.event_bus else 0,
    }
