"""MLOps endpoints: registry status, monitoring report, manual retraining."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from ...db import get_db
from ...db.models import User
from ...mlops.status import registry_status
from ..deps import require_admin, require_manager

router = APIRouter(prefix="/mlops", tags=["mlops"])
log = logging.getLogger(__name__)
_last_job: dict = {"status": "idle"}


@router.get("/models")
def models(db: Session = Depends(get_db), _u: User = Depends(require_manager)):
    """Registered model versions with their validation metrics and deployment status."""
    return registry_status(session=db)


@router.get("/monitoring")
def monitoring(window_days: int = 28, _u: User = Depends(require_manager)):
    """Live accuracy of the champion, data drift and business metrics."""
    from ...mlops.monitoring import run_monitoring
    report = run_monitoring(window_days=window_days, log_to_mlflow=False)
    return report


def _retrain_job(full: bool) -> None:
    from ...mlops.retrain import retrain
    _last_job.update(status="running")
    try:
        result = retrain(trigger="manual-api", full=full)
        _last_job.update(status="done", forecaster={k: v for k, v in result["forecaster"].items() if k != "metrics"})
    except Exception as exc:  # noqa: BLE001
        log.exception("manual retrain failed")
        _last_job.update(status="failed", error=str(exc))


@router.post("/retrain", status_code=202)
def retrain_now(background: BackgroundTasks, full: bool = False, _u: User = Depends(require_admin)):
    """Start a challenger training run in the background (admin only)."""
    if _last_job.get("status") == "running":
        return {"status": "already running"}
    background.add_task(_retrain_job, full)
    return {"status": "started"}


@router.get("/retrain/status")
def retrain_status(_u: User = Depends(require_manager)):
    return _last_job
