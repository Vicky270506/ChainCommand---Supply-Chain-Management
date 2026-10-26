"""Retraining: scheduled (daily / weekly / monthly), drift-triggered and manual.

``python -m chaincommand.mlops scheduler`` runs :func:`tick` in a loop.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..config import settings
from ..db.base import get_engine
from ..decision.recommend import generate_recommendations
from ..features.dataset import build_dataset
from ..features.engineering import feature_columns
from ..forecasting.predict import save_forecasts
from . import registry
from .monitoring import run_monitoring
from .tracking import wait_for_server
from .training import train_forecaster, train_stockout

log = logging.getLogger("chaincommand.mlops")


def forecast_with_champion(engine=None, horizon: int | None = None,
                           extend: int | None = None) -> tuple[str, pd.DataFrame]:
    """Forecast the next days with the registered champion and save them to the forecasts table."""
    engine = engine or get_engine()
    horizon = horizon or settings.mlops_horizon
    model, version = registry.load_champion(registry.forecaster_name(horizon))
    df = build_dataset(engine, horizon=horizon, extend_days=extend or horizon)
    future = df[df["is_future"]].copy()
    pred = model.predict(future[feature_columns(horizon)])
    out = future[["date", "product_id"]].copy()
    out["forecast"] = pred["forecast"].to_numpy()
    out["lower"] = pred["lower"].to_numpy()
    out["upper"] = pred["upper"].to_numpy()
    version_tag = save_forecasts(engine, out, horizon, model_name=f"mlflow-v{version}")
    return version_tag, out


def refresh_plan(engine=None) -> dict:
    """New forecasts from the champion, then new pending recommendations for a human to review."""
    engine = engine or get_engine()
    version, out = forecast_with_champion(engine)
    _, plan = generate_recommendations(engine)
    return {"forecast_version": version, "forecast_rows": len(out), "recommendations": len(plan)}


def retrain(engine=None, trigger: str = "manual", full: bool = False) -> dict:
    """Train a challenger (forecaster, and stockout model when ``full``), promote if better, refresh the plan."""
    engine = engine or get_engine()
    results = {"forecaster": train_forecaster(engine, trigger=trigger)}
    if full:
        results["stockout"] = train_stockout(engine, trigger=trigger)
    results["plan"] = refresh_plan(engine)
    return results


# ── scheduler ────────────────────────────────────────────────────────────
def _state_path() -> Path:
    return Path(settings.mlops_state_file)


def _load_state() -> dict:
    try:
        return json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def _due(state: dict, job: str, every: timedelta, now: datetime) -> bool:
    last = state.get(job)
    return last is None or now - datetime.fromisoformat(last) >= every


def tick(engine=None, now: datetime | None = None) -> dict:
    """Run whichever jobs are due. Returns what was done (used by tests and logs)."""
    engine = engine or get_engine()
    now = now or datetime.now(timezone.utc)
    state = _load_state()
    done: dict = {}

    if _due(state, "full", timedelta(days=settings.mlops_full_retrain_every_days), now):
        done["full"] = retrain(engine, trigger="schedule-monthly", full=True)
        state.update(full=now.isoformat(), weekly=now.isoformat(), monitor=now.isoformat())
    elif _due(state, "weekly", timedelta(days=settings.mlops_retrain_every_days), now):
        done["weekly"] = retrain(engine, trigger="schedule-weekly")
        state.update(weekly=now.isoformat(), monitor=now.isoformat())
    elif _due(state, "monitor", timedelta(hours=settings.mlops_monitor_every_hours), now):
        report = run_monitoring(engine)
        done["monitor"] = {k: report[k] for k in ("retrain_recommended", "reasons", "drift_share")}
        state["monitor"] = now.isoformat()
        if report["retrain_recommended"]:                       # event / drift driven
            trigger = "drift" if report["drift_share"] >= settings.mlops_drift_share else "performance"
            done["triggered"] = retrain(engine, trigger=trigger)
            state["weekly"] = now.isoformat()
    _save_state(state)
    return done


def run_forever(poll_minutes: float = 15.0) -> None:
    wait_for_server()
    log.info("scheduler started (monitor every %sh, retrain every %sd, full every %sd)",
             settings.mlops_monitor_every_hours, settings.mlops_retrain_every_days,
             settings.mlops_full_retrain_every_days)
    while True:
        try:
            done = tick()
            if done:
                log.info("scheduler ran: %s", json.dumps(done, default=str)[:500])
        except Exception:  # keep the loop alive; the next tick tries again
            log.exception("scheduler tick failed")
        time.sleep(poll_minutes * 60)
