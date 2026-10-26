"""Monitoring: live accuracy of the champion, data drift and business metrics.

Accuracy is measured two ways:
* ``hold-out``  the champion scores the most recent days it was not trained on;
* ``forecast-log``  forecasts saved in the ``forecasts`` table are matched with the
  demand that happened afterwards (fills in as real days pass).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import mlflow

from ..config import settings
from ..db.base import get_engine
from ..db.models import Forecast, MLflowMetadata, RLRecommendation
from ..features.dataset import _daily_panel, build_dataset
from ..features.engineering import TARGET, feature_columns
from ..forecasting.metrics import bias, coverage, mae, rmse, smape
from . import registry
from .tracking import MONITOR_EXPERIMENT, configure, safe_metrics

PSI_CAP = 10.0          # keeps the value finite (and JSON-safe) when a feature changes completely
DRIFT_FEATURES = ["roll_mean_7", "roll_mean_28", "roll_std_28", "stock_at_origin",
                  "selling_price", "discount_pct", "demand"]


def _clean(obj):
    """Replace NaN/inf with None so the report is valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ── drift ────────────────────────────────────────────────────────────────
def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index (< 0.1 stable, 0.1-0.2 moderate, > 0.2 large shift)."""
    ref = reference.dropna().to_numpy(float)
    cur = current.dropna().to_numpy(float)
    if len(ref) < 20 or len(cur) < 20:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (almost) constant feature
        return 0.0 if np.allclose(np.mean(ref), np.mean(cur)) else PSI_CAP
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(min(np.sum((q - p) * np.log(q / p)), PSI_CAP))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features=DRIFT_FEATURES,
                 threshold: Optional[float] = None) -> pd.DataFrame:
    threshold = settings.mlops_psi_threshold if threshold is None else threshold
    rows = []
    for f in features:
        if f not in reference or f not in current:
            continue
        ks = stats.ks_2samp(reference[f].dropna(), current[f].dropna())
        value = psi(reference[f], current[f])
        rows.append({"feature": f, "psi": value, "ks_stat": float(ks.statistic),
                     "ref_mean": float(reference[f].mean()), "cur_mean": float(current[f].mean()),
                     "drifted": bool(value > threshold)})
    return pd.DataFrame(rows)


# ── accuracy ─────────────────────────────────────────────────────────────
def _accuracy(y, pred, lo=None, hi=None) -> dict:
    out = {"mae": mae(y, pred), "rmse": rmse(y, pred), "smape": smape(y, pred), "bias": bias(y, pred),
           "n": int(len(y))}
    if lo is not None:
        out["coverage"] = coverage(y, lo, hi)
    return out


def holdout_performance(model, recent: pd.DataFrame, horizon: int) -> dict:
    cols = feature_columns(horizon)
    pred = model.predict(recent[cols])
    return _accuracy(recent[TARGET].to_numpy(), pred["forecast"].to_numpy(),
                     pred["lower"].to_numpy(), pred["upper"].to_numpy())


def forecast_log_performance(engine, window_days: int) -> Optional[dict]:
    """Saved forecasts vs the demand that actually happened (None until some days have passed)."""
    panel = _daily_panel(engine)
    panel["actual"] = panel["units_sold"] + panel["lost_units"]
    since = panel["date"].max() - pd.Timedelta(days=window_days)
    fc = pd.read_sql(select(Forecast.product_id, Forecast.forecast_date, Forecast.predicted_demand,
                            Forecast.lower, Forecast.upper, Forecast.generated_at), engine)
    if fc.empty:
        return None
    fc["date"] = pd.to_datetime(fc["forecast_date"])
    # the latest forecast made *before* the day it predicts
    fc = fc[fc["generated_at"] < fc["date"] + pd.Timedelta(days=1)]
    fc = fc.sort_values("generated_at").drop_duplicates(["product_id", "date"], keep="last")
    m = fc.merge(panel[["product_id", "date", "actual"]], on=["product_id", "date"])
    m = m[m["date"] > since]
    if m.empty:
        return None
    return _accuracy(m["actual"].to_numpy(), m["predicted_demand"].to_numpy(),
                     m["lower"].to_numpy(), m["upper"].to_numpy())


# ── business metrics ─────────────────────────────────────────────────────
def business_metrics(engine, window_days: int) -> dict:
    panel = _daily_panel(engine)
    last = panel["date"].max()
    cur = panel[panel["date"] > last - pd.Timedelta(days=window_days)]
    prev = panel[(panel["date"] <= last - pd.Timedelta(days=window_days))]
    demand = (cur["units_sold"] + cur["lost_units"]).sum()
    out = {
        "fill_rate": float(cur["units_sold"].sum() / demand) if demand else 1.0,
        "stockout_rate": float((cur["lost_units"] > 0).mean()),
        "baseline_stockout_rate": float((prev["lost_units"] > 0).mean()) if len(prev) else 0.0,
        "avg_stock_units": float(cur["stock_close"].mean()),
    }
    with Session(engine) as s:
        rows = s.execute(select(RLRecommendation.approval_status, func.count())
                         .group_by(RLRecommendation.approval_status)).all()
    counts = {r[0]: r[1] for r in rows}
    decided = sum(v for k, v in counts.items() if k != "pending")
    out["recommendations_pending"] = float(counts.get("pending", 0))
    if decided:
        out["recommendation_approval_rate"] = counts.get("approved", 0) / decided
    return out


# ── the monitoring pass ──────────────────────────────────────────────────
def run_monitoring(engine=None, window_days: int = 28, log_to_mlflow: bool = True) -> dict:
    """Score the champion, check drift, and say whether retraining is advisable."""
    engine = engine or get_engine()
    horizon = settings.mlops_horizon
    name = registry.forecaster_name(horizon)
    df = build_dataset(engine, horizon=horizon)
    labelled = df[(~df["is_future"]) & df[TARGET].notna()]
    last = labelled["date"].max()
    recent = labelled[labelled["date"] > last - pd.Timedelta(days=window_days)]
    older = labelled[labelled["date"] <= last - pd.Timedelta(days=window_days)]

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "window_days": window_days,
                    "model": name, "reasons": []}
    drift = drift_report(older, recent)
    report["drift_share"] = float(drift["drifted"].mean()) if len(drift) else 0.0
    report["drifted_features"] = drift.loc[drift["drifted"], "feature"].tolist() if len(drift) else []
    if report["drift_share"] >= settings.mlops_drift_share:
        report["reasons"].append(f"data drift: {len(report['drifted_features'])}/{len(drift)} features "
                                 f"({', '.join(report['drifted_features'])})")

    champ = registry.get_by_alias(name, registry.CHAMPION)
    report["champion_version"] = int(champ.version) if champ else None
    if champ is None:
        report["reasons"].append("no champion model yet")
    else:
        model, _ = registry.load_champion(name)
        perf = holdout_performance(model, recent, horizon)
        report["holdout"] = perf
        with Session(engine) as s:
            ref = s.scalar(select(MLflowMetadata.validation_metrics)
                           .where(MLflowMetadata.model_name == name,
                                  MLflowMetadata.model_version == str(champ.version)))
        ref_mae = (ref or {}).get("test_mae")
        report["reference_mae"] = ref_mae
        if ref_mae and perf["mae"] > ref_mae * (1 + settings.mlops_mae_degradation):
            report["reasons"].append(f"accuracy dropped: MAE {perf['mae']:.2f} vs {ref_mae:.2f} at training time")
    report["forecast_log"] = forecast_log_performance(engine, window_days)

    biz = business_metrics(engine, window_days)
    report["business"] = biz
    if biz["stockout_rate"] > max(0.05, 2 * biz["baseline_stockout_rate"]):
        report["reasons"].append(f"stockout spike: {biz['stockout_rate']:.1%} of product-days "
                                 f"(was {biz['baseline_stockout_rate']:.1%})")
    report["retrain_recommended"] = bool(report["reasons"])

    if log_to_mlflow:
        configure(MONITOR_EXPERIMENT)
        with mlflow.start_run(run_name=f"monitor-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
            mlflow.set_tags({"pipeline": "monitoring", "retrain_recommended": str(report["retrain_recommended"])})
            flat = {"drift_share": report["drift_share"], **{f"business_{k}": v for k, v in biz.items()}}
            for source in ("holdout", "forecast_log"):
                for k, v in (report.get(source) or {}).items():
                    flat[f"{source}_{k}"] = v
            mlflow.log_metrics(safe_metrics(flat))
            mlflow.log_dict(drift.to_dict("records"), "drift_report.json")
            mlflow.log_dict({k: v for k, v in report.items()}, "monitoring_report.json")
    report["drift"] = drift.to_dict("records")
    return _clean(report)
