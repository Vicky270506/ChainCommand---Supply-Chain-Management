"""Training pipelines. Every run is tracked in MLflow, registered, compared with the
current champion on the same hold-out window, and promoted only if it is better.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

import mlflow

from ..config import settings
from ..db.base import get_engine, init_db
from ..features.dataset import build_dataset
from ..features.engineering import TARGET, feature_columns, time_split
from ..forecasting.baselines import LinearBaseline, SeasonalNaive
from ..forecasting.evaluate import linear_columns, score
from ..forecasting.gbm import GBMForecaster
from ..forecasting.intervals import IntervalForecaster
from ..forecasting.metrics import coverage
from ..stockout.classifier import StockoutClassifier, build_stockout_table, report
from ..stockout.features import STOCKOUT_FEATURES_FULL
from ..stockout.labels import LABEL
from . import registry
from .pyfunc_models import ForecastModel, StockoutModel
from .tracking import configure, fingerprint, log_dataset, safe_metrics

_PIP = ["mlflow", "lightgbm", "scikit-learn", "pandas", "numpy", "joblib"]
LEVEL = 0.8


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _db_source(engine, table: str) -> str:
    return f"{engine.url.render_as_string(hide_password=True)}#{table}"


def _log_frame(df: pd.DataFrame, file: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / file
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def train_forecaster(engine=None, horizon: Optional[int] = None, trigger: str = "manual",
                     df: Optional[pd.DataFrame] = None) -> dict:
    """Train baselines + LightGBM (point and 80 % interval), track, register, promote if better."""
    engine = engine or get_engine()
    init_db(engine)
    horizon = horizon or settings.mlops_horizon
    name = registry.forecaster_name(horizon)
    configure()

    df = build_dataset(engine, horizon=horizon) if df is None else df
    train, valid, test = time_split(df, settings.mlops_valid_days, settings.mlops_test_days)
    cols = feature_columns(horizon)

    with mlflow.start_run(run_name=f"forecaster-h{horizon}-{_stamp()}") as run:
        mlflow.set_tags({"pipeline": "forecaster", "trigger": trigger, "horizon": str(horizon)})
        mlflow.log_params({
            "horizon": horizon, "n_features": len(cols), "interval_level": LEVEL,
            "valid_days": settings.mlops_valid_days, "test_days": settings.mlops_test_days,
            "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test),
            "train_end": str(train["date"].max().date()), "test_start": str(test["date"].min().date()),
        })
        log_dataset(train, "forecast_train", _db_source(engine, "sales+inventory_transactions"), "training")
        log_dataset(test, "forecast_test", _db_source(engine, "sales+inventory_transactions"), "evaluation")
        mlflow.log_dict(fingerprint(df), "data_fingerprint.json")

        # Baselines (report: linear baseline + seasonal naive for comparison)
        metrics: dict[str, float] = {}
        for bname, base in (("seasonal_naive", SeasonalNaive()), ("linear", LinearBaseline(linear_columns(horizon)))):
            base.fit(train, train[TARGET])
            for split, part in (("valid", valid), ("test", test)):
                for k, v in score(part[TARGET], base.predict(part)).items():
                    metrics[f"baseline_{bname}_{split}_{k.lower()}"] = v

        # LightGBM point + interval models (valid is used only to stop adding trees)
        point = GBMForecaster(cols).fit(train, train[TARGET], valid, valid[TARGET])
        interval = IntervalForecaster(cols, level=LEVEL).fit(train, train[TARGET], valid, valid[TARGET])
        mlflow.log_params({f"lgbm_{k}": v for k, v in point.params.items() if k != "verbosity"})
        mlflow.log_param("lgbm_best_iteration", point.model.best_iteration)
        for split, part in (("valid", valid), ("test", test)):
            pred = point.predict(part)
            for k, v in score(part[TARGET], pred).items():
                metrics[f"{split}_{k.lower()}"] = v
            lo, hi = interval.predict_interval(part)
            metrics[f"{split}_coverage"] = coverage(part[TARGET], lo, hi)
            metrics[f"{split}_interval_width"] = float(np.mean(hi - lo))
        metrics["test_mae_gain_vs_seasonal_naive"] = (
            1 - metrics["test_mae"] / metrics["baseline_seasonal_naive_test_mae"])
        mlflow.log_metrics(safe_metrics(metrics))
        _log_frame(point.importance(), "feature_importance.csv")

        # Package as an MLflow model with signature + input example
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "forecaster.joblib"
            joblib.dump({"point": point, "interval": interval, "horizon": horizon}, bundle)
            sample = test[cols].head(200)
            out = ForecastModel()
            out.bundle = {"point": point, "interval": interval}
            sig = infer_signature(sample, out.predict(None, sample))
            info = mlflow.pyfunc.log_model(
                name="model", python_model=ForecastModel(), artifacts={"bundle": str(bundle)},
                signature=sig, input_example=sample.head(3), pip_requirements=_PIP,
                registered_model_name=name,
            )
        version = int(info.registered_model_version)

        # Champion vs challenger on the SAME hold-out window
        champion_mae: Optional[float] = None
        champ = registry.get_by_alias(name, registry.CHAMPION)
        if champ is not None:
            champ_model = mlflow.pyfunc.load_model(f"models:/{name}@{registry.CHAMPION}")
            champ_pred = champ_model.predict(test[cols])["forecast"].to_numpy()
            champion_mae = score(test[TARGET], champ_pred)["MAE"]
            mlflow.log_metric("champion_test_mae_same_window", champion_mae)
        decision = registry.decide(metrics["test_mae"], champion_mae, higher_is_better=False,
                                   min_improvement=settings.mlops_min_improvement)
        status = registry.promote_or_reject(
            engine, name, version, run.info.run_id, metrics, decision,
            tags={"metric": "test_mae", "trigger": trigger, "horizon": horizon})
        mlflow.set_tags({"decision": status, "model_version": str(version)})

    return {"model": name, "version": version, "run_id": run.info.run_id, "status": status,
            "reason": decision.reason, "metrics": metrics}


def train_stockout(engine=None, trigger: str = "manual", horizon_days: int = 7) -> dict:
    """Train the stockout classifier, track it and compare it with the champion on the same window."""
    engine = engine or get_engine()
    init_db(engine)
    name = registry.STOCKOUT_MODEL
    configure()

    df = build_stockout_table(engine, horizon_days)
    train, valid, test = (p.dropna(subset=[LABEL])
                          for p in time_split(df, settings.mlops_valid_days, settings.mlops_test_days))
    cols = list(STOCKOUT_FEATURES_FULL)

    with mlflow.start_run(run_name=f"stockout-{_stamp()}") as run:
        mlflow.set_tags({"pipeline": "stockout", "trigger": trigger})
        mlflow.log_params({
            "label_horizon_days": horizon_days, "n_features": len(cols),
            "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test),
            "positive_rate_train": round(float(train[LABEL].mean()), 4),
        })
        log_dataset(train, "stockout_train", _db_source(engine, "stockout_table"), "training")
        log_dataset(test, "stockout_test", _db_source(engine, "stockout_table"), "evaluation")

        clf = StockoutClassifier().fit(train, train[LABEL], valid, valid[LABEL])
        mlflow.log_params({"lgbm_best_iteration": clf.model.best_iteration, "threshold": round(clf.threshold, 4)})
        metrics: dict[str, float] = {}
        for split, part in (("valid", valid), ("test", test)):
            prob = clf.predict_proba(part)
            for k, v in report(part[LABEL], clf.predict(part), prob).items():
                metrics[f"{split}_{k.lower()}"] = v
        rule = (test["lead_time_gap"] > 0).astype(int)      # simple rule baseline
        for k, v in report(test[LABEL], rule).items():
            metrics[f"baseline_rule_test_{k.lower()}"] = v
        mlflow.log_metrics(safe_metrics(metrics))
        imp = pd.DataFrame({"feature": clf.feature_cols,
                            "gain": clf.model.feature_importance(importance_type="gain")})
        _log_frame(imp.sort_values("gain", ascending=False), "feature_importance.csv")

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "stockout.joblib"
            joblib.dump({"classifier": clf}, bundle)
            sample = test[cols].head(200)
            tmp_model = StockoutModel()
            tmp_model.bundle = {"classifier": clf}
            sig = infer_signature(sample, tmp_model.predict(None, sample))
            info = mlflow.pyfunc.log_model(
                name="model", python_model=StockoutModel(), artifacts={"bundle": str(bundle)},
                signature=sig, input_example=sample.head(3), pip_requirements=_PIP,
                registered_model_name=name,
            )
        version = int(info.registered_model_version)

        champion_f1: Optional[float] = None
        if registry.get_by_alias(name, registry.CHAMPION) is not None:
            champ_model = mlflow.pyfunc.load_model(f"models:/{name}@{registry.CHAMPION}")
            flag = champ_model.predict(test[cols])["stockout_flag"].to_numpy()
            champion_f1 = report(test[LABEL], flag)["F1"]
            mlflow.log_metric("champion_test_f1_same_window", champion_f1)
        decision = registry.decide(metrics["test_f1"], champion_f1, higher_is_better=True,
                                   min_improvement=settings.mlops_min_improvement)
        status = registry.promote_or_reject(
            engine, name, version, run.info.run_id, metrics, decision,
            tags={"metric": "test_f1", "trigger": trigger})
        mlflow.set_tags({"decision": status, "model_version": str(version)})

    return {"model": name, "version": version, "run_id": run.info.run_id, "status": status,
            "reason": decision.reason, "metrics": metrics}
