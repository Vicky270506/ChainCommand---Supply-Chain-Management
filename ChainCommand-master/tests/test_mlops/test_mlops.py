"""MLflow layer: promotion rule, drift maths, champion/challenger flow, scheduler, API access."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from chaincommand.api.routes.auth import router as auth_router
from chaincommand.api.routes.mlops import router as mlops_router
from chaincommand.config import settings
from chaincommand.db import get_db, make_engine
from chaincommand.db import models as m
from chaincommand.db.seed import seed_database
from chaincommand.mlops import registry, retrain
from chaincommand.mlops.monitoring import drift_report, psi, run_monitoring
from chaincommand.mlops.registry import decide
from chaincommand.mlops.training import train_forecaster, train_stockout
from chaincommand.security import hash_password


# ── pure logic ───────────────────────────────────────────────
def test_decide_lower_is_better():
    assert decide(9.0, None).promote
    assert decide(9.0, 10.0).promote
    assert not decide(10.0, 10.0).promote                    # a tie keeps the champion
    assert not decide(9.9, 10.0, min_improvement=0.05).promote
    assert decide(9.4, 10.0, min_improvement=0.05).promote


def test_decide_higher_is_better():
    assert decide(0.8, 0.7, higher_is_better=True).promote
    assert not decide(0.7, 0.7, higher_is_better=True).promote


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 2000))
    assert psi(a, pd.Series(rng.normal(0, 1, 2000))) < 0.1
    assert psi(a, pd.Series(rng.normal(2, 1, 2000))) > 0.2


def test_drift_report_flags_only_shifted_feature():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 1500), "y": rng.normal(5, 1, 1500)})
    cur = pd.DataFrame({"x": rng.normal(3, 1, 500), "y": rng.normal(5, 1, 500)})
    rep = drift_report(ref, cur, features=["x", "y"]).set_index("feature")
    assert rep.loc["x", "drifted"] and not rep.loc["y", "drifted"]


# ── champion / challenger on a small database ────────────────
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mlops")
    old = (settings.mlflow_tracking_uri, settings.mlops_state_file)
    settings.mlflow_tracking_uri = f"sqlite:///{tmp / 'mlflow.db'}"
    settings.mlops_state_file = str(tmp / "state.json")
    engine = make_engine(f"sqlite:///{tmp / 'app.db'}")
    m.Base.metadata.create_all(engine)
    seed_database(engine, n_products=8, n_suppliers=4, days=420, seed=5)
    yield engine
    engine.dispose()
    settings.mlflow_tracking_uri, settings.mlops_state_file = old


def test_first_model_becomes_champion_and_second_identical_is_rejected(env):
    first = train_forecaster(env, horizon=7, trigger="test")
    assert first["status"] == "champion" and first["version"] == 1
    second = train_forecaster(env, horizon=7, trigger="test")
    assert second["status"] == "rejected"
    name = registry.forecaster_name(7)
    assert int(registry.get_by_alias(name, registry.CHAMPION).version) == 1
    assert registry.get_by_alias(name, registry.CHALLENGER) is None
    with Session(env) as s:
        rows = s.scalars(select(m.MLflowMetadata).where(m.MLflowMetadata.model_name == name)
                         .order_by(m.MLflowMetadata.id)).all()
    assert [r.deployment_status for r in rows] == ["champion", "rejected"]
    assert rows[0].validation_metrics["test_mae"] > 0


def test_better_challenger_replaces_champion(env, monkeypatch):
    monkeypatch.setattr(settings, "mlops_min_improvement", -1.0)    # accept anything -> forces a promotion
    third = train_forecaster(env, horizon=7, trigger="test")
    assert third["status"] == "champion"
    name = registry.forecaster_name(7)
    assert int(registry.get_by_alias(name, registry.CHAMPION).version) == third["version"]
    assert registry.get_by_alias(name, registry.PREVIOUS) is not None
    with Session(env) as s:
        champions = s.scalar(select(func.count()).select_from(m.MLflowMetadata)
                             .where(m.MLflowMetadata.model_name == name,
                                    m.MLflowMetadata.deployment_status == "champion"))
    assert champions == 1                                           # the old one was archived


def test_stockout_model_is_registered(env):
    r = train_stockout(env, trigger="test")
    assert r["status"] == "champion"
    model, _ = registry.load_champion(registry.STOCKOUT_MODEL)
    assert model.metadata.signature is not None


def test_champion_forecasts_and_plan_are_traceable(env, monkeypatch):
    monkeypatch.setattr(settings, "mlops_horizon", 7)
    out = retrain.refresh_plan(env)
    assert out["forecast_rows"] == 8 * 7 and out["recommendations"] == 8
    with Session(env) as s:
        rec = s.scalars(select(m.RLRecommendation)).first()
        assert rec.forecast_model_version.startswith("mlflow-v")
        assert rec.approval_status == m.ApprovalStatus.PENDING


def test_monitoring_report(env, monkeypatch):
    monkeypatch.setattr(settings, "mlops_horizon", 7)
    rep = run_monitoring(env, log_to_mlflow=True)
    assert rep["champion_version"] is not None
    assert {"mae", "rmse", "smape", "bias", "coverage"} <= set(rep["holdout"])
    assert 0 <= rep["business"]["fill_rate"] <= 1
    assert isinstance(rep["retrain_recommended"], bool)


# ── scheduler ────────────────────────────────────────────────
def test_scheduler_runs_jobs_when_due(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "mlops_state_file", str(tmp_path / "s.json"))
    calls = []
    monkeypatch.setattr(retrain, "retrain", lambda engine=None, trigger="manual", full=False:
                        calls.append((trigger, full)) or {})
    report = {"retrain_recommended": False, "reasons": [], "drift_share": 0.0}
    monkeypatch.setattr(retrain, "run_monitoring", lambda engine=None: dict(report))
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    retrain.tick(object(), t0)                                      # first start: full retrain
    assert calls == [("schedule-monthly", True)]
    retrain.tick(object(), t0 + timedelta(hours=1))                 # nothing due
    assert len(calls) == 1
    retrain.tick(object(), t0 + timedelta(days=1, hours=1))         # daily monitoring, healthy
    assert len(calls) == 1
    report.update(retrain_recommended=True, reasons=["data drift"], drift_share=0.5)
    retrain.tick(object(), t0 + timedelta(days=2, hours=2))         # drift -> event-driven retrain
    assert calls[-1] == ("drift", False)
    retrain.tick(object(), t0 + timedelta(days=40))                 # a month later: full retrain
    assert calls[-1] == ("schedule-monthly", True)


# ── API access control ───────────────────────────────────────
def test_mlops_api_roles(tmp_path, monkeypatch):
    engine = make_engine(f"sqlite:///{tmp_path / 'api.db'}")
    m.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with Session(engine) as s:
        for role in (m.Role.ADMIN, m.Role.MANAGER, m.Role.STAFF):
            s.add(m.User(name=role, email=f"{role}@t.com", role=role, password_hash=hash_password("Password123")))
        s.add(m.MLflowMetadata(model_name="x", model_version="1", run_id="r", validation_metrics={"test_mae": 1.0},
                               deployment_status="champion"))
        s.commit()
    app = FastAPI()
    app.include_router(auth_router, prefix="/api")
    app.include_router(mlops_router, prefix="/api")

    def override():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override
    monkeypatch.setattr("chaincommand.api.routes.mlops._retrain_job", lambda full: None)
    client = TestClient(app)

    def hdr(who):
        r = client.post("/api/auth/login", json={"email": f"{who}@t.com", "password": "Password123"})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    assert client.get("/api/mlops/models").status_code == 401
    assert client.get("/api/mlops/models", headers=hdr("staff")).status_code == 403
    r = client.get("/api/mlops/models", headers=hdr("manager"))
    assert r.status_code == 200 and r.json()[0]["metrics"]["test_mae"] == 1.0
    assert client.post("/api/mlops/retrain", headers=hdr("manager")).status_code == 403
    assert client.post("/api/mlops/retrain", headers=hdr("admin")).status_code == 202
    engine.dispose()


def test_registry_uses_configured_tracking_uri(tmp_path, monkeypatch):
    """A fresh process must talk to the configured server even if nothing called configure() first."""
    import mlflow
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"sqlite:///{tmp_path / 'other.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'wrong.db'}")
    assert registry.get_by_alias("nothing-here", registry.CHAMPION) is None
    assert mlflow.get_tracking_uri() == settings.mlflow_tracking_uri


def test_psi_is_finite_when_a_constant_feature_changes():
    ref = pd.Series(np.zeros(500))
    cur = pd.Series(np.full(500, 20.0))
    assert np.isfinite(psi(ref, cur)) and psi(ref, cur) > 0.2
