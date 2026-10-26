"""One-shot setup for a fresh environment (used by the Docker ``bootstrap`` service).

Idempotent: creates tables, seeds demo data, creates demo users, trains the first
champion models and builds the first pending recommendations - only where missing.
"""
from __future__ import annotations

import os
import time

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ..config import settings
from ..db import seed as seed_module
from ..db.base import get_engine, init_db
from ..db.create_user import _DEMO, _add
from ..db.models import Forecast, RLRecommendation
from . import registry
from .retrain import refresh_plan
from .tracking import wait_for_server
from .training import train_forecaster, train_stockout


def wait_for_database(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception as exc:  # noqa: BLE001
            if time.time() > deadline:
                raise RuntimeError(f"database not reachable: {exc}") from exc
            time.sleep(2)


def bootstrap(products: int = 50, suppliers: int = 20, years: float = 3.0) -> None:
    wait_for_database()
    wait_for_server()
    engine = get_engine()
    init_db(engine)

    if seed_module.table_counts(engine)["products"] == 0:
        print("seeding demo data ...", flush=True)
        seed_module.seed_database(engine, products, suppliers, int(years * 365))
    seed_module.sync_sequences(engine)

    password = os.environ.get("CC_DEMO_PASSWORD")
    if password:
        with Session(engine) as s:
            for name, email, role in _DEMO:
                _add(s, name, email, role, password)
            s.commit()
    else:
        print("CC_DEMO_PASSWORD not set: no demo users created "
              "(create one with: python -m chaincommand.db.create_user)")

    horizon = settings.mlops_horizon
    if registry.get_by_alias(registry.forecaster_name(horizon), registry.CHAMPION) is None:
        print("training the first forecaster ...", flush=True)
        r = train_forecaster(engine, trigger="bootstrap")
        print(f"  {r['model']} v{r['version']}: {r['status']} (test MAE {r['metrics']['test_mae']:.2f})", flush=True)
    if registry.get_by_alias(registry.STOCKOUT_MODEL, registry.CHAMPION) is None:
        print("training the first stockout classifier ...", flush=True)
        r = train_stockout(engine, trigger="bootstrap")
        print(f"  {r['model']} v{r['version']}: {r['status']} (test F1 {r['metrics']['test_f1']:.3f})", flush=True)

    with Session(engine) as s:
        has_plan = (s.scalar(select(func.count()).select_from(Forecast)) or 0) > 0 and \
                   (s.scalar(select(func.count()).select_from(RLRecommendation)) or 0) > 0
    if not has_plan:
        print("creating the first forecasts and recommendations ...", flush=True)
        print(" ", refresh_plan(engine), flush=True)
    print("bootstrap complete", flush=True)
