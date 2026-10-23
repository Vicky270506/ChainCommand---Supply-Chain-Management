from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from chaincommand.db.models import Base, Forecast
from chaincommand.forecasting.predict import save_forecasts


def _forecasts():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-10-22", "2026-10-23"]),
        "product_id": [1, 1],
        "forecast": [13.7, 13.6],
        "lower": [10.3, 10.0],
        "upper": [19.0, 19.2],
    })


def test_two_runs_are_both_kept():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _no_foreign_keys(dbapi_conn, _record):
        # This test is about append-only history, not product integrity,
        # so skip the FK check instead of seeding products and suppliers.
        dbapi_conn.execute("PRAGMA foreign_keys=OFF")

    Base.metadata.create_all(engine)

    v1 = save_forecasts(engine, _forecasts(), 7, generated_at=datetime(2026, 10, 23, 10, 0, 0))
    v2 = save_forecasts(engine, _forecasts(), 7, generated_at=datetime(2026, 10, 23, 11, 0, 0))

    assert v1 != v2
    assert "h7" in v1
    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(Forecast)) == 4
        assert s.scalar(select(func.count(func.distinct(Forecast.model_version)))) == 2
        row = s.scalars(select(Forecast).where(Forecast.model_version == v1)).first()
        assert row.lower <= row.predicted_demand <= row.upper