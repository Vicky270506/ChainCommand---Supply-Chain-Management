"""Tests for feature engineering: hand-checked values, no future leakage, DB integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from chaincommand.db import make_engine
from chaincommand.db import models as m
from chaincommand.db.seed import seed_database
from chaincommand.features import (
    TARGET,
    describe_features,
    feature_columns,
    make_features,
    rolling_origin_folds,
    time_split,
)
from chaincommand.features.dataset import build_dataset

START = pd.Timestamp("2024-01-01")
N_DAYS = 400


def _panel(demand_override: dict[tuple[int, int], float] | None = None) -> pd.DataFrame:
    """Product 1: demand = 10 + t.  Product 2 starts 30 days later: demand = 100 + 2t."""
    rows = []
    for t in range(N_DAYS):
        rows.append((1, START + pd.Timedelta(days=t), 10.0 + t, 0.0, 50.0))
        if t >= 30:
            rows.append((2, START + pd.Timedelta(days=t), 100.0 + 2 * t, 0.0, 80.0))
    df = pd.DataFrame(rows, columns=["product_id", "date", "units_sold", "lost_units", "stock_close"])
    for (pid, t), value in (demand_override or {}).items():
        df.loc[(df.product_id == pid) & (df.date == START + pd.Timedelta(days=t)), "units_sold"] = value
    return df


PRODUCTS = pd.DataFrame({"product_id": [1, 2], "category_id": [1, 2], "cost": [2.0, 3.0],
                         "selling_price": [5.0, 6.0], "supplier_lead_time_days": [7.0, 9.0]})
EVENTS = pd.DataFrame({
    "date": pd.to_datetime(["2024-02-10", "2024-03-05", "2024-12-25"]),
    "event_type": ["festival", "holiday", "festival"],
})
PROMOS = pd.DataFrame({"product_id": [1], "start_date": ["2024-05-01"], "end_date": ["2024-05-03"],
                       "discount_pct": [20.0]})


def feats(horizon=1, extend=0, panel=None):
    return make_features(_panel() if panel is None else panel, PRODUCTS, EVENTS, PROMOS,
                         horizon=horizon, extend_days=extend)


def row(df, pid, t):
    return df[(df.product_id == pid) & (df.date == START + pd.Timedelta(days=t))].iloc[0]


# ── values are exactly what they should be ───────────────────
def test_lags_and_rolling_horizon_1():
    df = feats(1)
    r = row(df, 1, 100)                                   # demand(t) = 10 + t
    assert r[TARGET] == 110
    assert r["lag_1"] == 109 and r["lag_7"] == 103 and r["lag_28"] == 82
    assert np.isnan(r["lag_364"])                          # no data a year earlier yet
    assert r["roll_mean_7"] == pytest.approx(np.mean([10 + t for t in range(93, 100)]))
    assert r["roll_mean_28"] == pytest.approx(np.mean([10 + t for t in range(72, 100)]))
    assert r["roll_std_7"] == pytest.approx(np.std([10 + t for t in range(93, 100)], ddof=1))
    assert r["stock_at_origin"] == 50
    late = row(df, 1, 380)
    assert late["lag_364"] == 10 + 16                      # same weekday one year earlier


def test_horizon_7_only_uses_data_older_than_a_week():
    df = feats(7)
    assert "lag_1" not in df.columns
    assert [c for c in df.columns if c.startswith("lag_")] == ["lag_7", "lag_14", "lag_21", "lag_28", "lag_364"]
    r = row(df, 1, 100)
    assert r["lag_7"] == 10 + 93
    assert r["roll_mean_7"] == pytest.approx(np.mean([10 + t for t in range(87, 94)]))   # window ends at t-7


def test_new_product_starts_at_its_first_day():
    df = feats(1)
    p2 = df[df.product_id == 2]
    assert p2.date.min() == START + pd.Timedelta(days=30)
    assert len(p2) == N_DAYS - 30
    assert np.isnan(row(df, 2, 35)["roll_mean_28"])        # not enough history yet -> NaN, not a guess


# ── no leakage from the future ───────────────────────────────
@pytest.mark.parametrize("horizon", [1, 7])
def test_features_never_use_demand_from_the_target_period(horizon):
    t = 200
    base = feats(horizon)
    changed = feats(horizon, panel=_panel({(1, t): 99999.0}))
    cols = feature_columns(horizon)
    day = START + pd.Timedelta(days=t)
    b = base[(base.product_id == 1) & (base.date < day + pd.Timedelta(days=horizon))]
    c = changed[(changed.product_id == 1) & (changed.date < day + pd.Timedelta(days=horizon))]
    pd.testing.assert_frame_equal(b[cols].reset_index(drop=True), c[cols].reset_index(drop=True))
    after = changed[(changed.product_id == 1) & (changed.date == day + pd.Timedelta(days=horizon))]
    before = base[(base.product_id == 1) & (base.date == day + pd.Timedelta(days=horizon))]
    assert not after[cols].equals(before[cols])           # the change does show up once it is in the past


def test_future_rows_for_forecasting():
    df = feats(1, extend=7)
    fut = df[df.is_future]
    assert len(fut) == 14 and fut[TARGET].isna().all()
    first = fut[fut.product_id == 1].sort_values("date").iloc[0]
    assert first.date == START + pd.Timedelta(days=N_DAYS)
    assert first["lag_1"] == 10 + (N_DAYS - 1)             # yesterday's real demand
    assert first["day_of_week"] == first.date.dayofweek
    assert not df[~df.is_future][TARGET].isna().any()
    h7 = feats(7, extend=7)
    last_future = h7[(h7.product_id == 1)].sort_values("date").iloc[-1]
    assert last_future["lag_7"] == 10 + (N_DAYS + 6 - 7)   # day 7 of the forecast still has real data


# ── seasonal, festival and promotion features ────────────────
def test_festival_features():
    df = feats(1)
    d = lambda s: df[(df.product_id == 1) & (df.date == pd.Timestamp(s))].iloc[0]  # noqa: E731
    assert d("2024-02-10")["is_festival"] == 1 and d("2024-02-10")["days_to_next_festival"] == 0
    assert d("2024-02-07")["days_to_next_festival"] == 3 and d("2024-02-07")["in_festival_window"] == 1
    assert d("2024-02-06")["in_festival_window"] == 0
    assert d("2024-02-11")["days_since_last_festival"] == 1 and d("2024-02-11")["in_festival_window"] == 1
    assert d("2024-02-12")["in_festival_window"] == 0
    assert d("2024-01-01")["days_to_next_festival"] == 40 and d("2024-01-01")["days_since_last_festival"] == 60
    assert d("2024-03-05")["is_holiday"] == 1 and d("2024-03-05")["is_festival"] == 0


def test_calendar_and_promo_and_static_features():
    df = feats(1)
    r = row(df, 1, 0)                                       # 2024-01-01 is a Monday
    assert r["day_of_week"] == 0 and r["month"] == 1 and r["quarter"] == 1 and r["is_weekend"] == 0
    assert row(df, 1, 5)["is_weekend"] == 1
    promo_days = df[(df.product_id == 1) & (df.is_promo == 1)].date
    assert list(promo_days) == list(pd.date_range("2024-05-01", "2024-05-03"))
    assert (df[df.is_promo == 1].discount_pct == 20.0).all()
    assert (df[df.product_id == 2].is_promo == 0).all()
    assert (df[df.product_id == 2].supplier_lead_time_days == 9.0).all()


def test_stockout_features_use_lost_units():
    panel = _panel()
    mask = (panel.product_id == 1) & (panel.date.between(START + pd.Timedelta(days=50), START + pd.Timedelta(days=52)))
    panel.loc[mask, "lost_units"] = 5.0
    df = feats(1, panel=panel)
    assert row(df, 1, 50)[TARGET] == 10 + 50 + 5            # demand counts the lost units
    assert row(df, 1, 54)["stockout_days_14"] == 3
    assert row(df, 1, 51)["stockout_days_14"] == 1          # day 50 only (day 51 itself is not yet known)


def test_all_declared_features_exist_and_are_documented():
    for h in (1, 7):
        df = feats(h)
        assert set(feature_columns(h)) <= set(df.columns)
        assert set(describe_features(h)["feature"]) == set(feature_columns(h))


def test_invalid_horizon():
    with pytest.raises(ValueError):
        feats(0)


# ── splitting ────────────────────────────────────────────────
def test_time_split_is_chronological_and_disjoint():
    df = feats(1)
    train, valid, test = time_split(df, valid_days=30, test_days=30)
    assert train.date.max() < valid.date.min() <= valid.date.max() < test.date.min()
    assert test.date.max() == START + pd.Timedelta(days=N_DAYS - 1)
    assert valid.date.nunique() == 30 and test.date.nunique() == 30


def test_rolling_origin_folds_move_forward():
    df = feats(1)
    folds = list(rolling_origin_folds(df, n_folds=3, valid_days=20))
    assert len(folds) == 3
    ends = [v.date.max() for _, v in folds]
    assert ends == sorted(ends) and len(set(ends)) == 3
    for train, valid in folds:
        assert train.date.max() < valid.date.min()


# ── against the real database tables ─────────────────────────
@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    engine = make_engine(f"sqlite:///{tmp_path_factory.mktemp('feat') / 'f.db'}")
    m.Base.metadata.create_all(engine)
    seed_database(engine, n_products=6, n_suppliers=4, days=500, seed=11)
    yield engine
    engine.dispose()


def test_dataset_matches_database_totals(seeded):
    df = build_dataset(seeded, horizon=1)
    with seeded.connect() as conn:
        sold = conn.execute(text("select sum(quantity) from sale_items")).scalar_one()
        lost = conn.execute(text("select sum(quantity) from inventory_transactions "
                                 "where transaction_type='lost_sale'")).scalar_one()
    assert df[TARGET].sum() == sold + lost
    assert df.units_sold.sum() == sold and df.lost_units.sum() == lost
    assert not df.is_future.any() and df[TARGET].notna().all()
    assert df.groupby("product_id").date.nunique().nunique() == 1       # every product has the same history
    assert df.groupby("product_id").date.apply(lambda s: s.is_monotonic_increasing).all()


def test_dataset_closing_stock_equals_ledger(seeded):
    # the stock the features see on the last day must equal products.current_stock
    from chaincommand.features.dataset import _daily_panel
    panel = _daily_panel(seeded)
    last = panel.sort_values("date").groupby("product_id").tail(1).set_index("product_id")["stock_close"]
    with seeded.connect() as conn:
        stock = dict(conn.execute(text("select product_id, current_stock from products")).all())
    assert {int(k): int(v) for k, v in last.items()} == stock


def test_dataset_with_future_rows_has_calendar_features(seeded):
    df = build_dataset(seeded, horizon=7, extend_days=14)
    fut = df[df.is_future]
    assert len(fut) == 6 * 14
    assert fut[TARGET].isna().all()
    assert fut["day_of_week"].notna().all() and fut["days_to_next_festival"].notna().all()
    # A horizon-7 model forecasts 7 days ahead: lag_7 is real data for the first 7 future days only.
    # For days 8-14 the day it would look at is itself in the future, so it is (correctly) missing.
    ordered = fut.sort_values(["product_id", "date"])
    first_week = ordered.groupby("product_id").head(7)
    second_week = ordered.groupby("product_id").tail(7)
    assert first_week["lag_7"].notna().all()
    assert second_week["lag_7"].isna().all()
