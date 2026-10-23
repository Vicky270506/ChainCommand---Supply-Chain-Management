"""Feature engineering for demand forecasting and stockout-risk models.

Implements the "Feature Engineering Layer" of the project report (section 6):
lag features, rolling statistics, seasonality indicators, holiday / festival and
promotion features, price and product features, historical stock availability
and supplier lead time.

Design rules
------------
* One row per product per day. The target is ``demand`` = units sold + units lost
  to stockouts (sales alone would under-count demand on stockout days).
* ``horizon`` h means: the forecast is made with information up to day t-h, so a
  model trained with horizon=7 can forecast the next 7 days directly. Every
  feature derived from demand or stock only uses data up to day t-h.
* Features that are known in advance (calendar, festivals, planned promotions,
  prices, supplier lead time) are used as they are for day t.
* The functions here are pure pandas, so they are easy to test without a database.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 21, 28, 364)          # days back from day t (364 = same weekday last year)
ROLL_MEAN_WINDOWS = (7, 14, 28)
ROLL_STD_WINDOWS = (7, 28)
STOCKOUT_WINDOW = 14
EVENT_CAP = 60                          # "no festival nearby" is encoded as 60 days

TARGET = "demand"
CATEGORICAL = ["product_id", "category_id"]

_AVAILABILITY = ["stock_at_origin", "stockout_days_14"]
_CALENDAR = ["day_of_week", "day_of_month", "week_of_year", "month", "quarter", "day_of_year", "is_weekend"]
_EVENTS = ["is_holiday", "is_festival", "days_to_next_festival", "days_since_last_festival", "in_festival_window"]
_PROMO = ["is_promo", "discount_pct"]
_STATIC = ["product_id", "category_id", "cost", "selling_price", "supplier_lead_time_days"]


def feature_columns(horizon: int = 1) -> list[str]:
    """Model input columns for a given forecast horizon."""
    lags = [f"lag_{d}" for d in LAGS if d >= horizon]
    means = [f"roll_mean_{w}" for w in ROLL_MEAN_WINDOWS]
    stds = [f"roll_std_{w}" for w in ROLL_STD_WINDOWS]
    return lags + means + stds + _AVAILABILITY + _CALENDAR + _EVENTS + _PROMO + _STATIC


def describe_features(horizon: int = 1) -> pd.DataFrame:
    """A table (feature, group, description) you can paste into the report."""
    rows = []
    for d in LAGS:
        if d >= horizon:
            note = " (same weekday last year)" if d == 364 else ""
            rows.append((f"lag_{d}", "Lagged demand", f"Demand {d} day(s) before the target day{note}"))
    for w in ROLL_MEAN_WINDOWS:
        rows.append((f"roll_mean_{w}", "Rolling statistics",
                     f"Mean demand over the {w} days ending {horizon} day(s) before"))
    for w in ROLL_STD_WINDOWS:
        rows.append((f"roll_std_{w}", "Rolling statistics",
                     f"Std. deviation of demand over the {w} days ending {horizon} day(s) before"))
    rows += [
        ("stock_at_origin", "Stock availability", "Closing stock on the forecast origin day"),
        ("stockout_days_14", "Stock availability",
         "Days with lost sales in the 14 days up to the forecast origin"),
        ("day_of_week", "Seasonality", "Monday=0 ... Sunday=6"),
        ("day_of_month", "Seasonality", "Day of the month"),
        ("week_of_year", "Seasonality", "ISO week number"),
        ("month", "Seasonality", "Month of the year"),
        ("quarter", "Seasonality", "Quarter of the year"),
        ("day_of_year", "Seasonality", "Day of the year"),
        ("is_weekend", "Seasonality", "1 on Saturday and Sunday"),
        ("is_holiday", "Holidays / festivals", "1 on a public holiday or festival"),
        ("is_festival", "Holidays / festivals", "1 on a major festival day"),
        ("days_to_next_festival", "Holidays / festivals",
         f"Days until the next festival (capped at {EVENT_CAP})"),
        ("days_since_last_festival", "Holidays / festivals",
         f"Days since the last festival (capped at {EVENT_CAP})"),
        ("in_festival_window", "Holidays / festivals",
         "1 during the 3 days before, the day of, and the day after a festival"),
        ("is_promo", "Promotions", "1 if the product is on promotion that day"),
        ("discount_pct", "Promotions", "Promotion discount in percent (0 if none)"),
        ("product_id", "Product", "Product identifier (categorical)"),
        ("category_id", "Product", "Product category (categorical)"),
        ("cost", "Price", "Unit purchase cost"),
        ("selling_price", "Price", "Unit selling price"),
        ("supplier_lead_time_days", "Supplier", "Mean lead time of the product's supplier"),
    ]
    return pd.DataFrame(rows, columns=["feature", "group", "description"])


def _product_grid(daily: pd.DataFrame, extend_days: int) -> pd.DataFrame:
    """One row per product from its first recorded day to the last day (+ future days)."""
    end = daily["date"].max() + pd.Timedelta(days=extend_days)
    starts = daily.groupby("product_id")["date"].min()
    frames = [pd.DataFrame({"product_id": pid, "date": pd.date_range(start, end)}) for pid, start in starts.items()]
    return pd.concat(frames, ignore_index=True)


def _calendar_frame(dates: pd.DatetimeIndex, events: pd.DataFrame) -> pd.DataFrame:
    cal = pd.DataFrame({"date": dates})
    cal["day_of_week"] = cal["date"].dt.dayofweek
    cal["day_of_month"] = cal["date"].dt.day
    cal["week_of_year"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["date"].dt.month
    cal["quarter"] = cal["date"].dt.quarter
    cal["day_of_year"] = cal["date"].dt.dayofyear
    cal["is_weekend"] = (cal["day_of_week"] >= 5).astype(int)

    if len(events):
        ev = events.copy()
        ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
        holiday_dates = set(ev["date"])
        festival_dates = ev.loc[ev["event_type"] == "festival", "date"]
    else:
        holiday_dates, festival_dates = set(), pd.Series([], dtype="datetime64[ns]")
    fest = np.sort(pd.to_datetime(festival_dates).drop_duplicates().to_numpy("datetime64[ns]"))
    day = np.timedelta64(1, "D")
    arr = cal["date"].to_numpy("datetime64[ns]")

    to_next = np.full(len(arr), EVENT_CAP, dtype=float)
    since_last = np.full(len(arr), EVENT_CAP, dtype=float)
    if len(fest):
        nxt = np.searchsorted(fest, arr, side="left")            # first festival on/after the day
        ok = nxt < len(fest)
        to_next[ok] = (fest[nxt[ok]] - arr[ok]) / day
        prv = np.searchsorted(fest, arr, side="right") - 1       # last festival on/before the day
        ok = prv >= 0
        since_last[ok] = (arr[ok] - fest[prv[ok]]) / day
    to_next = np.minimum(to_next, EVENT_CAP)
    since_last = np.minimum(since_last, EVENT_CAP)

    cal["is_holiday"] = cal["date"].isin(holiday_dates).astype(int)
    cal["is_festival"] = (to_next == 0).astype(int)
    cal["days_to_next_festival"] = to_next
    cal["days_since_last_festival"] = since_last
    cal["in_festival_window"] = ((to_next <= 3) | (since_last == 1)).astype(int)
    return cal


def _promo_frame(promos: pd.DataFrame) -> pd.DataFrame:
    """Expand (start, end) promotion ranges to one row per product per day."""
    if promos is None or len(promos) == 0:
        return pd.DataFrame({"product_id": pd.Series(dtype=int), "date": pd.Series(dtype="datetime64[ns]"),
                             "discount_pct": pd.Series(dtype=float)})
    rows = []
    for r in promos.itertuples(index=False):
        for d in pd.date_range(pd.to_datetime(r.start_date), pd.to_datetime(r.end_date)):
            rows.append((r.product_id, d, float(r.discount_pct)))
    expanded = pd.DataFrame(rows, columns=["product_id", "date", "discount_pct"])
    return expanded.groupby(["product_id", "date"], as_index=False)["discount_pct"].max()


def make_features(
    daily: pd.DataFrame,
    products: pd.DataFrame,
    events: pd.DataFrame,
    promos: Optional[pd.DataFrame] = None,
    horizon: int = 1,
    extend_days: int = 0,
) -> pd.DataFrame:
    """Build the modelling table.

    Args:
        daily: columns product_id, date, units_sold, lost_units, stock_close
            (days without a row count as zero sales; stock_close is carried forward).
        products: product_id, category_id, cost, selling_price, supplier_lead_time_days.
        events: date, event_type ("holiday" | "festival"), including future dates.
        promos: product_id, start_date, end_date, discount_pct (may be empty).
        horizon: forecast horizon in days (>= 1), see the module docstring.
        extend_days: also create this many future rows (demand = NaN) for forecasting.

    Returns:
        One row per product-day with ``demand``, ``is_future`` and ``feature_columns(horizon)``.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    if "stock_close" not in daily.columns:
        daily["stock_close"] = np.nan
    history_end = daily["date"].max()

    df = _product_grid(daily, extend_days).merge(
        daily[["product_id", "date", "units_sold", "lost_units", "stock_close"]],
        on=["product_id", "date"], how="left",
    )
    df = df.sort_values(["product_id", "date"], ignore_index=True)
    df["is_future"] = df["date"] > history_end
    future = df["is_future"]

    # Missing history rows are days without sales; the future is unknown.
    for col in ("units_sold", "lost_units"):
        df[col] = df[col].fillna(0.0).where(~future, np.nan)
    df["stock_close"] = df.groupby("product_id")["stock_close"].ffill().where(~future, np.nan)
    df[TARGET] = df["units_sold"] + df["lost_units"]
    stockout_flag = (df["lost_units"] > 0).astype(float).where(~future, np.nan)

    by_product = df.groupby("product_id")[TARGET]
    for d in LAGS:
        if d >= horizon:
            df[f"lag_{d}"] = by_product.shift(d)
    for w in ROLL_MEAN_WINDOWS:
        df[f"roll_mean_{w}"] = by_product.transform(lambda s, w=w: s.shift(horizon).rolling(w, min_periods=w).mean())
    for w in ROLL_STD_WINDOWS:
        df[f"roll_std_{w}"] = by_product.transform(lambda s, w=w: s.shift(horizon).rolling(w, min_periods=w).std())
    df["stock_at_origin"] = df.groupby("product_id")["stock_close"].shift(horizon)
    df["stockout_days_14"] = stockout_flag.groupby(df["product_id"]).transform(
        lambda s: s.shift(horizon).rolling(STOCKOUT_WINDOW, min_periods=STOCKOUT_WINDOW).sum()
    )

    cal = _calendar_frame(pd.DatetimeIndex(sorted(df["date"].unique())), events)
    df = df.merge(cal, on="date", how="left")

    promo = _promo_frame(promos)
    df = df.merge(promo, on=["product_id", "date"], how="left")
    df["discount_pct"] = df["discount_pct"].fillna(0.0)
    df["is_promo"] = (df["discount_pct"] > 0).astype(int)

    static = products[["product_id", "category_id", "cost", "selling_price", "supplier_lead_time_days"]]
    df = df.merge(static, on="product_id", how="left")

    front = ["product_id", "date", TARGET, "units_sold", "lost_units", "is_future"]
    cols = front + [c for c in feature_columns(horizon) if c not in front]
    return df[cols].reset_index(drop=True)


# ── time-based splitting (never shuffle time series) ─────────────────────

def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df[(~df["is_future"]) & df[TARGET].notna()]


def time_split(
    df: pd.DataFrame, valid_days: int = 60, test_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split on dates (the last days are the test set)."""
    labelled = _labelled(df)
    last = labelled["date"].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    train = labelled[labelled["date"] < valid_start]
    valid = labelled[(labelled["date"] >= valid_start) & (labelled["date"] < test_start)]
    test = labelled[labelled["date"] >= test_start]
    return train, valid, test


def rolling_origin_folds(
    df: pd.DataFrame, n_folds: int = 3, valid_days: int = 30
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Walk-forward validation: each fold trains on everything before its validation window."""
    labelled = _labelled(df)
    last = labelled["date"].max()
    for k in range(n_folds, 0, -1):
        valid_end = last - pd.Timedelta(days=(k - 1) * valid_days)
        valid_start = valid_end - pd.Timedelta(days=valid_days - 1)
        yield (labelled[labelled["date"] < valid_start],
               labelled[(labelled["date"] >= valid_start) & (labelled["date"] <= valid_end)])
