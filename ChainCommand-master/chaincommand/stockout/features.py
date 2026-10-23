"""Point-in-time features for the stockout classifier (all known at forecast origin)."""
import numpy as np
import pandas as pd

from sqlalchemy import select

from chaincommand.db.models import Purchase, PurchaseItem


def add_stockout_features(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    out = df.copy()
    mean7 = out["roll_mean_7"].replace(0, np.nan)
    mean28 = out["roll_mean_28"].replace(0, np.nan)

    # Expected demand over the window, from the recent average (not a model
    # forecast: those were trained on these days and would leak).
    out["expected_demand_h"] = out["roll_mean_7"] * horizon_days
    out["stock_minus_expected"] = out["stock_at_origin"] - out["expected_demand_h"]
    out["stock_cover_days"] = out["stock_at_origin"] / mean7
    out["demand_cv"] = out["roll_std_28"] / mean28
    out["lead_time_gap"] = out["supplier_lead_time_days"] - out["stock_cover_days"]
    return out


STOCKOUT_FEATURES = [
    "stock_at_origin", "stockout_days_14", "expected_demand_h",
    "stock_minus_expected", "stock_cover_days", "demand_cv",
    "supplier_lead_time_days", "lead_time_gap",
]

def load_orders(engine) -> pd.DataFrame:
    """One row per purchase line: product, order date, received date, quantity."""
    return pd.read_sql(
        select(PurchaseItem.product_id, Purchase.date, Purchase.received_date,
               PurchaseItem.quantity)
        .join(Purchase, Purchase.purchase_id == PurchaseItem.purchase_id),
        engine,
    )


def incoming_units(df: pd.DataFrame, orders: pd.DataFrame, origin_lag_days: int = 1) -> pd.Series:
    """Units on order but not yet received at the end of (row date - lag)."""
    placed = pd.DataFrame({"product_id": orders["product_id"], "day": orders["date"],
                           "delta": orders["quantity"]})
    got = orders.dropna(subset=["received_date"])
    received = pd.DataFrame({"product_id": got["product_id"], "day": got["received_date"],
                             "delta": -got["quantity"]})
    events = pd.concat([placed, received], ignore_index=True)
    events["day"] = pd.to_datetime(events["day"]).astype("datetime64[ns]")

    daily = events.groupby(["product_id", "day"], as_index=False)["delta"].sum()
    daily = daily.sort_values("day")
    daily["open_units"] = daily.groupby("product_id")["delta"].cumsum()

    left = df[["product_id", "date"]].copy()
    left["_row"] = np.arange(len(left))
    left["_origin"] = (pd.to_datetime(left["date"]) - pd.Timedelta(days=origin_lag_days)
                       ).astype("datetime64[ns]")
    left = left.sort_values("_origin")
    merged = pd.merge_asof(left, daily[["product_id", "day", "open_units"]],
                           left_on="_origin", right_on="day",
                           by="product_id", direction="backward")
    merged = merged.sort_values("_row")
    return pd.Series(merged["open_units"].fillna(0).to_numpy(), index=df.index)


def add_incoming_features(df: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["incoming_units"] = incoming_units(out, orders)
    out["stock_plus_incoming_minus_expected"] = (
        out["stock_at_origin"] + out["incoming_units"] - out["expected_demand_h"]
    )
    return out


STOCKOUT_FEATURES_FULL = STOCKOUT_FEATURES + [
    "incoming_units", "stock_plus_incoming_minus_expected",
]