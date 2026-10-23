"""Forecast-based replenishment rule + order constraints -> pending recommendations."""
import argparse
import math

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from chaincommand.db.models import (
    ApprovalStatus, Forecast, InventoryTransaction, Product, Purchase, PurchaseItem,
    PurchaseStatus, RLRecommendation, Supplier, utcnow,
)
from chaincommand.features.dataset import get_engine

POLICY_VERSION = "rule-forecast-v1"
Z_SERVICE = 1.645   # 95% service level
Z_80 = 1.2816       # converts an 80% interval width to a standard deviation


def constrain(raw, moq, pack, room):
    """Apply minimum order quantity, pack size and storage room. Returns (qty, action)."""
    if raw <= 0:
        return 0, "no_order"
    qty = math.ceil(max(raw, moq) / pack) * pack
    if qty > room:
        qty = math.floor(max(room, 0) / pack) * pack
        if qty < moq:
            return 0, "blocked_storage"
        return qty, "order_capped_storage"
    return qty, "order"


def latest_forecasts(engine):
    with Session(engine) as s:
        version = s.scalar(select(Forecast.model_version).order_by(Forecast.generated_at.desc()).limit(1))
    fc = pd.read_sql(
        select(Forecast.product_id, Forecast.predicted_demand, Forecast.lower, Forecast.upper)
        .where(Forecast.model_version == version), engine)
    return version, fc


def build_plan(engine):
    version, fc = latest_forecasts(engine)
    g = fc.groupby("product_id").agg(daily=("predicted_demand", "mean"),
                                     lo=("lower", "mean"), hi=("upper", "mean")).reset_index()
    g["sigma"] = (g["hi"] - g["lo"]) / (2 * Z_80)
    prod = pd.read_sql(
        select(Product.product_id, Product.cost, Product.min_order_qty, Product.pack_size,
               Product.max_stock, Product.current_stock, Supplier.lead_time_mean_days)
        .join(Supplier, Supplier.supplier_id == Product.supplier_id), engine)
    inc = pd.read_sql(
        select(PurchaseItem.product_id, func.sum(PurchaseItem.quantity).label("incoming"))
        .join(Purchase, Purchase.purchase_id == PurchaseItem.purchase_id)
        .where(Purchase.status == PurchaseStatus.ORDERED)
        .group_by(PurchaseItem.product_id), engine)
    df = g.merge(prod, on="product_id").merge(inc, on="product_id", how="left")
    df["incoming"] = df["incoming"].fillna(0)
    df["L"] = np.ceil(df["lead_time_mean_days"])
    df["forecast_L"] = df["daily"] * df["L"]
    df["safety"] = Z_SERVICE * df["sigma"] * np.sqrt(df["L"])
    df["raw"] = df["forecast_L"] + df["safety"] - df["current_stock"] - df["incoming"]
    return version, df


def apply_constraints(df, budget=None):
    rows = []
    for r in df.itertuples():
        room = r.max_stock - r.current_stock - r.incoming
        qty, action = constrain(r.raw, max(r.min_order_qty, 1), max(r.pack_size, 1), room)
        rows.append((r.product_id, qty, action,
                     (r.current_stock + r.incoming) / max(r.daily, 1e-9)))
    plan = pd.DataFrame(rows, columns=["product_id", "qty", "action", "cover_days"])
    if budget is not None:
        cost = df.set_index("product_id")["cost"]
        spent = 0.0
        for i in plan.sort_values("cover_days").index:   # most urgent products first
            value = plan.at[i, "qty"] * cost[plan.at[i, "product_id"]]
            if plan.at[i, "qty"] > 0 and spent + value > budget:
                plan.at[i, "qty"], plan.at[i, "action"] = 0, "blocked_budget"
            else:
                spent += value
    return df.merge(plan, on="product_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    args = ap.parse_args()
    engine = get_engine()
    version, df = build_plan(engine)
    df = apply_constraints(df, args.budget)
    with Session(engine) as s:
        data_ts = s.scalar(select(func.max(InventoryTransaction.timestamp)))
        now = utcnow()
        s.add_all([
            RLRecommendation(
                product_id=int(r.product_id), recommended_quantity=int(r.qty), action=r.action,
                policy_version=POLICY_VERSION, generated_at=now, input_data_timestamp=data_ts,
                forecast_model_version=version, forecast_demand=float(r.forecast_L),
                rule_based_quantity=int(round(max(r.raw, 0))),
                explanation=(f"Forecast {r.daily:.1f}/day x {r.L:.0f}-day lead time + safety {r.safety:.0f} "
                             f"= {r.forecast_L + r.safety:.0f}; stock {r.current_stock} + incoming "
                             f"{r.incoming:.0f}; shortfall {r.raw:.0f}; MOQ {r.min_order_qty}, pack "
                             f"{r.pack_size}, max stock {r.max_stock} -> {r.action}, order {r.qty}."),
                approval_status=ApprovalStatus.PENDING)
            for r in df.itertuples()])
        s.commit()
    print(f"saved {len(df)} pending recommendations using forecasts {version}")
    print(df["action"].value_counts().to_string())
    print(df[["product_id", "current_stock", "incoming", "forecast_L", "raw", "qty", "action"]]
          .head(8).to_string(index=False, float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main()