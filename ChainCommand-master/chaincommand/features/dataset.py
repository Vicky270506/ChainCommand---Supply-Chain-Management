"""Load the modelling table from the database.

    python -m chaincommand.features.dataset --horizon 1
    python -m chaincommand.features.dataset --horizon 7 --extend 14 --out features_h7.parquet
"""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd
from sqlalchemy import Date, Engine, func, select

from ..db.base import get_engine
from ..db.models import (
    CalendarEvent,
    InventoryTransaction,
    Product,
    Promotion,
    Sale,
    SaleItem,
    Supplier,
    TxType,
)
from .engineering import describe_features, feature_columns, make_features


def _daily_panel(engine: Engine) -> pd.DataFrame:
    """Per product-day: units sold, lost units and closing stock (from the ledger)."""
    sold = pd.read_sql(
        select(Sale.date.label("date"), SaleItem.product_id, func.sum(SaleItem.quantity).label("units_sold"))
        .join(Sale, Sale.sale_id == SaleItem.sale_id)
        .group_by(Sale.date, SaleItem.product_id),
        engine,
    )
    day = func.date(InventoryTransaction.timestamp, type_=Date)   # DATE(ts) works on SQLite and PostgreSQL
    lost = pd.read_sql(
        select(day.label("date"), InventoryTransaction.product_id,
               func.sum(InventoryTransaction.quantity).label("lost_units"))
        .where(InventoryTransaction.transaction_type == TxType.LOST_SALE)
        .group_by(day, InventoryTransaction.product_id),
        engine,
    )
    net = pd.read_sql(
        select(day.label("date"), InventoryTransaction.product_id,
               func.sum(InventoryTransaction.quantity).label("net_change"))
        .where(InventoryTransaction.transaction_type != TxType.LOST_SALE)
        .group_by(day, InventoryTransaction.product_id),
        engine,
    )
    if net.empty:
        raise ValueError("The database has no inventory transactions. Run: python -m chaincommand.db.seed --reset")
    for frame in (sold, lost, net):
        frame["date"] = pd.to_datetime(frame["date"])

    start_by_product = net.groupby("product_id")["date"].min()
    end = max(f["date"].max() for f in (sold, lost, net) if not f.empty)
    grid = pd.concat(
        [pd.DataFrame({"product_id": pid, "date": pd.date_range(start, end)})
         for pid, start in start_by_product.items()],
        ignore_index=True,
    )
    panel = (grid.merge(sold, on=["product_id", "date"], how="left")
                 .merge(lost, on=["product_id", "date"], how="left")
                 .merge(net, on=["product_id", "date"], how="left"))
    panel[["units_sold", "lost_units", "net_change"]] = panel[["units_sold", "lost_units", "net_change"]].fillna(0)
    panel = panel.sort_values(["product_id", "date"], ignore_index=True)
    panel["stock_close"] = panel.groupby("product_id")["net_change"].cumsum()
    return panel.drop(columns="net_change")


def build_dataset(engine: Optional[Engine] = None, horizon: int = 1, extend_days: int = 0) -> pd.DataFrame:
    """Modelling table for all products. ``extend_days`` adds future rows for forecasting."""
    engine = engine or get_engine()
    daily = _daily_panel(engine)
    products = pd.read_sql(
        select(Product.product_id, Product.category_id, Product.cost, Product.selling_price,
               Supplier.lead_time_mean_days.label("supplier_lead_time_days"))
        .join(Supplier, Supplier.supplier_id == Product.supplier_id),
        engine,
    )
    events = pd.read_sql(select(CalendarEvent.date, CalendarEvent.event_type), engine)
    promos = pd.read_sql(
        select(Promotion.product_id, Promotion.start_date, Promotion.end_date, Promotion.discount_pct), engine)
    return make_features(daily, products, events, promos, horizon=horizon, extend_days=extend_days)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the forecasting feature table from the database")
    parser.add_argument("--horizon", type=int, default=1, help="forecast horizon in days (default 1)")
    parser.add_argument("--extend", type=int, default=0, help="also create this many future rows")
    parser.add_argument("--out", help="save to this file (.parquet or .csv)")
    parser.add_argument("--describe", action="store_true", help="print the feature description table and exit")
    args = parser.parse_args()

    if args.describe:
        with pd.option_context("display.max_colwidth", 100, "display.width", 200):
            print(describe_features(args.horizon).to_string(index=False))
        return

    df = build_dataset(horizon=args.horizon, extend_days=args.extend)
    hist = df[~df["is_future"]]
    feats = feature_columns(args.horizon)
    print(f"rows: {len(df):,}  products: {df['product_id'].nunique()}  features: {len(feats)}  horizon: {args.horizon}")
    print(f"history: {hist['date'].min().date()} -> {hist['date'].max().date()}"
          f"   future rows: {int(df['is_future'].sum()):,}")
    missing = hist[feats].isna().mean().sort_values(ascending=False)
    print("features with missing values (history rows, tree models handle NaN):")
    print(missing[missing > 0].round(3).to_string() if (missing > 0).any() else "  none")
    if args.out:
        df.to_parquet(args.out, index=False) if args.out.endswith(".parquet") else df.to_csv(args.out, index=False)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
