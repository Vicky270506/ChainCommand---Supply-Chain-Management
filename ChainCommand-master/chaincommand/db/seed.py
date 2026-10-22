"""Seed the database with a realistic synthetic business history.

What gets created
-----------------
* categories, suppliers, products (with reorder constraints and storage capacity)
* ~3 years of daily sales (sales + sale_items) with weekly, monthly, annual,
  promotion and Indian festival effects
* promotions and calendar_events (holidays / festivals, incl. future dates)
* purchase orders placed by a simple (s, S) rule, received after a random
  supplier lead time (purchases + purchase_items)
* an inventory ledger (inventory_transactions) that is the single source of truth
  for stock; it includes lost sales, returns and small adjustments.

Reused from the original ChainCommand repo: the product/supplier templates and the
base demand generator (``chaincommand.data.generator``). Everything else here
(festival uplift, ledger simulation, database loading) is new.

Usage:
    python -m chaincommand.db.seed --reset
    python -m chaincommand.db.seed --reset --products 50 --suppliers 20 --years 3
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from datetime import UTC, date, datetime, timedelta
from typing import Any

import holidays
import numpy as np
from sqlalchemy import Engine, func, insert, select, text

from ..data.generator import (
    assign_suppliers,
    generate_demand_history,
    generate_products,
    generate_suppliers,
)
from ..data.schemas import ProductCategory
from . import models as m
from .base import get_engine, reset_db

# Festivals that lift demand in gift / consumption categories.
_FESTIVAL_KEYWORDS = ("diwali", "deepavali", "holi", "dussehra", "id-ul", "christmas", "janmashtami", "guru nanak")
_FESTIVAL_CATEGORIES = (ProductCategory.ELECTRONICS, ProductCategory.APPAREL, ProductCategory.FOOD)
_PRE_FESTIVAL_FACTOR = 1.4   # festival day and the 3 days before it
_POST_FESTIVAL_FACTOR = 1.1  # the day after
_PACK_SIZES = (1, 5, 10, 25)  # all divide the generator's MOQs (50/100/200/500)

_CHUNK = 20_000


def _festival_day_factors(events: dict[date, str]) -> dict[date, float]:
    factors: dict[date, float] = {}
    for d, name in events.items():
        if any(k in name.lower() for k in _FESTIVAL_KEYWORDS):
            for off in range(-3, 1):
                factors[d + timedelta(days=off)] = max(factors.get(d + timedelta(days=off), 1.0), _PRE_FESTIVAL_FACTOR)
            nxt = d + timedelta(days=1)
            factors[nxt] = max(factors.get(nxt, 1.0), _POST_FESTIVAL_FACTOR)
    return factors


def _bulk_insert(engine: Engine, model: Any, rows: list[dict]) -> None:
    for i in range(0, len(rows), _CHUNK):
        with engine.begin() as conn:
            conn.execute(insert(model), rows[i:i + _CHUNK])


def seed_database(
    engine: Engine,
    n_products: int = 50,
    n_suppliers: int = 20,
    days: int = 3 * 365,
    seed: int = 42,
) -> dict[str, int]:
    """Fill an (empty) database. Returns the row count per table."""
    rng = random.Random(seed)

    # 1. Base data from the original generator ------------------------------
    products = generate_products(n_products, rng)
    suppliers = generate_suppliers(n_suppliers, rng)
    suppliers = assign_suppliers(products, suppliers, rng)
    demand = generate_demand_history(products, days=days, rng=rng)
    demand["day"] = demand["date"].dt.date
    # History ends yesterday: today's business is recorded live through the API.
    demand = demand[demand["day"] < datetime.now(UTC).date()].copy()

    first_day: date = demand["day"].min()
    last_day: date = demand["day"].max()

    # 2. Calendar: holidays + festivals (also future dates for forecasting) -
    cal = holidays.India(years=range(first_day.year, last_day.year + 2))
    events = {d: n for d, n in cal.items() if d <= last_day + timedelta(days=365)}
    event_rows = []
    for i, (d, name) in enumerate(sorted(events.items()), start=1):
        is_fest = any(k in name.lower() for k in _FESTIVAL_KEYWORDS)
        event_rows.append({"event_id": i, "date": d, "name": name[:200],
                           "event_type": "festival" if is_fest else "holiday"})

    # 3. Festival uplift on the generated demand ----------------------------
    day_factor = _festival_day_factors(events)
    cat_by_pid = {p.product_id: p.category for p in products}
    factor = demand["day"].map(day_factor).fillna(1.0)
    eligible = demand["product_id"].map(cat_by_pid).isin(_FESTIVAL_CATEGORIES)
    factor = factor.where(eligible, 1.0)
    demand["quantity"] = np.rint(demand["quantity"] * factor).astype(int)

    # 4. Master data --------------------------------------------------------
    categories = sorted({p.category for p in products}, key=lambda c: c.value)
    cat_id = {c: i for i, c in enumerate(categories, start=1)}
    category_rows = [{"category_id": cat_id[c], "name": c.value.title()} for c in categories]

    sup_id = {s.supplier_id: i for i, s in enumerate(suppliers, start=1)}
    supplier_rows = [{
        "supplier_id": sup_id[s.supplier_id], "name": s.name,
        "lead_time_mean_days": round(float(s.lead_time_mean), 2),
        "lead_time_std_days": round(float(s.lead_time_std), 2),
        "on_time_rate": float(s.on_time_rate), "defect_rate": float(s.defect_rate),
        "cost_multiplier": round(float(s.cost_multiplier), 3), "capacity": round(float(s.capacity), 0),
    } for s in suppliers]

    stats = demand.groupby("product_id")["quantity"].agg(["mean", "std"])
    product_rows: list[dict] = []
    plan: dict[str, dict] = {}   # generator product_id -> simulation parameters
    for i, p in enumerate(products, start=1):
        offering = [s for s in suppliers if p.product_id in s.products]
        primary = max(offering, key=lambda s: s.on_time_rate)
        avg = float(stats.loc[p.product_id, "mean"])
        std = float(stats.loc[p.product_id, "std"]) if not math.isnan(stats.loc[p.product_id, "std"]) else 0.0
        lead = float(primary.lead_time_mean)
        safety = 1.65 * std * math.sqrt(lead)
        reorder_point = avg * lead + safety
        order_up_to = reorder_point + 14 * avg
        pack = rng.choice(_PACK_SIZES)
        overstocked = rng.random() < 0.3      # ChainInsight "2.2x overstock" pattern
        opening = int(round(reorder_point * (2.2 if overstocked else 1.3)))
        cost = round(float(p.unit_cost), 2)
        product_rows.append({
            "product_id": i, "sku": p.product_id, "name": p.name,
            "category_id": cat_id[p.category], "supplier_id": sup_id[primary.supplier_id],
            "cost": cost, "selling_price": round(float(p.selling_price), 2),
            "min_order_qty": int(p.min_order_qty), "pack_size": pack,
            "reorder_point": round(reorder_point, 1), "safety_stock": round(safety, 1),
            "max_stock": int(order_up_to * 1.5), "current_stock": 0,
        })
        plan[p.product_id] = {
            "db_id": i, "supplier": primary, "supplier_db_id": sup_id[primary.supplier_id],
            "reorder_point": reorder_point, "order_up_to": order_up_to, "pack": pack,
            "moq": int(p.min_order_qty), "opening": opening,
            "price": round(float(p.selling_price), 2),
            "buy_price": round(cost * float(primary.cost_multiplier), 2),
        }

    # 5. Promotions ---------------------------------------------------------
    promo_rows = []
    promo_df = demand[demand["is_promotion"]]
    for j, (pid, d) in enumerate(zip(promo_df["product_id"], promo_df["day"], strict=True), start=1):
        promo_rows.append({"promotion_id": j, "product_id": plan[pid]["db_id"], "start_date": d,
                           "end_date": d, "discount_pct": round(rng.uniform(10, 30), 1)})

    # 6. Simulate the business day by day, per product ----------------------
    purchase_ids = itertools.count(1)
    purchase_item_ids = itertools.count(1)
    tx_ids = itertools.count(1)
    purchase_rows: list[dict] = []
    purchase_item_rows: list[dict] = []
    tx_rows: list[dict] = []
    sales_by_day: dict[date, list[tuple[int, int, float]]] = {}
    final_stock: dict[int, int] = {}

    def tx(pid: int, kind: str, qty: int, day: date, hour: int) -> None:
        tx_rows.append({"transaction_id": next(tx_ids), "product_id": pid, "transaction_type": kind,
                        "quantity": int(qty), "timestamp": datetime(day.year, day.month, day.day, hour)})

    for code, g in demand.sort_values("date").groupby("product_id"):
        cfg = plan[code]
        pid = cfg["db_id"]
        sup = cfg["supplier"]
        days_list = g["day"].tolist()
        demand_list = g["quantity"].tolist()

        stock = int(cfg["opening"])
        tx(pid, m.TxType.OPENING, stock, days_list[0], 0)
        pending: list[tuple[date, int, dict]] = []   # (arrival_day, qty, purchase_row)

        for day, wanted in zip(days_list, demand_list, strict=True):
            # a) receive orders that arrive today
            still = []
            for arrive, qty, prow in pending:
                if arrive <= day:
                    stock += qty
                    prow["status"] = m.PurchaseStatus.RECEIVED
                    prow["received_date"] = day
                    tx(pid, m.TxType.PURCHASE_RECEIPT, qty, day, 9)
                else:
                    still.append((arrive, qty, prow))
            pending = still

            # b) serve demand from stock; the rest is a lost sale
            served = min(stock, int(wanted))
            lost = int(wanted) - served
            if served > 0:
                stock -= served
                sales_by_day.setdefault(day, []).append((pid, served, cfg["price"]))
                tx(pid, m.TxType.SALE, -served, day, 18)
            if lost > 0:
                tx(pid, m.TxType.LOST_SALE, lost, day, 18)

            # c) occasional customer returns and small stock adjustments
            if served > 0 and rng.random() < 0.01:
                back = max(1, int(served * rng.uniform(0.02, 0.10)))
                stock += back
                tx(pid, m.TxType.RETURN, back, day, 19)
            if stock > 10 and rng.random() < 0.003:
                shrink = rng.randint(1, min(10, stock))
                stock -= shrink
                tx(pid, m.TxType.ADJUSTMENT, -shrink, day, 23)

            # d) (s, S) reorder rule on inventory position
            position = stock + sum(q for _, q, _ in pending)
            if position <= cfg["reorder_point"]:
                raw = cfg["order_up_to"] - position
                qty = max(cfg["moq"], int(math.ceil(raw / cfg["pack"])) * cfg["pack"])
                lead = max(1, int(round(rng.gauss(sup.lead_time_mean, sup.lead_time_std))))
                arrive = day + timedelta(days=lead)
                prow = {"purchase_id": next(purchase_ids), "supplier_id": cfg["supplier_db_id"], "date": day,
                        "expected_date": arrive, "received_date": None, "status": m.PurchaseStatus.ORDERED}
                purchase_rows.append(prow)
                purchase_item_rows.append({"purchase_item_id": next(purchase_item_ids),
                                           "purchase_id": prow["purchase_id"], "product_id": pid,
                                           "quantity": qty, "purchase_price": cfg["buy_price"]})
                pending.append((arrive, qty, prow))

        final_stock[pid] = stock

    for row in product_rows:
        row["current_stock"] = int(final_stock[row["product_id"]])

    # sales headers (one POS "walk-in" sale per day) + items
    sale_rows: list[dict] = []
    sale_item_rows: list[dict] = []
    item_ids = itertools.count(1)
    for sale_id, day in enumerate(sorted(sales_by_day), start=1):
        lines = sales_by_day[day]
        sale_rows.append({"sale_id": sale_id, "date": day, "customer_id": None,
                          "total_amount": round(sum(q * pr for _, q, pr in lines), 2)})
        for pid, q, pr in lines:
            sale_item_rows.append({"sale_item_id": next(item_ids), "sale_id": sale_id,
                                   "product_id": pid, "quantity": int(q), "selling_price": pr})

    # 7. Load (parents before children) -------------------------------------
    load_order = [
        (m.Category, category_rows), (m.Supplier, supplier_rows), (m.Product, product_rows),
        (m.Sale, sale_rows), (m.SaleItem, sale_item_rows),
        (m.Purchase, purchase_rows), (m.PurchaseItem, purchase_item_rows),
        (m.InventoryTransaction, tx_rows), (m.Promotion, promo_rows), (m.CalendarEvent, event_rows),
    ]
    for model, rows in load_order:
        if rows:
            _bulk_insert(engine, model, rows)

    sync_sequences(engine)
    return table_counts(engine)


def sync_sequences(engine: Engine) -> None:
    """PostgreSQL only: move each auto-ID counter past the highest id already used.

    The seed inserts rows with explicit ids, and PostgreSQL does not advance its
    sequences for those, so the next normal INSERT would collide with id 1.
    """
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        for table in m.Base.metadata.sorted_tables:
            pk = list(table.primary_key.columns)[0]
            conn.execute(text(
                f"SELECT setval(pg_get_serial_sequence('{table.name}', '{pk.name}'), "  # noqa: S608
                f"(SELECT COALESCE(MAX({pk.name}), 0) + 1 FROM {table.name}), false)"  # noqa: S608
            ))


def table_counts(engine: Engine) -> dict[str, int]:
    counts = {}
    with engine.connect() as conn:
        for table in m.Base.metadata.sorted_tables:
            counts[table.name] = conn.execute(select(func.count()).select_from(table)).scalar_one()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the inventory database with synthetic data")
    parser.add_argument("--reset", action="store_true", help="drop and recreate all tables first (DESTROYS DATA)")
    parser.add_argument("--products", type=int, default=50)
    parser.add_argument("--suppliers", type=int, default=20)
    parser.add_argument("--years", type=float, default=3.0, help="years of sales history")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync-sequences", action="store_true",
                        help="only repair PostgreSQL id counters on an already-seeded database, then exit")
    args = parser.parse_args()

    engine = get_engine()
    if args.sync_sequences:
        sync_sequences(engine)
        print("Sequences synchronised.")
        return
    if args.reset:
        reset_db(engine)
    else:
        m.Base.metadata.create_all(engine)
        if table_counts(engine)["products"] > 0:
            raise SystemExit("Database already contains data. Re-run with --reset to rebuild it.")

    counts = seed_database(engine, args.products, args.suppliers, int(args.years * 365), args.seed)
    print(f"Seeded {engine.url.render_as_string(hide_password=True)}")
    for name, n in counts.items():
        print(f"  {name:24s} {n:>9,}")


if __name__ == "__main__":
    main()
