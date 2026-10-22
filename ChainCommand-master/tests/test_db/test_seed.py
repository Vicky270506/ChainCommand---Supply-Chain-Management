"""Tests for the database schema and the seed script."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import inspect, select, text

from chaincommand.db import make_engine
from chaincommand.db import models as m
from chaincommand.db.seed import seed_database, table_counts

REPORT_TABLES = {
    "users", "products", "sales", "sale_items", "purchases", "purchase_items",
    "inventory_transactions", "forecasts", "rl_recommendations", "mlflow_metadata",
}


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.db"
    eng = make_engine(f"sqlite:///{path}")
    m.Base.metadata.create_all(eng)
    seed_database(eng, n_products=8, n_suppliers=5, days=500, seed=7)
    yield eng
    eng.dispose()


def test_all_report_tables_exist(engine):
    assert REPORT_TABLES <= set(inspect(engine).get_table_names())


def test_row_counts(engine):
    counts = table_counts(engine)
    assert counts["products"] == 8
    assert counts["suppliers"] == 5
    assert counts["sales"] >= 400
    assert counts["sale_items"] > counts["sales"]
    assert counts["purchases"] == counts["purchase_items"] > 0
    assert counts["calendar_events"] > 0


def test_no_orphan_foreign_keys(engine):
    # (child table, child column, parent table, parent column)
    links = [
        ("sale_items", "sale_id", "sales", "sale_id"),
        ("sale_items", "product_id", "products", "product_id"),
        ("purchase_items", "purchase_id", "purchases", "purchase_id"),
        ("inventory_transactions", "product_id", "products", "product_id"),
        ("products", "supplier_id", "suppliers", "supplier_id"),
    ]
    queries = [
        f"select count(*) from {c} left join {p} on {p}.{pc} = {c}.{cc} where {p}.{pc} is null"  # noqa: S608
        for c, cc, p, pc in links
    ]
    with engine.connect() as conn:
        assert all(conn.execute(text(q)).scalar_one() == 0 for q in queries)


def test_sale_totals_match_items(engine):
    df = pd.read_sql(
        text("select s.sale_id, s.total_amount, sum(si.quantity*si.selling_price) as items_total "
             "from sales s join sale_items si on si.sale_id=s.sale_id group by s.sale_id, s.total_amount"),
        engine,
    )
    assert ((df["total_amount"] - df["items_total"]).abs() < 0.05).all()


def test_ledger_matches_current_stock(engine):
    ledger = pd.read_sql(
        text("select product_id, sum(quantity) as stock from inventory_transactions "
             "where transaction_type != 'lost_sale' group by product_id"), engine,
    ).set_index("product_id")["stock"]
    products = pd.read_sql(select(m.Product.product_id, m.Product.current_stock), engine).set_index("product_id")
    assert (ledger.reindex(products.index) == products["current_stock"]).all()


def test_stock_never_negative(engine):
    tx = pd.read_sql(
        text("select * from inventory_transactions where transaction_type != 'lost_sale'"), engine,
        parse_dates=["timestamp"],
    ).sort_values(["product_id", "timestamp", "transaction_id"])
    assert (tx.groupby("product_id")["quantity"].cumsum() >= 0).all()


def test_stockouts_exist_for_the_classifier(engine):
    # The stockout-risk model needs both classes to learn from.
    with engine.connect() as conn:
        count_sql = "select count(*) from inventory_transactions where transaction_type = :kind"
        lost = conn.execute(text(count_sql), {"kind": "lost_sale"}).scalar_one()
        sold = conn.execute(text(count_sql), {"kind": "sale"}).scalar_one()
    assert 0 < lost < sold


def test_reorder_constraints_respected(engine):
    df = pd.read_sql(
        text("select pi.quantity, p.min_order_qty, p.pack_size from purchase_items pi "
             "join products p on p.product_id = pi.product_id"), engine,
    )
    assert (df["quantity"] >= df["min_order_qty"]).all()
    assert (df["quantity"] % df["pack_size"] == 0).all()


def test_calendar_contains_festivals_including_future(engine):
    ev = pd.read_sql(select(m.CalendarEvent.date, m.CalendarEvent.name, m.CalendarEvent.event_type), engine,
                     parse_dates=["date"])
    assert (ev["event_type"] == "festival").any()
    assert ev["name"].str.contains("Diwali").any()
    last_sale = pd.read_sql(text("select max(date) d from sales"), engine, parse_dates=["d"])["d"].iloc[0]
    assert ev["date"].max() > last_sale
