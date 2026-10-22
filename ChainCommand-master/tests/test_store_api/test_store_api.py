"""Tests for login, role-based access and the database-backed inventory routes."""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from chaincommand.api.deps import get_current_user
from chaincommand.api.routes.auth import router as auth_router
from chaincommand.api.routes.store import router as store_router
from chaincommand.api.routes.store import stock_status
from chaincommand.db import get_db, make_engine
from chaincommand.db import models as m
from chaincommand.db.seed import seed_database
from chaincommand.security import create_access_token, hash_password, verify_password

PASSWORD = "CorrectHorse1"  # noqa: S105


@pytest.fixture()
def env(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'api.db'}")
    m.Base.metadata.create_all(engine)
    seed_database(engine, n_products=6, n_suppliers=4, days=200, seed=3)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with Session(engine) as s:
        for name, role in (("Admin", m.Role.ADMIN), ("Manager", m.Role.MANAGER), ("Staff", m.Role.STAFF)):
            s.add(m.User(name=name, email=f"{name.lower()}@test.com", role=role, password_hash=hash_password(PASSWORD)))
        s.commit()

    app = FastAPI()
    app.include_router(auth_router, prefix="/api")
    app.include_router(store_router, prefix="/api")

    def override_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    client = TestClient(app)
    yield client, factory
    engine.dispose()


def login(client, who: str) -> dict:
    r = client.post("/api/auth/login", json={"email": f"{who}@test.com", "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ── passwords & tokens ───────────────────────────────────────
def test_password_hash_is_not_plaintext_and_verifies():
    hashed = hash_password(PASSWORD)
    assert hashed != PASSWORD and hashed.startswith("$argon2")
    assert verify_password(PASSWORD, hashed)
    assert not verify_password("wrong-password", hashed)
    assert not verify_password(PASSWORD, "not-a-hash")


def test_stored_passwords_are_hashed(env):
    _, factory = env
    with factory() as s:
        hashes = s.scalars(select(m.User.password_hash)).all()
    assert hashes and all(h.startswith("$argon2") for h in hashes)


def test_login_success_and_failure(env):
    client, _ = env
    ok = client.post("/api/auth/login", json={"email": "STAFF@test.com", "password": PASSWORD})
    assert ok.status_code == 200 and ok.json()["user"]["role"] == "staff" and ok.json()["token_type"] == "bearer"  # noqa: S105
    assert "password" not in ok.text
    assert client.post("/api/auth/login", json={"email": "staff@test.com", "password": "nope"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "ghost@test.com", "password": PASSWORD}).status_code == 401


def test_endpoints_require_a_valid_token(env):
    client, factory = env
    assert client.get("/api/products").status_code == 401
    assert client.get("/api/products", headers={"Authorization": "Bearer garbage"}).status_code == 401
    expired, _ = create_access_token(1, m.Role.ADMIN, expires_minutes=-1)
    assert client.get("/api/products", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    token = login(client, "staff")["Authorization"]
    tampered = {"Authorization": token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")}
    assert client.get("/api/products", headers=tampered).status_code == 401
    assert client.get("/api/auth/me", headers=login(client, "staff")).json()["email"] == "staff@test.com"


# ── role-based access ────────────────────────────────────────
def test_role_permissions(env):
    client, _ = env
    staff, manager, admin = login(client, "staff"), login(client, "manager"), login(client, "admin")
    new_user = {"name": "New", "email": "new@test.com", "password": "LongEnough1", "role": "staff"}
    assert client.post("/api/auth/users", json=new_user, headers=staff).status_code == 403
    assert client.post("/api/auth/users", json=new_user, headers=manager).status_code == 403
    assert client.post("/api/auth/users", json=new_user, headers=admin).status_code == 201
    assert client.post("/api/auth/users", json=new_user, headers=admin).status_code == 409
    assert client.post("/api/auth/users", json={**new_user, "email": "x@test.com", "password": "short"},
                       headers=admin).status_code == 422
    assert client.post("/api/auth/users", json={**new_user, "email": "y@test.com", "role": "root"},
                       headers=admin).status_code == 422
    assert client.get("/api/auth/users", headers=manager).status_code == 403

    product = {"sku": "NEW-1", "name": "New item", "category_id": 1, "supplier_id": 1, "cost": 2.0,
               "selling_price": 5.0, "max_stock": 100, "initial_stock": 10}
    assert client.post("/api/products", json=product, headers=staff).status_code == 403
    created = client.post("/api/products", json=product, headers=manager)
    assert created.status_code == 201 and created.json()["current_stock"] == 10
    assert client.post("/api/products", json=product, headers=manager).status_code == 409
    assert client.post("/api/stock/adjust", json={"product_id": 1, "quantity": 1}, headers=staff).status_code == 403


# ── sales ────────────────────────────────────────────────────
def _stock(factory, pid):
    with factory() as s:
        return s.get(m.Product, pid).current_stock


def _ledger_sum(factory, pid):
    with factory() as s:
        return s.execute(text("select sum(quantity) from inventory_transactions "
                              "where product_id=:p and transaction_type != 'lost_sale'"), {"p": pid}).scalar_one()


def test_sale_reduces_stock_and_writes_ledger(env):
    client, factory = env
    staff = login(client, "staff")
    with factory() as s:
        p = s.scalars(select(m.Product).where(m.Product.current_stock > 5)).first()
        pid, before, price = p.product_id, p.current_stock, p.selling_price
    r = client.post("/api/sales", json={"items": [{"product_id": pid, "quantity": 2},
                                                   {"product_id": pid, "quantity": 1}]}, headers=staff)
    assert r.status_code == 201, r.text
    body = r.json()
    assert len(body["items"]) == 1 and body["items"][0]["quantity"] == 3     # duplicate lines merged
    assert body["total_amount"] == pytest.approx(3 * price, abs=0.01)
    assert body["date"] == dt.datetime.now(dt.UTC).date().isoformat()
    assert _stock(factory, pid) == before - 3
    assert _ledger_sum(factory, pid) == before - 3                           # ledger still equals stock
    assert client.get(f"/api/sales/{body['sale_id']}", headers=staff).json()["items"][0]["product_id"] == pid


def test_sale_with_insufficient_stock_changes_nothing(env):
    client, factory = env
    staff = login(client, "staff")
    with factory() as s:
        p = s.get(m.Product, 1)
        pid, before = p.product_id, p.current_stock
        n_sales = len(s.scalars(select(m.Sale.sale_id)).all())
    r = client.post("/api/sales", json={"items": [{"product_id": pid, "quantity": before + 1}]}, headers=staff)
    assert r.status_code == 409 and r.json()["detail"]["items"][0]["available"] == before
    assert _stock(factory, pid) == before
    with factory() as s:
        assert len(s.scalars(select(m.Sale.sale_id)).all()) == n_sales
    assert client.post("/api/sales", json={"items": [{"product_id": 99999, "quantity": 1}]},
                       headers=staff).status_code == 404
    assert client.post("/api/sales", json={"items": [{"product_id": pid, "quantity": 0}]},
                       headers=staff).status_code == 422
    assert client.post("/api/sales", json={"items": []}, headers=staff).status_code == 422


# ── purchases ────────────────────────────────────────────────
def test_purchase_rules_and_receiving(env):
    client, factory = env
    manager, staff = login(client, "manager"), login(client, "staff")
    with factory() as s:
        p = s.get(m.Product, 2)
        pid, moq, pack, before, sup = p.product_id, p.min_order_qty, p.pack_size, p.current_stock, p.supplier_id

    assert client.post("/api/purchases", json={"supplier_id": sup, "items": [{"product_id": pid, "quantity": moq}]},
                       headers=staff).status_code == 403
    below = client.post("/api/purchases", json={"supplier_id": sup, "items": [{"product_id": pid, "quantity": 1}]},
                        headers=manager)
    assert below.status_code == 422 and "minimum order quantity" in str(below.json()["detail"])
    if pack > 1:
        odd = client.post("/api/purchases", headers=manager,
                          json={"supplier_id": sup, "items": [{"product_id": pid, "quantity": moq + 1}]})
        assert odd.status_code == 422 and "pack size" in str(odd.json()["detail"])

    created = client.post("/api/purchases", headers=manager,
                          json={"supplier_id": sup, "items": [{"product_id": pid, "quantity": moq}]})
    assert created.status_code == 201, created.text
    po = created.json()
    assert po["status"] == "ordered" and po["items"][0]["purchase_price"] > 0
    assert _stock(factory, pid) == before                                     # nothing arrives until received

    got = client.post(f"/api/purchases/{po['purchase_id']}/receive", headers=manager)
    assert got.status_code == 200 and got.json()["status"] == "received" and got.json()["received_date"]
    assert _stock(factory, pid) == before + moq
    assert _ledger_sum(factory, pid) == before + moq
    assert client.post(f"/api/purchases/{po['purchase_id']}/receive", headers=manager).status_code == 409
    assert client.post(f"/api/purchases/{po['purchase_id']}/cancel", headers=manager).status_code == 409


# ── stock ────────────────────────────────────────────────────
def test_adjustment_cannot_make_stock_negative(env):
    client, factory = env
    manager = login(client, "manager")
    before = _stock(factory, 1)
    assert client.post("/api/stock/adjust", json={"product_id": 1, "quantity": -(before + 1)},
                       headers=manager).status_code == 409
    assert client.post("/api/stock/adjust", json={"product_id": 1, "quantity": 0}, headers=manager).status_code == 422
    ok = client.post("/api/stock/adjust", json={"product_id": 1, "quantity": 7}, headers=manager)
    assert ok.status_code == 200 and ok.json()["current_stock"] == before + 7
    assert _ledger_sum(factory, 1) == before + 7
    tx = client.get("/api/stock/1/transactions?limit=1", headers=manager).json()
    assert tx[0]["transaction_type"] == "adjustment" and tx[0]["quantity"] == 7


def test_stock_overview_labels(env):
    client, _ = env
    staff = login(client, "staff")
    rows = client.get("/api/stock", headers=staff).json()
    assert len(rows) == 6
    assert {r["status"] for r in rows} <= {"stockout", "critical", "low", "overstock", "healthy"}
    only = client.get("/api/stock?status=healthy", headers=staff).json()
    assert all(r["status"] == "healthy" for r in only)


def test_stock_status_rules():
    assert stock_status(0, 10, 50, 5.0) == "stockout"
    assert stock_status(5, 10, 50, 1.0) == "critical"
    assert stock_status(30, 10, 50, 3.0) == "low"
    assert stock_status(500, 10, 50, 90.0) == "overstock"
    assert stock_status(100, 10, 50, 20.0) == "healthy"
    assert stock_status(100, 10, 50, None) == "healthy"


def test_get_current_user_is_the_auth_dependency():
    assert callable(get_current_user)
