"""SQLAlchemy models for the Hybrid Inventory Management System.

Implements the database design in section 16 of the project report
(users, products, sales, sale_items, purchases, purchase_items,
inventory_transactions, forecasts, rl_recommendations, mlflow_metadata)
plus the supporting tables the data layer needs (categories, suppliers,
promotions, calendar_events).

The models are portable: they run on PostgreSQL (target) and SQLite (dev/tests).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Optional

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Naive UTC timestamp (all DateTime columns store naive UTC)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# ── Allowed string values (kept as plain strings for portability) ─────────

class Role:
    ADMIN = "admin"
    MANAGER = "manager"
    STAFF = "staff"


class TxType:
    OPENING = "opening_balance"       # starting stock when the ledger begins
    SALE = "sale"                     # negative quantity
    PURCHASE_RECEIPT = "purchase_receipt"  # positive quantity
    RETURN = "return"                 # positive quantity (customer return)
    ADJUSTMENT = "adjustment"         # signed quantity (shrinkage / correction)
    LOST_SALE = "lost_sale"           # demand that could not be served; NO stock effect


class PurchaseStatus:
    ORDERED = "ordered"
    RECEIVED = "received"
    CANCELLED = "cancelled"


class ApprovalStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ── Core tables ──────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    role: Mapped[str] = mapped_column(String(20), default=Role.STAFF)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Category(Base):
    __tablename__ = "categories"

    category_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)


class Supplier(Base):
    __tablename__ = "suppliers"

    supplier_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150), unique=True)
    lead_time_mean_days: Mapped[float] = mapped_column(Float)
    lead_time_std_days: Mapped[float] = mapped_column(Float)
    on_time_rate: Mapped[float] = mapped_column(Float)
    defect_rate: Mapped[float] = mapped_column(Float)
    cost_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    capacity: Mapped[float] = mapped_column(Float)


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.category_id"))
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.supplier_id"))
    cost: Mapped[float] = mapped_column(Float)
    selling_price: Mapped[float] = mapped_column(Float)
    # reorder constraints (report: "reorder constraints")
    min_order_qty: Mapped[int] = mapped_column(Integer, default=1)
    pack_size: Mapped[int] = mapped_column(Integer, default=1)
    reorder_point: Mapped[float] = mapped_column(Float, default=0.0)
    safety_stock: Mapped[float] = mapped_column(Float, default=0.0)
    max_stock: Mapped[int] = mapped_column(Integer)          # storage capacity
    current_stock: Mapped[int] = mapped_column(Integer, default=0)  # == sum of ledger


class Sale(Base):
    __tablename__ = "sales"

    sale_id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    customer_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_amount: Mapped[float] = mapped_column(Float, default=0.0)


class SaleItem(Base):
    __tablename__ = "sale_items"

    sale_item_id: Mapped[int] = mapped_column(primary_key=True)
    sale_id: Mapped[int] = mapped_column(ForeignKey("sales.sale_id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    selling_price: Mapped[float] = mapped_column(Float)


class Purchase(Base):
    __tablename__ = "purchases"

    purchase_id: Mapped[int] = mapped_column(primary_key=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.supplier_id"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    expected_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    received_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default=PurchaseStatus.ORDERED)


class PurchaseItem(Base):
    __tablename__ = "purchase_items"

    purchase_item_id: Mapped[int] = mapped_column(primary_key=True)
    purchase_id: Mapped[int] = mapped_column(ForeignKey("purchases.purchase_id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    purchase_price: Mapped[float] = mapped_column(Float)


class InventoryTransaction(Base):
    __tablename__ = "inventory_transactions"
    __table_args__ = (Index("ix_invtx_product_ts", "product_id", "timestamp"),)

    transaction_id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"))
    transaction_type: Mapped[str] = mapped_column(String(30))
    quantity: Mapped[int] = mapped_column(Integer)   # signed; see TxType
    timestamp: Mapped[datetime] = mapped_column(DateTime)


# ── Context tables for feature engineering (report: promotions, holidays) ─

class Promotion(Base):
    __tablename__ = "promotions"

    promotion_id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"), index=True)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    discount_pct: Mapped[float] = mapped_column(Float)


class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    event_id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    name: Mapped[str] = mapped_column(String(200))
    event_type: Mapped[str] = mapped_column(String(20))  # "holiday" | "festival"


# ── ML / decision tables ─────────────────────────────────────────────────

class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (Index("ix_forecast_product_date", "product_id", "forecast_date"),)

    forecast_id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"))
    forecast_date: Mapped[date] = mapped_column(Date)
    predicted_demand: Mapped[float] = mapped_column(Float)
    lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50))
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RLRecommendation(Base):
    """One replenishment recommendation with full traceability (report section 18)."""

    __tablename__ = "rl_recommendations"

    recommendation_id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"), index=True)
    recommended_quantity: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(30))
    expected_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    policy_version: Mapped[str] = mapped_column(String(50))
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)  # prediction timestamp
    # traceability
    input_data_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    forecast_model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    forecast_demand: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stockout_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rule_based_quantity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # human approval
    approval_status: Mapped[str] = mapped_column(String(20), default=ApprovalStatus.PENDING)
    decided_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class MLflowMetadata(Base):
    __tablename__ = "mlflow_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    model_version: Mapped[str] = mapped_column(String(20))
    run_id: Mapped[str] = mapped_column(String(64))
    validation_metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    deployment_status: Mapped[str] = mapped_column(String(30), default="candidate")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
