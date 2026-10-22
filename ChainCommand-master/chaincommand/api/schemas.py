"""Request / response models for the database-backed API."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..db.models import Role

_ROLES = (Role.ADMIN, Role.MANAGER, Role.STAFF)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── auth / users ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(ORMModel):
    user_id: int
    name: str
    email: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int
    user: UserOut


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)
    role: str = Role.STAFF

    @field_validator("email")
    @classmethod
    def _email_shape(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError("invalid email address")
        return v

    @field_validator("role")
    @classmethod
    def _known_role(cls, v: str) -> str:
        if v not in _ROLES:
            raise ValueError(f"role must be one of {_ROLES}")
        return v


# ── master data ──────────────────────────────────────────────
class CategoryOut(ORMModel):
    category_id: int
    name: str


class SupplierBase(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    lead_time_mean_days: float = Field(gt=0)
    lead_time_std_days: float = Field(ge=0)
    on_time_rate: float = Field(ge=0, le=1)
    defect_rate: float = Field(ge=0, le=1)
    cost_multiplier: float = Field(default=1.0, gt=0)
    capacity: float = Field(gt=0)


class SupplierCreate(SupplierBase):
    pass


class SupplierUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    lead_time_mean_days: Optional[float] = Field(default=None, gt=0)
    lead_time_std_days: Optional[float] = Field(default=None, ge=0)
    on_time_rate: Optional[float] = Field(default=None, ge=0, le=1)
    defect_rate: Optional[float] = Field(default=None, ge=0, le=1)
    cost_multiplier: Optional[float] = Field(default=None, gt=0)
    capacity: Optional[float] = Field(default=None, gt=0)


class SupplierOut(SupplierBase, ORMModel):
    supplier_id: int


class ProductBase(BaseModel):
    sku: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=200)
    category_id: int
    supplier_id: int
    cost: float = Field(ge=0)
    selling_price: float = Field(ge=0)
    min_order_qty: int = Field(default=1, ge=1)
    pack_size: int = Field(default=1, ge=1)
    reorder_point: float = Field(default=0, ge=0)
    safety_stock: float = Field(default=0, ge=0)
    max_stock: int = Field(gt=0)


class ProductCreate(ProductBase):
    initial_stock: int = Field(default=0, ge=0)


class ProductUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    category_id: Optional[int] = None
    supplier_id: Optional[int] = None
    cost: Optional[float] = Field(default=None, ge=0)
    selling_price: Optional[float] = Field(default=None, ge=0)
    min_order_qty: Optional[int] = Field(default=None, ge=1)
    pack_size: Optional[int] = Field(default=None, ge=1)
    reorder_point: Optional[float] = Field(default=None, ge=0)
    safety_stock: Optional[float] = Field(default=None, ge=0)
    max_stock: Optional[int] = Field(default=None, gt=0)


class ProductOut(ProductBase, ORMModel):
    product_id: int
    current_stock: int


# ── sales ────────────────────────────────────────────────────
class LineIn(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)


class SaleCreate(BaseModel):
    customer_id: Optional[int] = None
    items: list[LineIn] = Field(min_length=1)


class SaleItemOut(ORMModel):
    product_id: int
    quantity: int
    selling_price: float


class SaleOut(ORMModel):
    sale_id: int
    date: dt.date
    customer_id: Optional[int]
    total_amount: float
    items: list[SaleItemOut] = []


# ── purchases ────────────────────────────────────────────────
class PurchaseLineIn(BaseModel):
    product_id: int
    quantity: int = Field(gt=0)
    purchase_price: Optional[float] = Field(default=None, ge=0)


class PurchaseCreate(BaseModel):
    supplier_id: int
    expected_date: Optional[dt.date] = None
    items: list[PurchaseLineIn] = Field(min_length=1)


class PurchaseItemOut(ORMModel):
    product_id: int
    quantity: int
    purchase_price: float


class PurchaseOut(ORMModel):
    purchase_id: int
    supplier_id: int
    date: dt.date
    expected_date: Optional[dt.date]
    received_date: Optional[dt.date]
    status: str
    items: list[PurchaseItemOut] = []


# ── stock ────────────────────────────────────────────────────
class StockRow(BaseModel):
    product_id: int
    sku: str
    name: str
    current_stock: int
    reorder_point: float
    safety_stock: float
    max_stock: int
    avg_daily_demand_30d: Optional[float]
    days_of_cover: Optional[float]
    status: str   # stockout | critical | low | overstock | healthy


class TransactionOut(ORMModel):
    transaction_id: int
    product_id: int
    transaction_type: str
    quantity: int
    timestamp: dt.datetime


class AdjustRequest(BaseModel):
    product_id: int
    quantity: int   # signed, non-zero: +found / -damaged

    @field_validator("quantity")
    @classmethod
    def _non_zero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("quantity must not be zero")
        return v
