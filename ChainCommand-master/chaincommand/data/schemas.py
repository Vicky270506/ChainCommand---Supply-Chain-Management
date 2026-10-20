"""Pydantic data models for the supply chain domain."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Normalize datetimes to aware UTC for safe comparisons."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

# ── Enums ────────────────────────────────────────────────

class ProductCategory(str, Enum):
    ELECTRONICS = "electronics"
    FOOD = "food"
    PHARMA = "pharma"
    INDUSTRIAL = "industrial"
    APPAREL = "apparel"


class OrderStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


# ── Core Domain Models ───────────────────────────────────

class Product(BaseModel):
    product_id: str = Field(default_factory=lambda: f"PRD-{uuid4().hex[:8]}")
    name: str
    category: ProductCategory
    unit_cost: float = Field(ge=0)
    selling_price: float = Field(ge=0)
    lead_time_days: int = 7
    min_order_qty: int = 100
    current_stock: float = Field(default=0.0, ge=0)
    reorder_point: float = 0.0
    safety_stock: float = 0.0
    daily_demand_avg: float = 0.0
    daily_demand_std: float = 0.0


class Supplier(BaseModel):
    supplier_id: str = Field(default_factory=lambda: f"SUP-{uuid4().hex[:8]}")
    name: str
    reliability_score: float = 0.85  # 0-1
    lead_time_mean: float = 7.0
    lead_time_std: float = 2.0
    cost_multiplier: float = 1.0
    capacity: float = 10000.0
    products: List[str] = Field(default_factory=list)  # product_ids
    defect_rate: float = 0.02
    on_time_rate: float = 0.90
    is_active: bool = True


class DemandRecord(BaseModel):
    date: datetime
    product_id: str
    quantity: float
    is_promotion: bool = False
    is_holiday: bool = False
    temperature: Optional[float] = None
    day_of_week: int = 0
    month: int = 1


class InventorySnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    product_id: str
    on_hand: float
    in_transit: float = 0.0
    allocated: float = 0.0
    available: float = 0.0
    days_of_supply: float = 0.0


class PurchaseOrder(BaseModel):
    po_id: str = Field(default_factory=lambda: f"PO-{uuid4().hex[:8]}")
    supplier_id: str
    product_id: str
    quantity: float = Field(ge=0)
    unit_cost: float = Field(ge=0)
    total_cost: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    expected_delivery: Optional[datetime] = None
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None

    @model_validator(mode="after")
    def _auto_calculate_total_cost(self) -> "PurchaseOrder":
        """Auto-calculate total_cost from quantity * unit_cost when total_cost is 0."""
        if self.total_cost == 0 and self.quantity > 0 and self.unit_cost > 0:
            self.total_cost = self.quantity * self.unit_cost
        return self


class AnomalyRecord(BaseModel):
    anomaly_id: str = Field(default_factory=lambda: f"ANM-{uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=utc_now)
    anomaly_type: str  # demand_spike, cost_anomaly, lead_time_anomaly, quality_issue
    product_id: Optional[str] = None
    supplier_id: Optional[str] = None
    severity: AlertSeverity = AlertSeverity.MEDIUM
    score: float = 0.0  # anomaly score from model
    description: str = ""
    resolved: bool = False


class MarketIntelligence(BaseModel):
    intel_id: str = Field(default_factory=lambda: f"MKT-{uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=utc_now)
    source: str = "simulated"
    topic: str = ""
    summary: str = ""
    impact_score: float = 0.0  # -1 to 1
    affected_products: List[str] = Field(default_factory=list)
    raw_data: Dict[str, Any] = Field(default_factory=dict)


# ── KPI Models ───────────────────────────────────────────

class KPISnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    otif: float = 0.0  # On-Time In-Full
    fill_rate: float = 0.0
    mape: float | None = None  # Mean Absolute Percentage Error
    dsi: float = 0.0  # Days Sales of Inventory
    stockout_count: int = 0
    total_inventory_value: float = 0.0
    carrying_cost: float = 0.0
    order_cycle_time: float = 0.0
    perfect_order_rate: float = 0.0
    inventory_turnover: float = 0.0
    backorder_rate: float = 0.0
    supplier_defect_rate: float = 0.0


# ── Event / Alert Models ────────────────────────────────

class SupplyChainEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"EVT-{uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=utc_now)
    event_type: str
    severity: AlertSeverity = AlertSeverity.MEDIUM
    source_agent: str = ""
    description: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    resolution: Optional[str] = None


class HumanApprovalRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"APR-{uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=utc_now)
    request_type: str  # purchase_order, inventory_adjustment, supplier_change
    description: str = ""
    estimated_cost: float = 0.0
    risk_level: AlertSeverity = AlertSeverity.MEDIUM
    data: Dict[str, Any] = Field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None
    reason: Optional[str] = None


# ── Agent Action Models ──────────────────────────────────

class AgentAction(BaseModel):
    action_id: str = Field(default_factory=lambda: f"ACT-{uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=utc_now)
    agent_name: str
    action_type: str
    description: str = ""
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class ForecastResult(BaseModel):
    product_id: str
    forecast_date: datetime
    predicted_demand: float
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0
    model_used: str = "ensemble"
    mape: float = 0.0


class OptimizationResult(BaseModel):
    product_id: str
    recommended_reorder_point: float
    recommended_safety_stock: float
    recommended_order_qty: float
    expected_cost_saving: float = 0.0
    method: str = "genetic_algorithm"


# ── v2.0: Supplier Allocation Models ────────────────────

class SupplierCandidate(BaseModel):
    """A candidate supplier for CP-SAT allocation."""

    supplier_id: str
    unit_cost: float
    risk_score: float = 0.0
    capacity: float = 10_000.0
    min_order_qty: float = 0.0
    lead_time_days: float = 7.0


class AllocationResult(BaseModel):
    """Result of supplier allocation optimization."""

    allocations: Dict[str, float] = Field(default_factory=dict)
    total_cost: float = 0.0
    total_risk: float = 0.0
    objective_value: float = 0.0
    solver_status: str = "unknown"
    solve_time_ms: float = 0.0
    method: str = "cpsat"


class SensitivityResult(BaseModel):
    """Result of sensitivity analysis over risk-cost trade-off."""

    lambda_values: List[float] = Field(default_factory=list)
    costs: List[float] = Field(default_factory=list)
    risks: List[float] = Field(default_factory=list)
    elbow_lambda: float = 0.0
    elbow_cost: float = 0.0
    elbow_risk: float = 0.0
