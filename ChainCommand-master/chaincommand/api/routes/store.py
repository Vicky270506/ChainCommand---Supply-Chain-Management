"""Database-backed inventory routes: products, suppliers, sales, purchases, stock.

Access levels:  staff = read + record sales | manager = + edit master data, purchases,
stock adjustments | admin = + user management (see routes/auth.py).
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ...config import settings
from ...db import get_db
from ...db.models import (
    Category,
    InventoryTransaction,
    Product,
    Purchase,
    PurchaseItem,
    PurchaseStatus,
    Sale,
    SaleItem,
    Supplier,
    TxType,
    User,
    utcnow,
)
from ..deps import require_manager, require_staff
from ..schemas import (
    AdjustRequest,
    CategoryOut,
    ProductCreate,
    ProductOut,
    ProductUpdate,
    PurchaseCreate,
    PurchaseOut,
    SaleCreate,
    SaleOut,
    StockRow,
    SupplierCreate,
    SupplierOut,
    SupplierUpdate,
    TransactionOut,
)

router = APIRouter(tags=["inventory"])


def _get_or_404(db: Session, model, key: int, label: str):
    obj = db.get(model, key)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{label} {key} not found")
    return obj


# ── categories & suppliers ───────────────────────────────────
@router.get("/categories", response_model=list[CategoryOut])
def list_categories(db: Session = Depends(get_db), _u: User = Depends(require_staff)):
    return db.scalars(select(Category).order_by(Category.category_id)).all()


@router.get("/suppliers", response_model=list[SupplierOut])
def list_suppliers(db: Session = Depends(get_db), _u: User = Depends(require_staff)):
    return db.scalars(select(Supplier).order_by(Supplier.supplier_id)).all()


@router.get("/suppliers/{supplier_id}", response_model=SupplierOut)
def get_supplier(supplier_id: int, db: Session = Depends(get_db), _u: User = Depends(require_staff)):
    return _get_or_404(db, Supplier, supplier_id, "Supplier")


@router.post("/suppliers", response_model=SupplierOut, status_code=status.HTTP_201_CREATED)
def create_supplier(body: SupplierCreate, db: Session = Depends(get_db), _u: User = Depends(require_manager)):
    if db.scalar(select(Supplier).where(Supplier.name == body.name)):
        raise HTTPException(status.HTTP_409_CONFLICT, "A supplier with this name already exists")
    supplier = Supplier(**body.model_dump())
    db.add(supplier)
    db.commit()
    return supplier


@router.put("/suppliers/{supplier_id}", response_model=SupplierOut)
def update_supplier(supplier_id: int, body: SupplierUpdate, db: Session = Depends(get_db),
                    _u: User = Depends(require_manager)):
    supplier = _get_or_404(db, Supplier, supplier_id, "Supplier")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(supplier, field, value)
    db.commit()
    return supplier


# ── products ─────────────────────────────────────────────────
@router.get("/products", response_model=list[ProductOut])
def list_products(
    q: Optional[str] = Query(default=None, description="search in name or SKU"),
    category_id: Optional[int] = None,
    supplier_id: Optional[int] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _u: User = Depends(require_staff),
):
    stmt = select(Product).order_by(Product.product_id)
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(or_(func.lower(Product.name).like(like), func.lower(Product.sku).like(like)))
    if category_id is not None:
        stmt = stmt.where(Product.category_id == category_id)
    if supplier_id is not None:
        stmt = stmt.where(Product.supplier_id == supplier_id)
    return db.scalars(stmt.limit(limit).offset(offset)).all()


@router.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db), _u: User = Depends(require_staff)):
    return _get_or_404(db, Product, product_id, "Product")


@router.post("/products", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def create_product(body: ProductCreate, db: Session = Depends(get_db), _u: User = Depends(require_manager)):
    if db.scalar(select(Product).where(Product.sku == body.sku)):
        raise HTTPException(status.HTTP_409_CONFLICT, f"SKU {body.sku} already exists")
    _get_or_404(db, Category, body.category_id, "Category")
    _get_or_404(db, Supplier, body.supplier_id, "Supplier")
    data = body.model_dump(exclude={"initial_stock"})
    product = Product(**data, current_stock=body.initial_stock)
    db.add(product)
    db.flush()
    if body.initial_stock:
        db.add(InventoryTransaction(product_id=product.product_id, transaction_type=TxType.OPENING,
                                    quantity=body.initial_stock, timestamp=utcnow()))
    db.commit()
    return product


@router.put("/products/{product_id}", response_model=ProductOut)
def update_product(product_id: int, body: ProductUpdate, db: Session = Depends(get_db),
                   _u: User = Depends(require_manager)):
    product = _get_or_404(db, Product, product_id, "Product")
    changes = body.model_dump(exclude_unset=True)
    if "category_id" in changes:
        _get_or_404(db, Category, changes["category_id"], "Category")
    if "supplier_id" in changes:
        _get_or_404(db, Supplier, changes["supplier_id"], "Supplier")
    for field, value in changes.items():
        setattr(product, field, value)
    db.commit()
    return product


# ── sales ────────────────────────────────────────────────────
def _sale_out(db: Session, sale: Sale) -> SaleOut:
    items = db.scalars(select(SaleItem).where(SaleItem.sale_id == sale.sale_id).order_by(SaleItem.sale_item_id)).all()
    out = SaleOut.model_validate(sale)
    out.items = [item for item in items]  # type: ignore[misc]
    return out


@router.get("/sales", response_model=list[SaleOut])
def list_sales(
    date_from: Optional[dt.date] = None,
    date_to: Optional[dt.date] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _u: User = Depends(require_staff),
):
    """Sales headers, newest first (items are returned by GET /sales/{id})."""
    stmt = select(Sale).order_by(Sale.date.desc(), Sale.sale_id.desc())
    if date_from:
        stmt = stmt.where(Sale.date >= date_from)
    if date_to:
        stmt = stmt.where(Sale.date <= date_to)
    return [SaleOut.model_validate(s) for s in db.scalars(stmt.limit(limit).offset(offset)).all()]


@router.get("/sales/{sale_id}", response_model=SaleOut)
def get_sale(sale_id: int, db: Session = Depends(get_db), _u: User = Depends(require_staff)):
    return _sale_out(db, _get_or_404(db, Sale, sale_id, "Sale"))


@router.post("/sales", response_model=SaleOut, status_code=status.HTTP_201_CREATED)
def create_sale(body: SaleCreate, db: Session = Depends(get_db), _u: User = Depends(require_staff)):
    """Record a sale. Stock is reduced and one ledger row is written per product."""
    wanted: dict[int, int] = {}
    for line in body.items:
        wanted[line.product_id] = wanted.get(line.product_id, 0) + line.quantity

    products = {
        p.product_id: p
        for p in db.scalars(select(Product).where(Product.product_id.in_(wanted)).with_for_update()).all()
    }
    missing = sorted(set(wanted) - set(products))
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown product id(s): {missing}")
    short = [{"product_id": pid, "requested": qty, "available": products[pid].current_stock}
             for pid, qty in wanted.items() if products[pid].current_stock < qty]
    if short:
        raise HTTPException(status.HTTP_409_CONFLICT, {"message": "Insufficient stock", "items": short})

    now = utcnow()
    sale = Sale(date=now.date(), customer_id=body.customer_id, total_amount=0.0)
    db.add(sale)
    db.flush()
    total = 0.0
    for pid, qty in wanted.items():
        product = products[pid]
        db.add(SaleItem(sale_id=sale.sale_id, product_id=pid, quantity=qty, selling_price=product.selling_price))
        db.add(InventoryTransaction(product_id=pid, transaction_type=TxType.SALE, quantity=-qty, timestamp=now))
        product.current_stock -= qty
        total += qty * product.selling_price
    sale.total_amount = round(total, 2)
    db.commit()
    return _sale_out(db, sale)


# ── purchases ────────────────────────────────────────────────
def _purchase_out(db: Session, purchase: Purchase) -> PurchaseOut:
    items = db.scalars(select(PurchaseItem).where(PurchaseItem.purchase_id == purchase.purchase_id)
                       .order_by(PurchaseItem.purchase_item_id)).all()
    out = PurchaseOut.model_validate(purchase)
    out.items = [item for item in items]  # type: ignore[misc]
    return out


@router.get("/purchases", response_model=list[PurchaseOut])
def list_purchases(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _u: User = Depends(require_staff),
):
    stmt = select(Purchase).order_by(Purchase.date.desc(), Purchase.purchase_id.desc())
    if status_filter:
        stmt = stmt.where(Purchase.status == status_filter)
    return [_purchase_out(db, p) for p in db.scalars(stmt.limit(limit).offset(offset)).all()]


@router.get("/purchases/{purchase_id}", response_model=PurchaseOut)
def get_purchase(purchase_id: int, db: Session = Depends(get_db), _u: User = Depends(require_staff)):
    return _purchase_out(db, _get_or_404(db, Purchase, purchase_id, "Purchase"))


@router.post("/purchases", response_model=PurchaseOut, status_code=status.HTTP_201_CREATED)
def create_purchase(body: PurchaseCreate, db: Session = Depends(get_db), _u: User = Depends(require_manager)):
    """Create a purchase order. Lines must respect each product's minimum order quantity and pack size."""
    supplier = _get_or_404(db, Supplier, body.supplier_id, "Supplier")
    errors: list[str] = []
    lines: list[tuple[Product, int, float]] = []
    for line in body.items:
        product = db.get(Product, line.product_id)
        if product is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Product {line.product_id} not found")
        if line.quantity < product.min_order_qty:
            errors.append(f"{product.sku}: quantity {line.quantity} is below the minimum order quantity "
                          f"{product.min_order_qty}")
        if line.quantity % product.pack_size != 0:
            errors.append(f"{product.sku}: quantity {line.quantity} is not a multiple of pack size "
                          f"{product.pack_size}")
        price = line.purchase_price
        if price is None:
            price = round(product.cost * supplier.cost_multiplier, 2)
        lines.append((product, line.quantity, price))
    if errors:
        raise HTTPException(422, errors)

    today = utcnow().date()
    expected = body.expected_date or today + dt.timedelta(days=max(1, round(supplier.lead_time_mean_days)))
    purchase = Purchase(supplier_id=supplier.supplier_id, date=today, expected_date=expected,
                        status=PurchaseStatus.ORDERED)
    db.add(purchase)
    db.flush()
    for product, qty, price in lines:
        db.add(PurchaseItem(purchase_id=purchase.purchase_id, product_id=product.product_id,
                            quantity=qty, purchase_price=price))
    db.commit()
    return _purchase_out(db, purchase)


@router.post("/purchases/{purchase_id}/receive", response_model=PurchaseOut)
def receive_purchase(purchase_id: int, db: Session = Depends(get_db), _u: User = Depends(require_manager)):
    """Goods arrived: add the quantities to stock and write ledger rows."""
    purchase = _get_or_404(db, Purchase, purchase_id, "Purchase")
    if purchase.status != PurchaseStatus.ORDERED:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Purchase is already {purchase.status}")
    now = utcnow()
    items = db.scalars(select(PurchaseItem).where(PurchaseItem.purchase_id == purchase_id)).all()
    for item in items:
        product = db.scalar(select(Product).where(Product.product_id == item.product_id).with_for_update())
        product.current_stock += item.quantity
        db.add(InventoryTransaction(product_id=item.product_id, transaction_type=TxType.PURCHASE_RECEIPT,
                                    quantity=item.quantity, timestamp=now))
    purchase.status = PurchaseStatus.RECEIVED
    purchase.received_date = now.date()
    db.commit()
    return _purchase_out(db, purchase)


@router.post("/purchases/{purchase_id}/cancel", response_model=PurchaseOut)
def cancel_purchase(purchase_id: int, db: Session = Depends(get_db), _u: User = Depends(require_manager)):
    purchase = _get_or_404(db, Purchase, purchase_id, "Purchase")
    if purchase.status != PurchaseStatus.ORDERED:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Purchase is already {purchase.status}")
    purchase.status = PurchaseStatus.CANCELLED
    db.commit()
    return _purchase_out(db, purchase)


# ── stock ────────────────────────────────────────────────────
def stock_status(stock: int, safety: float, reorder: float, cover: Optional[float]) -> str:
    """Classify one product. ``cover`` = days of cover at the recent average demand."""
    if stock <= 0:
        return "stockout"
    if stock < safety:
        return "critical"
    if stock < reorder:
        return "low"
    if cover is not None and cover > settings.dsi_max:
        return "overstock"
    return "healthy"


@router.get("/stock", response_model=list[StockRow])
def stock_overview(
    status_filter: Optional[str] = Query(default=None, alias="status",
                                         description="stockout | critical | low | overstock | healthy"),
    db: Session = Depends(get_db),
    _u: User = Depends(require_staff),
):
    """Current stock for every product with days of cover and a status label."""
    last_day = db.scalar(select(func.max(Sale.date)))
    demand: dict[int, float] = {}
    if last_day is not None:
        rows = db.execute(
            select(SaleItem.product_id, func.sum(SaleItem.quantity))
            .join(Sale, Sale.sale_id == SaleItem.sale_id)
            .where(Sale.date > last_day - dt.timedelta(days=30))
            .group_by(SaleItem.product_id)
        ).all()
        demand = {pid: float(qty) / 30.0 for pid, qty in rows}

    result: list[StockRow] = []
    for p in db.scalars(select(Product).order_by(Product.product_id)).all():
        avg = demand.get(p.product_id)
        cover = round(p.current_stock / avg, 1) if avg else None
        label = stock_status(p.current_stock, p.safety_stock, p.reorder_point, cover)
        if status_filter and label != status_filter:
            continue
        result.append(StockRow(product_id=p.product_id, sku=p.sku, name=p.name, current_stock=p.current_stock,
                               reorder_point=p.reorder_point, safety_stock=p.safety_stock, max_stock=p.max_stock,
                               avg_daily_demand_30d=round(avg, 2) if avg else None,
                               days_of_cover=cover, status=label))
    return result


@router.get("/stock/{product_id}/transactions", response_model=list[TransactionOut])
def stock_transactions(product_id: int, limit: int = Query(default=100, ge=1, le=1000),
                       db: Session = Depends(get_db), _u: User = Depends(require_staff)):
    _get_or_404(db, Product, product_id, "Product")
    return db.scalars(
        select(InventoryTransaction).where(InventoryTransaction.product_id == product_id)
        .order_by(InventoryTransaction.timestamp.desc(), InventoryTransaction.transaction_id.desc()).limit(limit)
    ).all()


@router.post("/stock/adjust", response_model=ProductOut)
def adjust_stock(body: AdjustRequest, db: Session = Depends(get_db), _u: User = Depends(require_manager)):
    """Manual correction (stock count, damage, found stock). Writes an 'adjustment' ledger row."""
    product = db.scalar(select(Product).where(Product.product_id == body.product_id).with_for_update())
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Product {body.product_id} not found")
    if product.current_stock + body.quantity < 0:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"Adjustment would make stock negative (current stock {product.current_stock})")
    product.current_stock += body.quantity
    db.add(InventoryTransaction(product_id=product.product_id, transaction_type=TxType.ADJUSTMENT,
                                quantity=body.quantity, timestamp=utcnow()))
    db.commit()
    return product
