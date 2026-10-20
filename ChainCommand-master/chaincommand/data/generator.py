"""Synthetic supply chain data generator.

Incorporates ChainInsight-discovered patterns:
- 2.2x overstock pattern in certain categories
- Seasonal demand spikes
- Supplier reliability variance
- Lead time volatility
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from chaincommand.config import settings
from chaincommand.data.schemas import (
    Product,
    ProductCategory,
    Supplier,
)

# Realistic product templates per category
_PRODUCT_TEMPLATES = {
    ProductCategory.ELECTRONICS: [
        ("Wireless Earbuds", 15.0, 39.99),
        ("USB-C Hub", 8.0, 24.99),
        ("Power Bank 10K", 12.0, 29.99),
        ("Smart Watch Band", 3.0, 14.99),
        ("Bluetooth Speaker", 18.0, 49.99),
        ("Laptop Stand", 10.0, 34.99),
        ("Webcam HD", 14.0, 44.99),
        ("Wireless Mouse", 6.0, 19.99),
        ("Phone Case Premium", 2.0, 12.99),
        ("LED Desk Lamp", 9.0, 29.99),
    ],
    ProductCategory.FOOD: [
        ("Organic Coffee 1kg", 5.0, 14.99),
        ("Protein Bar Box", 8.0, 24.99),
        ("Olive Oil 500ml", 4.0, 12.99),
        ("Almond Butter", 3.5, 9.99),
        ("Granola Mix", 2.0, 7.99),
        ("Green Tea 100ct", 3.0, 11.99),
        ("Dried Mango Pack", 2.5, 8.99),
        ("Coconut Water 12pk", 6.0, 18.99),
        ("Dark Chocolate Bar", 1.5, 4.99),
        ("Quinoa 2kg", 4.0, 13.99),
    ],
    ProductCategory.PHARMA: [
        ("Vitamin D3 5000IU", 3.0, 12.99),
        ("Omega-3 Fish Oil", 5.0, 19.99),
        ("Probiotic 30ct", 4.0, 24.99),
        ("Zinc Tablets", 1.5, 8.99),
        ("Magnesium Citrate", 2.5, 14.99),
        ("Collagen Powder", 8.0, 29.99),
        ("Melatonin 5mg", 1.0, 7.99),
        ("Iron Supplement", 2.0, 11.99),
        ("Vitamin C 1000mg", 1.5, 9.99),
        ("Calcium + D3", 3.0, 13.99),
    ],
    ProductCategory.INDUSTRIAL: [
        ("Steel Bolt M10 100pk", 4.0, 12.99),
        ("Bearing 6205-2RS", 3.0, 9.99),
        ("Hydraulic Hose 1m", 8.0, 22.99),
        ("Safety Gloves 12pk", 5.0, 18.99),
        ("Cable Ties 500pk", 1.5, 6.99),
        ("Lubricant WD-40 400ml", 2.0, 7.99),
        ("Sandpaper Set", 1.0, 5.99),
        ("Drill Bit Set", 6.0, 19.99),
        ("Welding Rod 5kg", 7.0, 24.99),
        ("Pipe Fitting 1in", 2.5, 8.99),
    ],
    ProductCategory.APPAREL: [
        ("Cotton T-Shirt", 3.0, 14.99),
        ("Running Socks 6pk", 4.0, 16.99),
        ("Work Boots", 25.0, 79.99),
        ("Rain Jacket", 15.0, 49.99),
        ("Baseball Cap", 4.0, 14.99),
        ("Thermal Underwear", 8.0, 24.99),
        ("Cargo Pants", 12.0, 39.99),
        ("Fleece Hoodie", 10.0, 34.99),
        ("Wool Scarf", 6.0, 19.99),
        ("Leather Belt", 5.0, 22.99),
    ],
}

_SUPPLIER_NAMES = [
    "GlobalTech Supply", "Pacific Rim Trading", "EuroLogistics GmbH",
    "ShenZhen Direct", "Midwest Materials", "Atlantic Imports",
    "Nordic Components", "Sahara Trading Co", "Southern Cross Supply",
    "Maple Leaf Logistics", "Rhine Valley Parts", "Tokyo Express Co",
    "Mumbai Manufacturing", "Santiago Exports", "Cairo Distribution",
    "Seoul Smart Supply", "Berlin Precision", "Sydney Freight",
    "Dubai Hub Trading", "Vancouver Components",
]


def generate_products(
    n: int | None = None,
    rng: Optional[random.Random] = None,
) -> List[Product]:
    """Generate a list of synthetic products."""
    n = n or settings.num_products
    rng = rng or random.Random()
    products = []
    categories = list(ProductCategory)

    for i in range(n):
        cat = categories[i % len(categories)]
        templates = _PRODUCT_TEMPLATES[cat]
        tmpl = templates[i % len(templates)]
        name, unit_cost, selling_price = tmpl

        # Add variant suffix for duplicates
        variant = i // len(templates)
        if variant > 0:
            name = f"{name} v{variant + 1}"

        lead_time = rng.randint(3, 21)
        daily_demand_avg = rng.uniform(10, 200)
        daily_demand_std = daily_demand_avg * rng.uniform(0.15, 0.45)

        # ChainInsight: 2.2x overstock pattern — some products over-stocked
        overstock_factor = 2.2 if rng.random() < 0.3 else 1.0
        safety_stock = daily_demand_std * np.sqrt(lead_time) * 1.65
        reorder_point = daily_demand_avg * lead_time + safety_stock

        products.append(Product(
            product_id=f"PRD-{i + 1:04d}",
            name=name,
            category=cat,
            unit_cost=unit_cost * rng.uniform(0.9, 1.1),
            selling_price=selling_price * rng.uniform(0.95, 1.05),
            lead_time_days=lead_time,
            min_order_qty=rng.choice([50, 100, 200, 500]),
            current_stock=daily_demand_avg * lead_time * overstock_factor,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            daily_demand_avg=daily_demand_avg,
            daily_demand_std=daily_demand_std,
        ))

    return products


def generate_suppliers(
    n: int | None = None,
    rng: Optional[random.Random] = None,
) -> List[Supplier]:
    """Generate a list of synthetic suppliers."""
    n = n or settings.num_suppliers
    rng = rng or random.Random()
    suppliers = []

    for i in range(n):
        name = _SUPPLIER_NAMES[i % len(_SUPPLIER_NAMES)]
        variant = i // len(_SUPPLIER_NAMES)
        if variant > 0:
            name = f"{name} #{variant + 1}"

        reliability = np.clip(rng.gauss(0.85, 0.1), 0.5, 0.99)
        suppliers.append(Supplier(
            supplier_id=f"SUP-{i + 1:04d}",
            name=name,
            reliability_score=round(reliability, 3),
            lead_time_mean=rng.uniform(3, 14),
            lead_time_std=rng.uniform(0.5, 4),
            cost_multiplier=rng.uniform(0.85, 1.25),
            capacity=rng.uniform(5000, 50000),
            defect_rate=round(np.clip(rng.gauss(0.02, 0.01), 0.001, 0.1), 4),
            on_time_rate=round(np.clip(rng.gauss(0.9, 0.08), 0.6, 0.99), 3),
        ))

    return suppliers


def assign_suppliers(
    products: List[Product],
    suppliers: List[Supplier],
    rng: Optional[random.Random] = None,
) -> List[Supplier]:
    """Assign each product to 1-3 suppliers."""
    rng = rng or random.Random()
    for product in products:
        n_suppliers = rng.randint(1, min(3, len(suppliers)))
        chosen = rng.sample(suppliers, n_suppliers)
        for s in chosen:
            if product.product_id not in s.products:
                s.products.append(product.product_id)
    return suppliers


def generate_demand_history(
    products: List[Product],
    days: int | None = None,
    rng: Optional[random.Random] = None,
) -> pd.DataFrame:
    """Generate daily demand history with realistic patterns.

    Patterns included:
    - Weekly seasonality (lower weekends)
    - Monthly seasonality
    - Annual seasonality (holiday spikes)
    - Random promotions (~5% of days)
    - Trend component
    - Noise
    """
    days = days or settings.history_days
    rng = rng or random.Random()
    end_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start_date, end_date, freq="D")

    records: list[dict] = []
    for product in products:
        avg = product.daily_demand_avg
        std = product.daily_demand_std
        trend_slope = rng.uniform(-0.05, 0.15)  # slight upward trend usually

        for i, date in enumerate(dates):
            # Weekly pattern (weekend dip)
            dow = date.dayofweek
            weekly_factor = 1.0 if dow < 5 else 0.7

            # Monthly pattern
            monthly_factor = 1.0 + 0.1 * np.sin(2 * np.pi * date.day / 30)

            # Annual seasonality (Q4 spike for electronics/apparel)
            annual_factor = 1.0
            if product.category in (ProductCategory.ELECTRONICS, ProductCategory.APPAREL):
                if date.month in (11, 12):
                    annual_factor = 1.5
                elif date.month in (1, 2):
                    annual_factor = 0.8

            # Trend
            trend = 1.0 + trend_slope * (i / days)

            # Promotion spike
            is_promo = rng.random() < 0.05
            promo_factor = rng.uniform(1.5, 3.0) if is_promo else 1.0

            # Holiday — demand varies by category
            is_holiday = date.month == 12 and date.day in range(20, 32)
            if is_holiday:
                if product.category in (ProductCategory.ELECTRONICS, ProductCategory.APPAREL):
                    holiday_factor = rng.uniform(1.3, 1.5)  # gift-oriented categories surge
                elif product.category == ProductCategory.FOOD:
                    holiday_factor = rng.uniform(1.2, 1.4)  # holiday meals & treats
                elif product.category == ProductCategory.INDUSTRIAL:
                    holiday_factor = rng.uniform(0.6, 0.8)  # factories slow down
                else:
                    holiday_factor = rng.uniform(1.1, 1.3)  # mild holiday uplift
            else:
                holiday_factor = 1.0

            # Combine
            base = avg * weekly_factor * monthly_factor * annual_factor * trend * promo_factor * holiday_factor
            noise = rng.gauss(0, std)
            quantity = max(0, base + noise)

            records.append({
                "date": date,
                "product_id": product.product_id,
                "quantity": round(quantity, 1),
                "is_promotion": is_promo,
                "is_holiday": is_holiday,
                "temperature": round(15 + 10 * np.sin(2 * np.pi * (date.timetuple().tm_yday - 80) / 365), 1),
                "day_of_week": dow,
                "month": date.month,
            })

    return pd.DataFrame(records)


def generate_all(
    rng: Optional[random.Random] = None,
) -> Tuple[List[Product], List[Supplier], pd.DataFrame]:
    """Generate complete synthetic dataset."""
    products = generate_products(rng=rng)
    suppliers = generate_suppliers(rng=rng)
    suppliers = assign_suppliers(products, suppliers, rng=rng)
    demand_df = generate_demand_history(products, rng=rng)
    return products, suppliers, demand_df
