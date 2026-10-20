"""Anomaly detection using Isolation Forest (with statistical fallback)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import settings
from ..data.schemas import AlertSeverity, AnomalyRecord
from ..utils.logging_config import get_logger

log = get_logger(__name__)


class AnomalyDetector:
    """Detects demand anomalies, cost anomalies, and lead-time anomalies.

    Uses scikit-learn IsolationForest when available;
    falls back to Z-score detection otherwise.
    """

    def __init__(self) -> None:
        self._contamination = settings.isolation_contamination
        self._stats: Dict[str, dict] = {}  # product_id -> statistics
        self._trained = False
        self._use_sklearn = False

        try:
            from sklearn.ensemble import IsolationForest  # noqa: F401
            self._use_sklearn = True
        except ImportError:
            log.info("sklearn_unavailable", fallback="z-score")

        self._models: Dict[str, Any] = {}

    def train(self, data: pd.DataFrame) -> None:
        """Train anomaly detection models per product."""
        product_ids = data["product_id"].unique()

        for pid in product_ids:
            series = data[data["product_id"] == pid]["quantity"].values
            if len(series) < 10:
                continue

            self._stats[pid] = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series, ddof=1)),
                "median": float(np.median(series)),
                "q1": float(np.percentile(series, 25)),
                "q3": float(np.percentile(series, 75)),
                "iqr": float(np.percentile(series, 75) - np.percentile(series, 25)),
                "max": float(np.max(series)),
                "min": float(np.min(series)),
            }

            if self._use_sklearn:
                from sklearn.ensemble import IsolationForest

                model = IsolationForest(
                    contamination=self._contamination,
                    random_state=42,
                    n_estimators=100,
                )
                model.fit(series.reshape(-1, 1))
                self._models[pid] = model

        self._trained = True
        log.info("anomaly_detector_trained", products=len(self._stats))

    def detect(
        self,
        current_data: Dict[str, Any],
        max_products: int = 0,
    ) -> List[AnomalyRecord]:
        """Detect anomalies in current data point or snapshot.

        Args:
            current_data: Dict with product_id / daily_demand_avg / current_stock,
                          or a "products" list for multi-product snapshots.
            max_products: Maximum products to check when scanning all trained
                          products (0 = unlimited).
        """
        anomalies: List[AnomalyRecord] = []

        if not self._trained:
            return anomalies

        # Build per-product demand and stock lookups when a "products" list is provided
        product_demand_map: Dict[str, float] = {}
        product_stock_map: Dict[str, float] = {}
        if "products" in current_data:
            for p in current_data["products"]:
                pid = p.get("product_id") or getattr(p, "product_id", None)
                if pid:
                    product_demand_map[pid] = (
                        p.get("daily_demand_avg")
                        if isinstance(p, dict)
                        else getattr(p, "daily_demand_avg", None)
                    ) or 0.0
                    product_stock_map[pid] = (
                        p.get("current_stock")
                        if isinstance(p, dict)
                        else getattr(p, "current_stock", 0)
                    ) or 0

        # Check across all trained products if no specific product given
        if "product_id" in current_data:
            product_ids = [current_data["product_id"]]
        else:
            product_ids = list(self._stats.keys())
            if max_products > 0:
                product_ids = product_ids[:max_products]

        for pid in product_ids:
            stats = self._stats.get(pid)
            if stats is None:
                continue

            # Use per-product demand: first from product_demand_map, then from
            # current_data (single-product mode), finally fall back to trained mean.
            demand = product_demand_map.get(
                pid, current_data.get("daily_demand_avg", stats["mean"])
            )

            # Demand spike detection — combine Z-score and IsolationForest
            z_score = abs(demand - stats["mean"]) / max(stats["std"], 0.01 * abs(stats["mean"]), 0.01)
            z_anomaly = z_score > 2.5

            # IsolationForest prediction (if trained for this product)
            if_anomaly = False
            if_score = 0.0
            if pid in self._models:
                if_model = self._models[pid]
                if_pred = if_model.predict(np.array([[demand]]))[0]  # -1 = anomaly
                if_raw = if_model.decision_function(np.array([[demand]]))[0]
                if_anomaly = if_pred == -1
                if_score = max(0.0, -if_raw)  # higher = more anomalous

            # Flag anomaly if either detector triggers
            if z_anomaly or if_anomaly:
                # Boost confidence when both detectors agree
                base_score = min(z_score / 5, 1.0)
                if z_anomaly and if_anomaly:
                    combined_score = min(1.0, base_score * 0.6 + if_score * 0.4 + 0.1)
                elif if_anomaly:
                    combined_score = min(1.0, max(0.3, if_score * 0.8))
                else:
                    combined_score = base_score

                severity = (
                    AlertSeverity.CRITICAL if z_score > 4 or combined_score > 0.8
                    else AlertSeverity.HIGH if z_score > 3 or combined_score > 0.6
                    else AlertSeverity.MEDIUM
                )
                detection_methods = []
                if z_anomaly:
                    detection_methods.append(f"z-score={z_score:.2f}")
                if if_anomaly:
                    detection_methods.append(f"IF-score={if_score:.2f}")

                anomalies.append(AnomalyRecord(
                    anomaly_type="demand_spike",
                    product_id=pid,
                    severity=severity,
                    score=round(combined_score, 3),
                    description=(
                        f"Demand anomaly ({', '.join(detection_methods)}): "
                        f"current={demand:.1f}, mean={stats['mean']:.1f}"
                    ),
                ))

            # Stock level anomaly — use per-product stock when available
            current_stock = product_stock_map.get(
                pid, current_data.get("current_stock", 0)
            )
            if current_stock > 0 and stats["mean"] > 0:
                dsi = current_stock / stats["mean"]
                if dsi > settings.dsi_max:
                    anomalies.append(AnomalyRecord(
                        anomaly_type="overstock",
                        product_id=pid,
                        severity=AlertSeverity.MEDIUM,
                        score=round(min(dsi / 100, 1.0), 3),
                        description=f"Overstock: DSI={dsi:.1f} days (max={settings.dsi_max})",
                    ))
                elif dsi < settings.dsi_min:
                    anomalies.append(AnomalyRecord(
                        anomaly_type="understock",
                        product_id=pid,
                        severity=AlertSeverity.HIGH,
                        score=round(1 - dsi / settings.dsi_min, 3),
                        description=f"Understock: DSI={dsi:.1f} days (min={settings.dsi_min})",
                    ))

        return anomalies

    def detect_batch(self, products: list) -> List[AnomalyRecord]:
        """Run detection across all products."""
        all_anomalies: List[AnomalyRecord] = []
        for product in products:
            data = {
                "product_id": product.product_id,
                "daily_demand_avg": product.daily_demand_avg,
                "current_stock": product.current_stock,
            }
            all_anomalies.extend(self.detect(data))
        return all_anomalies
