"""Supplier risk scoring — rule-based + ML composite scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..utils.logging_config import get_logger

log = get_logger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class RiskScore:
    """Composite risk score for a supplier."""
    supplier_id: str
    overall_score: float  # 0-1, higher = riskier
    delivery_risk: float  # 0-1
    quality_risk: float  # 0-1
    financial_risk: float  # 0-1
    geographic_risk: float  # 0-1
    concentration_risk: float  # 0-1, single-source dependency
    risk_level: str  # "low", "medium", "high", "critical"
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SupplierMetrics:
    """Input metrics for risk scoring."""
    supplier_id: str
    on_time_rate: float = 0.9  # 0-1
    defect_rate: float = 0.02  # 0-1
    lead_time_mean: float = 7.0
    lead_time_std: float = 2.0
    financial_score: float = 0.8  # 0-1 (1=healthy)
    geographic_zone: str = "domestic"  # domestic, regional, overseas
    num_products_supplied: int = 1
    total_products_in_category: int = 10
    years_relationship: int = 3
    capacity_utilization: float = 0.7  # 0-1
    recent_incidents: int = 0


class SupplierRiskScorer:
    """Multi-factor supplier risk scoring with rule-based + ML components.

    Replaces DoWhy causal inference with a transparent, auditable scoring model.
    """

    # Weight configuration
    WEIGHTS = {
        "delivery": 0.30,
        "quality": 0.25,
        "financial": 0.20,
        "geographic": 0.10,
        "concentration": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights is not None:
            required_keys = {"delivery", "quality", "financial", "geographic", "concentration"}
            missing = required_keys - set(weights.keys())
            if missing:
                raise ValueError(f"Custom weights missing required keys: {missing}")
            if any(v < 0 for v in weights.values()):
                raise ValueError("All weight values must be non-negative")
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 0.05:
                log.warning(
                    "risk_weights_sum_not_one",
                    weight_sum=weight_sum,
                    msg="Weights should sum to ~1.0; results may be unexpected",
                )
        self._weights = weights or self.WEIGHTS
        self._ml_model = None
        self._ml_trained = False

    def score_supplier(self, metrics: SupplierMetrics) -> RiskScore:
        """Calculate composite risk score for a supplier."""
        # Rule-based component scores (each 0-1, higher = riskier)
        delivery_risk = self._score_delivery(metrics)
        quality_risk = self._score_quality(metrics)
        financial_risk = self._score_financial(metrics)
        geographic_risk = self._score_geographic(metrics)
        concentration_risk = self._score_concentration(metrics)

        # Weighted composite
        composite = (
            self._weights["delivery"] * delivery_risk
            + self._weights["quality"] * quality_risk
            + self._weights["financial"] * financial_risk
            + self._weights["geographic"] * geographic_risk
            + self._weights["concentration"] * concentration_risk
        )

        # ML adjustment if available
        if self._ml_trained and self._ml_model is not None:
            ml_risk = self._ml_predict(metrics)
            # Blend: 70% rule-based, 30% ML
            composite = 0.7 * composite + 0.3 * ml_risk

        composite = min(1.0, max(0.0, composite))

        # Risk level classification
        if composite >= 0.75:
            level = "critical"
        elif composite >= 0.50:
            level = "high"
        elif composite >= 0.25:
            level = "medium"
        else:
            level = "low"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, delivery_risk, quality_risk, financial_risk, geographic_risk, concentration_risk
        )

        return RiskScore(
            supplier_id=metrics.supplier_id,
            overall_score=round(composite, 3),
            delivery_risk=round(delivery_risk, 3),
            quality_risk=round(quality_risk, 3),
            financial_risk=round(financial_risk, 3),
            geographic_risk=round(geographic_risk, 3),
            concentration_risk=round(concentration_risk, 3),
            risk_level=level,
            factors={
                "on_time_rate": metrics.on_time_rate,
                "defect_rate": metrics.defect_rate,
                "lead_time_variability": metrics.lead_time_std / max(metrics.lead_time_mean, 1),
                "financial_score": metrics.financial_score,
                "capacity_utilization": metrics.capacity_utilization,
                "recent_incidents": metrics.recent_incidents,
            },
            recommendations=recommendations,
        )

    def score_all(self, suppliers: List[SupplierMetrics]) -> List[RiskScore]:
        """Score all suppliers and return sorted by risk (highest first)."""
        scores = [self.score_supplier(s) for s in suppliers]
        scores.sort(key=lambda s: -s.overall_score)
        return scores

    def train_ml_model(self, historical_data: List[Dict], seed: int = 42) -> float:
        """Train ML risk model on historical supplier performance data.

        Args:
            historical_data: List of dicts with supplier metrics + 'disrupted' bool label
            seed: Random seed

        Returns:
            Training accuracy score
        """
        if not HAS_SKLEARN or len(historical_data) < 20:
            reason = "sklearn_unavailable" if not HAS_SKLEARN else "insufficient_data"
            log.info("ml_risk_skipped", reason=reason)
            return 0.0

        X = []
        y = []
        for record in historical_data:
            features = [
                record.get("on_time_rate", 0.9),
                record.get("defect_rate", 0.02),
                record.get("lead_time_std", 2) / max(record.get("lead_time_mean", 7), 1),
                record.get("financial_score", 0.8),
                record.get("capacity_utilization", 0.7),
                record.get("recent_incidents", 0),
                record.get("years_relationship", 3),
            ]
            X.append(features)
            y.append(1 if record.get("disrupted", False) else 0)

        X = np.array(X)
        y = np.array(y)

        # Need at least two classes for meaningful risk prediction
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            log.info("ml_risk_skipped", reason="single_class", unique_classes=unique_classes.tolist())
            return 0.0

        # Disable stratification when minority class is too small for a split
        min_class_count = int(class_counts.min())
        use_stratify = min_class_count >= 2
        if not use_stratify:
            log.info("ml_risk_stratify_disabled", min_class_count=min_class_count)

        # Split into train/test sets (80/20) to avoid overfitting
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y if use_stratify else None,
        )

        # Verify training set still has both classes after split
        if len(np.unique(y_train)) < 2:
            log.info("ml_risk_skipped", reason="single_class_after_split")
            return 0.0

        self._ml_model = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=seed
        )
        self._ml_model.fit(X_train, y_train)
        self._ml_trained = True

        accuracy = float(self._ml_model.score(X_test, y_test))
        log.info("ml_risk_trained", accuracy=accuracy, samples=len(y), test_samples=len(y_test))
        return accuracy

    def generate_synthetic_history(self, n_suppliers: int = 100, seed: int = 42) -> List[Dict]:
        """Generate synthetic historical supplier data for ML training."""
        rng = np.random.default_rng(seed)
        data = []
        for i in range(n_suppliers):
            on_time = float(np.clip(rng.normal(0.88, 0.08), 0.5, 0.99))
            defect = float(np.clip(rng.normal(0.03, 0.02), 0.001, 0.15))
            lt_mean = float(rng.uniform(3, 21))
            lt_std = float(rng.uniform(0.5, 5))
            financial = float(np.clip(rng.normal(0.75, 0.15), 0.2, 1.0))
            cap_util = float(rng.uniform(0.3, 0.95))
            incidents = int(rng.poisson(0.5))
            years = int(rng.uniform(1, 15))

            # Disruption probability based on risk factors
            disruption_prob = (
                0.3 * (1 - on_time)
                + 0.2 * defect * 10
                + 0.2 * (1 - financial)
                + 0.15 * min(1, incidents / 3)
                + 0.15 * min(1, lt_std / lt_mean)
            )
            disruption_prob = float(np.clip(disruption_prob, 0.0, 1.0))
            disrupted = bool(rng.random() < disruption_prob)

            data.append({
                "supplier_id": f"SUP-{i+1:04d}",
                "on_time_rate": on_time,
                "defect_rate": defect,
                "lead_time_mean": lt_mean,
                "lead_time_std": lt_std,
                "financial_score": financial,
                "capacity_utilization": cap_util,
                "recent_incidents": incidents,
                "years_relationship": years,
                "disrupted": disrupted,
            })
        return data

    def _score_delivery(self, m: SupplierMetrics) -> float:
        """Delivery risk: based on on-time rate and lead time variability."""
        otd_risk = max(0, 1 - m.on_time_rate)  # 0.9 -> 0.1 risk
        lt_cv = m.lead_time_std / max(m.lead_time_mean, 1)
        lt_risk = min(1.0, lt_cv)  # CV > 1.0 = max risk
        return 0.6 * otd_risk + 0.4 * lt_risk

    def _score_quality(self, m: SupplierMetrics) -> float:
        """Quality risk: based on defect rate and recent incidents."""
        defect_risk = min(1.0, m.defect_rate * 10)  # 0.1 = max risk
        incident_risk = min(1.0, m.recent_incidents / 3)
        return 0.7 * defect_risk + 0.3 * incident_risk

    def _score_financial(self, m: SupplierMetrics) -> float:
        """Financial risk: inverse of financial health score."""
        base = max(0, 1 - m.financial_score)
        # High capacity utilization is a warning sign
        cap_stress = max(0, m.capacity_utilization - 0.85) * 5
        return min(1.0, base + cap_stress * 0.3)

    def _score_geographic(self, m: SupplierMetrics) -> float:
        """Geographic risk: based on location zone."""
        zone_scores = {
            "domestic": 0.1,
            "regional": 0.3,
            "overseas": 0.6,
        }
        score = zone_scores.get(m.geographic_zone)
        if score is None:
            log.warning(
                "unknown_geographic_zone",
                supplier_id=m.supplier_id,
                zone=m.geographic_zone,
                fallback=0.5,
            )
            return 0.5
        return score

    def _score_concentration(self, m: SupplierMetrics) -> float:
        """Concentration risk: dependency on this supplier."""
        if m.total_products_in_category <= 0:
            return 0.5
        ratio = m.num_products_supplied / m.total_products_in_category
        return min(1.0, ratio * 1.5)

    def _ml_predict(self, m: SupplierMetrics) -> float:
        """Get ML-predicted disruption probability."""
        if not self._ml_model:
            return 0.5
        features = np.array([[
            m.on_time_rate, m.defect_rate,
            m.lead_time_std / max(m.lead_time_mean, 1),
            m.financial_score, m.capacity_utilization,
            m.recent_incidents, m.years_relationship,
        ]])
        prob = self._ml_model.predict_proba(features)[0]
        if len(prob) < 2:
            # Single-class model should not be used — fall back to neutral
            return 0.5
        # Multi-class: find the index of the disrupted class (label=1)
        classes = list(self._ml_model.classes_)
        if 1 in classes:
            return float(prob[classes.index(1)])
        return float(prob[-1])

    def _generate_recommendations(
        self, m: SupplierMetrics, delivery: float, quality: float,
        financial: float, geographic: float, concentration: float,
    ) -> List[str]:
        """Generate actionable recommendations based on risk factors."""
        recs = []
        if delivery > 0.4:
            recs.append(f"Delivery risk elevated ({delivery:.0%}) — consider backup supplier or safety stock increase")
        if quality > 0.3:
            recs.append(f"Quality risk ({quality:.0%}) — implement incoming inspection or audit")
        if financial > 0.4:
            recs.append(f"Financial risk ({financial:.0%}) — monitor supplier financial health quarterly")
        if geographic > 0.4:
            recs.append(f"Geographic risk ({geographic:.0%}) — develop regional sourcing alternative")
        if concentration > 0.5:
            recs.append(f"Concentration risk ({concentration:.0%}) — dual-source strategy recommended")
        if m.capacity_utilization > 0.9:
            recs.append("Supplier near capacity limit — pre-book capacity or find secondary source")
        return recs
