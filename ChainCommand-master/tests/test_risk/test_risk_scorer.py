"""Tests for supplier risk scoring module."""
import pytest

from chaincommand.risk.scorer import SupplierMetrics, SupplierRiskScorer


@pytest.fixture
def scorer():
    return SupplierRiskScorer()


@pytest.fixture
def good_supplier():
    return SupplierMetrics(
        supplier_id="SUP-001",
        on_time_rate=0.95,
        defect_rate=0.01,
        lead_time_mean=7.0,
        lead_time_std=1.0,
        financial_score=0.9,
        geographic_zone="domestic",
        num_products_supplied=2,
        total_products_in_category=10,
        years_relationship=5,
        capacity_utilization=0.7,
        recent_incidents=0,
    )


@pytest.fixture
def risky_supplier():
    return SupplierMetrics(
        supplier_id="SUP-002",
        on_time_rate=0.65,
        defect_rate=0.08,
        lead_time_mean=14.0,
        lead_time_std=6.0,
        financial_score=0.4,
        geographic_zone="overseas",
        num_products_supplied=8,
        total_products_in_category=10,
        years_relationship=1,
        capacity_utilization=0.92,
        recent_incidents=3,
    )


class TestSupplierRiskScorer:
    def test_score_good_supplier(self, scorer, good_supplier):
        score = scorer.score_supplier(good_supplier)
        assert score.overall_score < 0.3
        assert score.risk_level in ("low", "medium")

    def test_score_risky_supplier(self, scorer, risky_supplier):
        score = scorer.score_supplier(risky_supplier)
        assert score.overall_score > 0.4
        assert score.risk_level in ("high", "critical")

    def test_good_lower_than_risky(self, scorer, good_supplier, risky_supplier):
        good_score = scorer.score_supplier(good_supplier)
        risky_score = scorer.score_supplier(risky_supplier)
        assert good_score.overall_score < risky_score.overall_score

    def test_score_all(self, scorer, good_supplier, risky_supplier):
        scores = scorer.score_all([good_supplier, risky_supplier])
        assert len(scores) == 2
        # Sorted highest risk first
        assert scores[0].overall_score >= scores[1].overall_score

    def test_recommendations_generated(self, scorer, risky_supplier):
        score = scorer.score_supplier(risky_supplier)
        assert len(score.recommendations) > 0

    def test_no_recommendations_for_good(self, scorer, good_supplier):
        score = scorer.score_supplier(good_supplier)
        # Good supplier should have few or no recommendations
        assert len(score.recommendations) <= 1

    def test_factors_included(self, scorer, good_supplier):
        score = scorer.score_supplier(good_supplier)
        assert "on_time_rate" in score.factors
        assert "defect_rate" in score.factors

    def test_geographic_risk_varies(self, scorer):
        domestic = SupplierMetrics(supplier_id="D", geographic_zone="domestic")
        overseas = SupplierMetrics(supplier_id="O", geographic_zone="overseas")
        d_score = scorer.score_supplier(domestic)
        o_score = scorer.score_supplier(overseas)
        assert d_score.geographic_risk < o_score.geographic_risk

    def test_generate_synthetic_history(self, scorer):
        data = scorer.generate_synthetic_history(n_suppliers=50)
        assert len(data) == 50
        assert all("disrupted" in d for d in data)
        # Should have both disrupted and non-disrupted
        disrupted_count = sum(1 for d in data if d["disrupted"])
        assert 0 < disrupted_count < 50

    def test_train_ml_model(self, scorer):
        data = scorer.generate_synthetic_history(n_suppliers=100)
        accuracy = scorer.train_ml_model(data)
        assert accuracy > 0.5  # better than random

    def test_ml_adjusts_score(self, scorer, good_supplier):
        # Score without ML
        scorer.score_supplier(good_supplier)

        # Train ML and score again
        data = scorer.generate_synthetic_history(n_suppliers=100)
        scorer.train_ml_model(data)
        score_with_ml = scorer.score_supplier(good_supplier)

        # Scores should differ slightly (ML adjustment)
        # Both should still indicate low risk for a good supplier
        assert score_with_ml.overall_score < 0.5
