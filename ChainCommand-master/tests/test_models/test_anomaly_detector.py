"""Tests for anomaly detection model."""

from __future__ import annotations

import pytest

from chaincommand.models.anomaly_detector import AnomalyDetector


@pytest.fixture
def detector(sample_demand_df):
    d = AnomalyDetector()
    d.train(sample_demand_df)
    return d


class TestAnomalyDetector:
    def test_train_sets_trained(self, detector):
        assert detector._trained is True
        assert len(detector._stats) > 0

    def test_detect_spike(self, detector):
        # Inject a massive demand spike
        data = {
            "product_id": "PRD-0000",
            "daily_demand_avg": 999.0,  # way above normal ~20
            "current_stock": 100.0,
        }
        anomalies = detector.detect(data)
        assert len(anomalies) >= 1
        spike = [a for a in anomalies if a.anomaly_type == "demand_spike"]
        assert len(spike) > 0

    def test_detect_batch(self, detector, sample_products):
        anomalies = detector.detect_batch(sample_products)
        assert isinstance(anomalies, list)

    def test_untrained_returns_empty(self):
        d = AnomalyDetector()
        assert d.detect({"product_id": "P1"}) == []
