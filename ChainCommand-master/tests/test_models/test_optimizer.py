"""Tests for GA, DQN, and Hybrid optimizers."""

from __future__ import annotations

import pytest

from chaincommand.models.optimizer import DQNOptimizer, GeneticOptimizer, HybridOptimizer


@pytest.fixture
def product():
    from chaincommand.data.schemas import Product, ProductCategory

    return Product(
        product_id="PRD-opt01",
        name="Opt Test",
        category=ProductCategory.ELECTRONICS,
        unit_cost=10.0,
        selling_price=25.0,
        current_stock=500.0,
        reorder_point=100.0,
        safety_stock=50.0,
        daily_demand_avg=20.0,
        daily_demand_std=5.0,
        min_order_qty=100,
    )


class TestGeneticOptimizer:
    def test_optimize_returns_result(self, product):
        ga = GeneticOptimizer()
        result = ga.optimize(product, [])
        assert result.product_id == "PRD-opt01"
        assert result.method == "genetic_algorithm"
        assert result.recommended_order_qty >= product.min_order_qty
        assert result.recommended_reorder_point > 0
        assert result.recommended_safety_stock > 0


class TestDQNOptimizer:
    def test_train_and_decide(self, product):
        dqn = DQNOptimizer()
        dqn.train(product)
        assert dqn._trained is True
        result = dqn.decide(product)
        assert result.product_id == "PRD-opt01"

    def test_untrained_fallback(self, product):
        dqn = DQNOptimizer()
        result = dqn.decide(product)
        assert result.recommended_reorder_point == product.reorder_point


class TestHybridOptimizer:
    def test_optimize_blends(self, product):
        hybrid = HybridOptimizer()
        result = hybrid.optimize(product, [])
        assert result.method == "hybrid_ga_dqn"
        assert result.recommended_order_qty >= 0
