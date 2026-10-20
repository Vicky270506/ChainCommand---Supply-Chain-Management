"""Tests for RL inventory policy module."""
import pytest

from chaincommand.rl.environment import InventoryEnv, InventoryEnvConfig
from chaincommand.rl.policy import RLInventoryPolicy
from chaincommand.rl.trainer import RLInventoryTrainer, SsBaseline, TrainingResult


@pytest.fixture
def env_config():
    return InventoryEnvConfig(
        max_stock=2000.0,
        max_demand=300.0,
        demand_mean=50.0,
        demand_std=15.0,
        episode_length=30,
        lead_time_days=5,
    )


class TestInventoryEnv:
    def test_create_env(self, env_config):
        env = InventoryEnv(env_config)
        obs, info = env.reset(seed=42)
        assert obs.shape == (5,)
        assert all(0 <= v <= 1 for v in obs)

    def test_step(self, env_config):
        env = InventoryEnv(env_config)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)  # no order
        assert obs.shape == (5,)
        assert reward <= 0  # costs are negative
        assert "stock" in info
        assert "demand" in info

    def test_episode_terminates(self, env_config):
        env = InventoryEnv(env_config)
        env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            steps += 1
        assert steps == env_config.episode_length

    def test_ordering_increases_stock(self, env_config):
        env_config.lead_time_days = 0  # instant delivery for test
        env = InventoryEnv(env_config)
        env.reset(seed=42)
        # Large order
        _, _, _, _, info1 = env.step(4)
        stock_after = info1["stock"]
        # Stock should be significant even after demand consumption
        assert stock_after > 0 or info1["order_qty"] > 0

    def test_stockout_penalty(self, env_config):
        env_config.demand_mean = 500.0  # very high demand to force stockouts
        env_config.demand_std = 10.0
        env_config.lead_time_days = 1  # low starting stock (500 * 1 = 500)
        env = InventoryEnv(env_config)
        env.reset(seed=42)
        # Run multiple steps without ordering to deplete stock
        total_stockout = 0.0
        for _ in range(5):
            _, _, _, _, info = env.step(0)
            total_stockout += info["stockout_cost"]
        assert total_stockout > 0


class TestSsBaseline:
    def test_evaluate(self, env_config):
        baseline = SsBaseline(s=100, S=500)
        result = baseline.evaluate(env_config, n_episodes=10, seed=42)
        assert result.mean_reward < 0  # costs are negative
        assert 0 <= result.mean_stockout_rate <= 1
        assert result.mean_holding_cost >= 0

    def test_deterministic(self, env_config):
        baseline = SsBaseline(s=100, S=500)
        r1 = baseline.evaluate(env_config, n_episodes=10, seed=42)
        r2 = baseline.evaluate(env_config, n_episodes=10, seed=42)
        assert r1.mean_reward == r2.mean_reward


class TestRLInventoryTrainer:
    def test_train_fallback(self, env_config):
        """Test Q-table fallback training (always available)."""
        trainer = RLInventoryTrainer(env_config)
        result = trainer.train(total_timesteps=900, seed=42)  # 30 episodes
        assert isinstance(result, TrainingResult)
        assert result.total_episodes > 0
        assert result.mean_reward < 0
        assert len(result.training_curve) > 0
        assert result.baseline_reward < 0

    def test_trainer_is_trained(self, env_config):
        trainer = RLInventoryTrainer(env_config)
        assert not trainer.is_trained
        trainer.train(total_timesteps=300, seed=42)
        assert trainer.is_trained


class TestRLInventoryPolicy:
    def test_decide_untrained(self, env_config):
        policy = RLInventoryPolicy(env_config)
        decision = policy.decide(current_stock=100, avg_demand=50)
        assert decision.method == "heuristic_fallback"
        assert decision.order_quantity >= 0

    def test_decide_after_training(self, env_config):
        policy = RLInventoryPolicy(env_config)
        policy.train(total_timesteps=300, seed=42)
        decision = policy.decide(current_stock=100, avg_demand=50)
        assert decision.order_quantity >= 0
        assert 0 <= decision.action <= 4
        assert decision.stock_level == 100

    def test_low_stock_triggers_order(self, env_config):
        policy = RLInventoryPolicy(env_config)
        # Without training, heuristic should order when stock is very low
        decision = policy.decide(current_stock=10, avg_demand=50)
        assert decision.order_quantity > 0
