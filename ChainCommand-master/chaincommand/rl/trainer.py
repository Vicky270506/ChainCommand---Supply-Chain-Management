"""RL Inventory Trainer — PPO training with (s,S) baseline comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..utils.logging_config import get_logger
from .environment import HAS_GYM, InventoryEnv, InventoryEnvConfig

log = get_logger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


@dataclass
class TrainingResult:
    """Result of RL training."""
    total_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    training_curve: List[float] = field(default_factory=list)
    baseline_reward: float = 0.0  # (s,S) heuristic baseline
    improvement_pct: float = 0.0
    method: str = "ppo"


@dataclass
class BaselineResult:
    """Result of (s,S) heuristic baseline evaluation."""
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_stockout_rate: float = 0.0
    mean_holding_cost: float = 0.0


class SsBaseline:
    """(s,S) heuristic baseline for comparison.

    Policy: When stock drops below reorder point s, order up to level S.
    """

    def __init__(self, s: float, S: float):
        self.s = s  # reorder point
        self.S = S  # order-up-to level

    def evaluate(self, env_config: InventoryEnvConfig, n_episodes: int = 50, seed: int = 42) -> BaselineResult:
        """Evaluate (s,S) policy on the inventory environment.

        Uses lead-time simulation (pending orders queue) to match the RL
        agent's environment for a fair comparison.
        """
        rng = np.random.default_rng(seed)
        episode_rewards = []
        stockout_rates = []
        holding_costs = []

        for _ in range(n_episodes):
            stock = env_config.demand_mean * env_config.lead_time_days
            total_reward = 0.0
            stockout_days = 0
            pending_orders: list = []  # list of (arrival_day, qty)

            for day in range(env_config.episode_length):
                # Receive orders that have arrived
                arrived_qty = sum(q for d, q in pending_orders if d <= day)
                stock += arrived_qty
                pending_orders = [(d, q) for d, q in pending_orders if d > day]

                # (s,S) decision — order is placed but arrives after lead_time
                if stock <= self.s:
                    order_qty = max(0, self.S - stock)
                    ordering_cost = env_config.ordering_cost_fixed + order_qty * env_config.ordering_cost_per_unit
                    arrival_day = day + env_config.lead_time_days
                    pending_orders.append((arrival_day, order_qty))
                else:
                    order_qty = 0
                    ordering_cost = 0.0

                # Demand
                demand = max(0, rng.normal(env_config.demand_mean, env_config.demand_std))

                stockout = max(0, demand - stock)
                stock = max(0, stock - demand)
                stock = min(stock, env_config.max_stock)

                holding_cost = stock * env_config.holding_cost_per_unit / 365
                stockout_cost = stockout * env_config.stockout_cost_per_unit
                reward = -(holding_cost + stockout_cost + ordering_cost)
                total_reward += reward

                if stockout > 0:
                    stockout_days += 1
                holding_costs.append(holding_cost)

            episode_rewards.append(total_reward)
            stockout_rates.append(stockout_days / env_config.episode_length)

        return BaselineResult(
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            mean_stockout_rate=float(np.mean(stockout_rates)),
            mean_holding_cost=float(np.mean(holding_costs)),
        )


class RLInventoryTrainer:
    """Train PPO agent for inventory replenishment.

    Falls back to Q-table learning if stable-baselines3 is not installed.
    """

    def __init__(self, config: Optional[InventoryEnvConfig] = None):
        self.config = config or InventoryEnvConfig()
        self._model = None
        self._trained = False

    def train(
        self,
        total_timesteps: int = 50_000,
        seed: int = 42,
    ) -> TrainingResult:
        """Train the RL agent."""
        if HAS_SB3 and HAS_GYM:
            return self._train_ppo(total_timesteps, seed)
        return self._train_qtable(total_timesteps, seed)

    def _train_ppo(self, total_timesteps: int, seed: int) -> TrainingResult:
        """Train with Stable-Baselines3 PPO."""
        log.info("rl_training_start", method="ppo", timesteps=total_timesteps)

        env = DummyVecEnv([lambda: InventoryEnv(self.config)])

        self._model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            seed=seed,
            verbose=0,
        )

        # Track training progress
        training_curve: List[float] = []

        # Train in chunks to record learning curve
        chunk_size = max(1000, total_timesteps // 20)
        for i in range(0, total_timesteps, chunk_size):
            steps = min(chunk_size, total_timesteps - i)
            self._model.learn(total_timesteps=steps, reset_num_timesteps=False)

            # Evaluate current policy
            eval_reward = self._evaluate_policy(n_episodes=5, seed=seed + i)
            training_curve.append(eval_reward)

        self._trained = True

        # Final evaluation
        mean_reward, std_reward = self._evaluate_policy_stats(n_episodes=20, seed=seed)

        # Baseline comparison
        baseline = SsBaseline(
            s=self.config.demand_mean * self.config.lead_time_days * 0.5,
            S=self.config.demand_mean * self.config.lead_time_days * 1.5,
        )
        baseline_result = baseline.evaluate(self.config, n_episodes=20, seed=seed)

        improvement = 0.0
        if baseline_result.mean_reward < 0:
            improvement = (mean_reward - baseline_result.mean_reward) / abs(baseline_result.mean_reward) * 100

        log.info("rl_training_complete", method="ppo", mean_reward=mean_reward, improvement_pct=improvement)

        env.close()

        return TrainingResult(
            total_episodes=total_timesteps // self.config.episode_length,
            mean_reward=mean_reward,
            std_reward=std_reward,
            training_curve=training_curve,
            baseline_reward=baseline_result.mean_reward,
            improvement_pct=round(improvement, 2),
            method="ppo",
        )

    def _train_qtable(self, total_timesteps: int, seed: int) -> TrainingResult:
        """Fallback: Q-table learning when SB3 is not available."""
        log.info("rl_training_start", method="qtable_fallback", timesteps=total_timesteps)

        rng = np.random.default_rng(seed)

        # Q-table: 10 stock levels x 3 demand trends x 5 actions
        n_stock_levels = 10
        n_demand_trends = 3  # 0=decreasing, 1=stable, 2=increasing
        n_actions = 5
        q_table = np.zeros((n_stock_levels, n_demand_trends, n_actions))

        # Finer stock discretization (10 buckets)
        dm = self.config.demand_mean
        stock_bins = [
            dm * 1, dm * 2, dm * 4, dm * 6, dm * 8,
            dm * 12, dm * 16, dm * 21, dm * 28,
        ]

        order_qtys = [0, dm * 3, dm * 7,
                      dm * 14, dm * 28]

        lr = 0.1
        gamma = 0.99
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.01

        training_curve: List[float] = []
        episode_rewards: List[float] = []

        episodes = total_timesteps // self.config.episode_length
        for ep in range(episodes):
            stock = self.config.demand_mean * self.config.lead_time_days
            total_reward = 0.0
            prev_demand = self.config.demand_mean
            demand_trend = 1  # start stable
            pending_orders: list = []  # list of (arrival_day, qty)

            for _day in range(self.config.episode_length):
                # Receive orders that have arrived
                arrived_qty = sum(q for d, q in pending_orders if d <= _day)
                stock += arrived_qty
                pending_orders = [(d, q) for d, q in pending_orders if d > _day]

                # Discretize stock state (10 buckets)
                state = np.digitize(stock, stock_bins)
                state = min(state, n_stock_levels - 1)

                # Epsilon-greedy (state is now 2D: stock_level, demand_trend)
                if rng.random() < epsilon:
                    action = int(rng.integers(0, n_actions))
                else:
                    action = int(np.argmax(q_table[state, demand_trend]))

                # Simulate
                order_qty = order_qtys[action]
                demand = max(0, rng.normal(self.config.demand_mean, self.config.demand_std))

                # Save old demand trend for Q-update
                old_demand_trend = demand_trend

                # Update demand trend for next step
                if demand > prev_demand * 1.1:
                    demand_trend = 2  # increasing
                elif demand < prev_demand * 0.9:
                    demand_trend = 0  # decreasing
                else:
                    demand_trend = 1  # stable
                prev_demand = demand

                # Orders arrive after lead_time_days, not instantly
                if order_qty > 0:
                    arrival_day = _day + self.config.lead_time_days
                    pending_orders.append((arrival_day, order_qty))

                stockout = max(0, demand - stock)
                stock = max(0, stock - demand)
                stock = min(stock, self.config.max_stock)

                holding_cost = stock * self.config.holding_cost_per_unit / 365
                stockout_cost = stockout * self.config.stockout_cost_per_unit
                ordering_cost = (
                    self.config.ordering_cost_fixed + order_qty * self.config.ordering_cost_per_unit
                ) if order_qty > 0 else 0

                reward = -(holding_cost + stockout_cost + ordering_cost)

                # Q-update with 2D state — use OLD demand trend for LHS
                next_state = min(np.digitize(stock, stock_bins), n_stock_levels - 1)
                td_target = reward + gamma * np.max(q_table[next_state, demand_trend])
                td_error = td_target - q_table[state, old_demand_trend, action]
                q_table[state, old_demand_trend, action] += lr * td_error

                total_reward += reward

            episode_rewards.append(total_reward)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if (ep + 1) % max(1, episodes // 20) == 0:
                training_curve.append(float(np.mean(episode_rewards[-10:])))

        self._q_table = q_table
        self._stock_bins = stock_bins
        self._order_qtys = order_qtys
        self._trained = True

        mean_reward = float(np.mean(episode_rewards[-20:]))
        std_reward = float(np.std(episode_rewards[-20:]))

        # Baseline comparison
        baseline = SsBaseline(
            s=self.config.demand_mean * self.config.lead_time_days * 0.5,
            S=self.config.demand_mean * self.config.lead_time_days * 1.5,
        )
        baseline_result = baseline.evaluate(self.config, n_episodes=20, seed=seed)

        improvement = 0.0
        if baseline_result.mean_reward < 0:
            improvement = (mean_reward - baseline_result.mean_reward) / abs(baseline_result.mean_reward) * 100

        log.info("rl_training_complete", method="qtable", mean_reward=mean_reward)

        return TrainingResult(
            total_episodes=episodes,
            mean_reward=mean_reward,
            std_reward=std_reward,
            training_curve=training_curve,
            baseline_reward=baseline_result.mean_reward,
            improvement_pct=round(improvement, 2),
            method="qtable_fallback",
        )

    def _evaluate_policy(self, n_episodes: int = 10, seed: int = 42) -> float:
        """Evaluate current PPO policy, return mean reward."""
        if not self._model:
            return 0.0
        env = InventoryEnv(self.config)
        try:
            rewards = []
            for i in range(n_episodes):
                obs, _ = env.reset(seed=seed + i)
                total = 0.0
                done = False
                while not done:
                    action, _ = self._model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    total += reward
                    done = terminated or truncated
                rewards.append(total)
            return float(np.mean(rewards))
        finally:
            if hasattr(env, 'close'):
                env.close()

    def _evaluate_policy_stats(self, n_episodes: int = 20, seed: int = 42):
        """Evaluate current PPO policy, return (mean, std)."""
        if not self._model:
            return 0.0, 0.0
        env = InventoryEnv(self.config)
        try:
            rewards = []
            for i in range(n_episodes):
                obs, _ = env.reset(seed=seed + i)
                total = 0.0
                done = False
                while not done:
                    action, _ = self._model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(int(action))
                    total += reward
                    done = terminated or truncated
                rewards.append(total)
            return float(np.mean(rewards)), float(np.std(rewards))
        finally:
            if hasattr(env, 'close'):
                env.close()

    @property
    def is_trained(self) -> bool:
        return self._trained
