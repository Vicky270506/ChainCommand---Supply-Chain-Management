"""Gymnasium environment for inventory replenishment."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.logging_config import get_logger

log = get_logger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


class InventoryEnvConfig:
    """Configuration for the inventory environment."""
    def __init__(
        self,
        max_stock: float = 5000.0,
        max_demand: float = 500.0,
        holding_cost_per_unit: float = 0.5,
        stockout_cost_per_unit: float = 10.0,
        ordering_cost_fixed: float = 50.0,
        ordering_cost_per_unit: float = 1.0,
        lead_time_days: int = 7,
        episode_length: int = 90,
        demand_mean: float = 100.0,
        demand_std: float = 30.0,
    ):
        self.max_stock = max_stock
        self.max_demand = max_demand
        self.holding_cost_per_unit = holding_cost_per_unit
        self.stockout_cost_per_unit = stockout_cost_per_unit
        self.ordering_cost_fixed = ordering_cost_fixed
        self.ordering_cost_per_unit = ordering_cost_per_unit
        self.lead_time_days = lead_time_days
        self.episode_length = episode_length
        self.demand_mean = demand_mean
        self.demand_std = demand_std


if HAS_GYM:
    class InventoryEnv(gym.Env):
        """Discrete inventory replenishment environment.

        State: [current_stock, avg_demand, day_of_week, pending_orders, days_since_last_order]
        Action: Discrete order quantity buckets (0=none, 1=small, 2=medium, 3=large, 4=xlarge)
        Reward: -(holding_cost + stockout_cost + ordering_cost)
        """
        metadata = {"render_modes": []}

        def __init__(self, config: Optional[InventoryEnvConfig] = None, **kwargs):
            super().__init__()
            self.config = config or InventoryEnvConfig()

            # Action space: 5 discrete order levels
            self.action_space = spaces.Discrete(5)
            self._order_quantities = [
                0,
                int(self.config.demand_mean * 3),   # ~3 days
                int(self.config.demand_mean * 7),   # ~1 week
                int(self.config.demand_mean * 14),  # ~2 weeks
                int(self.config.demand_mean * 28),  # ~4 weeks
            ]

            # Observation space: [stock_ratio, demand_ratio, day_of_week/7, pending_ratio, days_since_order/30]
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            )

            self._stock = 0.0
            self._day = 0
            self._pending_orders: list[Tuple[int, float]] = []  # (arrival_day, qty)
            self._days_since_order = 0
            self._recent_demands: list[float] = []
            self._episode_rewards: list[float] = []
            self._rng = np.random.default_rng()

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._rng = np.random.default_rng(seed)
            self._stock = self.config.demand_mean * self.config.lead_time_days  # start with lead-time worth
            self._day = 0
            self._pending_orders = []
            self._days_since_order = 30
            self._recent_demands = []
            self._episode_rewards = []
            return self._get_obs(), {}

        def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            # Process incoming orders
            arrived = [(d, q) for d, q in self._pending_orders if d <= self._day]
            for _, qty in arrived:
                self._stock += qty
            self._pending_orders = [(d, q) for d, q in self._pending_orders if d > self._day]

            # Place order
            order_qty = self._order_quantities[action]
            ordering_cost = 0.0
            if order_qty > 0:
                arrival = self._day + self.config.lead_time_days
                self._pending_orders.append((arrival, float(order_qty)))
                ordering_cost = self.config.ordering_cost_fixed + order_qty * self.config.ordering_cost_per_unit
                self._days_since_order = 0
            else:
                self._days_since_order += 1

            # Simulate demand
            demand = max(0, self._rng.normal(self.config.demand_mean, self.config.demand_std))
            self._recent_demands.append(demand)
            if len(self._recent_demands) > 30:
                self._recent_demands.pop(0)

            # Consume stock
            stockout = max(0, demand - self._stock)
            self._stock = max(0, self._stock - demand)

            # Cap stock
            self._stock = min(self._stock, self.config.max_stock)

            # Calculate reward
            holding_cost = self._stock * self.config.holding_cost_per_unit / 365
            stockout_cost = stockout * self.config.stockout_cost_per_unit
            reward = -(holding_cost + stockout_cost + ordering_cost)

            self._day += 1
            self._episode_rewards.append(reward)

            terminated = self._day >= self.config.episode_length
            truncated = False

            info = {
                "stock": self._stock,
                "demand": demand,
                "stockout": stockout,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
                "ordering_cost": ordering_cost,
                "order_qty": order_qty,
            }

            return self._get_obs(), reward, terminated, truncated, info

        def _get_obs(self) -> np.ndarray:
            max_stock = self.config.max_stock if self.config.max_stock > 0 else 1.0
            max_demand = self.config.max_demand if self.config.max_demand > 0 else 1.0
            stock_ratio = min(1.0, self._stock / max_stock)
            avg_demand = np.mean(self._recent_demands) if self._recent_demands else self.config.demand_mean
            demand_ratio = min(1.0, avg_demand / max_demand)
            dow = (self._day % 7) / 7.0
            pending_qty = sum(q for _, q in self._pending_orders)
            pending_ratio = min(1.0, pending_qty / max_stock)
            days_ratio = min(1.0, self._days_since_order / 30.0)
            return np.array([stock_ratio, demand_ratio, dow, pending_ratio, days_ratio], dtype=np.float32)

else:
    # Fallback stub when gymnasium is not installed
    class InventoryEnv:  # type: ignore[no-redef]
        """Stub InventoryEnv when gymnasium is not installed.

        Mirrors the real InventoryEnv interface so tests work without gymnasium.
        """
        def __init__(self, config=None, **kwargs):
            self.config = config or InventoryEnvConfig()
            self._order_quantities = [
                0,
                int(self.config.demand_mean * 3),
                int(self.config.demand_mean * 7),
                int(self.config.demand_mean * 14),
                int(self.config.demand_mean * 28),
            ]
            self._stock = 0.0
            self._day = 0
            self._pending_orders: list = []
            self._days_since_order = 0
            self._recent_demands: list = []
            self._episode_rewards: list = []
            self._rng = np.random.default_rng()
            log.warning("gymnasium_not_installed", fallback="stub_env")

        def reset(self, **kwargs):
            seed = kwargs.get("seed")
            self._rng = np.random.default_rng(seed)
            self._stock = self.config.demand_mean * self.config.lead_time_days
            self._day = 0
            self._pending_orders = []
            self._days_since_order = 30
            self._recent_demands = []
            self._episode_rewards = []
            return self._get_obs(), {}

        def step(self, action):
            # Process incoming orders
            arrived = [(d, q) for d, q in self._pending_orders if d <= self._day]
            for _, qty in arrived:
                self._stock += qty
            self._pending_orders = [(d, q) for d, q in self._pending_orders if d > self._day]

            # Place order
            order_qty = self._order_quantities[action]
            ordering_cost = 0.0
            if order_qty > 0:
                arrival = self._day + self.config.lead_time_days
                self._pending_orders.append((arrival, float(order_qty)))
                ordering_cost = self.config.ordering_cost_fixed + order_qty * self.config.ordering_cost_per_unit
                self._days_since_order = 0
            else:
                self._days_since_order += 1

            # Simulate demand
            demand = max(0, self._rng.normal(self.config.demand_mean, self.config.demand_std))
            self._recent_demands.append(demand)
            if len(self._recent_demands) > 30:
                self._recent_demands.pop(0)

            # Consume stock
            stockout = max(0, demand - self._stock)
            self._stock = max(0, self._stock - demand)
            self._stock = min(self._stock, self.config.max_stock)

            # Calculate reward
            holding_cost = self._stock * self.config.holding_cost_per_unit / 365
            stockout_cost = stockout * self.config.stockout_cost_per_unit
            reward = -(holding_cost + stockout_cost + ordering_cost)

            self._day += 1
            self._episode_rewards.append(reward)
            terminated = self._day >= self.config.episode_length
            truncated = False

            info = {
                "stock": self._stock,
                "demand": demand,
                "stockout": stockout,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
                "ordering_cost": ordering_cost,
                "order_qty": order_qty,
            }

            return self._get_obs(), reward, terminated, truncated, info

        def _get_obs(self) -> np.ndarray:
            max_stock = max(self.config.max_stock, 1.0)
            max_demand = max(self.config.max_demand, 1.0)
            stock_ratio = min(1.0, self._stock / max_stock)
            avg_demand = np.mean(self._recent_demands) if self._recent_demands else self.config.demand_mean
            demand_ratio = min(1.0, avg_demand / max_demand)
            dow = (self._day % 7) / 7.0
            pending_qty = sum(q for _, q in self._pending_orders)
            pending_ratio = min(1.0, pending_qty / max_stock)
            days_ratio = min(1.0, self._days_since_order / 30.0)
            return np.array([stock_ratio, demand_ratio, dow, pending_ratio, days_ratio], dtype=np.float32)
