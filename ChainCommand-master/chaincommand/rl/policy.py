"""RL Inventory Policy — inference wrapper for trained models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.logging_config import get_logger
from .environment import InventoryEnvConfig
from .trainer import RLInventoryTrainer

log = get_logger(__name__)


@dataclass
class PolicyDecision:
    """A single inventory policy decision."""
    action: int  # 0-4
    order_quantity: float
    stock_level: float
    confidence: float = 1.0
    method: str = "ppo"


class RLInventoryPolicy:
    """Inference wrapper for trained RL inventory policy."""

    def __init__(self, config: Optional[InventoryEnvConfig] = None):
        self.config = config or InventoryEnvConfig()
        self._trainer: Optional[RLInventoryTrainer] = None

    def train(self, total_timesteps: int = 50_000, seed: int = 42):
        """Train the underlying RL model."""
        self._trainer = RLInventoryTrainer(self.config)
        return self._trainer.train(total_timesteps=total_timesteps, seed=seed)

    def decide(
        self,
        current_stock: float,
        avg_demand: float,
        day_of_week: int = 0,
        pending_orders: float = 0.0,
        days_since_order: int = 0,
        demand_trend: int = 1,
    ) -> PolicyDecision:
        """Get inventory replenishment decision from trained policy.

        Args:
            demand_trend: 0=decreasing, 1=stable, 2=increasing (default 1).
        """
        order_qtys = [
            0,
            int(self.config.demand_mean * 3),
            int(self.config.demand_mean * 7),
            int(self.config.demand_mean * 14),
            int(self.config.demand_mean * 28),
        ]

        # Zero-guards: avoid division by zero when max_stock or max_demand is 0
        max_stock = self.config.max_stock if self.config.max_stock > 0 else 1.0
        max_demand = self.config.max_demand if self.config.max_demand > 0 else 1.0

        if self._trainer and self._trainer.is_trained and hasattr(self._trainer, '_model') and self._trainer._model:
            # PPO inference
            obs = np.array([
                min(1.0, current_stock / max_stock),
                min(1.0, avg_demand / max_demand),
                (day_of_week % 7) / 7.0,
                min(1.0, pending_orders / max_stock),
                min(1.0, days_since_order / 30.0),
            ], dtype=np.float32)

            action, _ = self._trainer._model.predict(obs, deterministic=True)
            action = int(action)

            return PolicyDecision(
                action=action,
                order_quantity=float(order_qtys[action]),
                stock_level=current_stock,
                # Heuristic confidence: PPO is the most expressive model but
                # we have no calibrated probability — 0.9 reflects high trust
                confidence=0.9,
                method="ppo",
            )

        elif self._trainer and self._trainer.is_trained and hasattr(self._trainer, '_q_table'):
            # Q-table inference — index with both stock_level and demand_trend
            # stock_bins has 9 boundaries for 10 buckets; np.digitize returns 0..9
            n_stock_levels = self._trainer._q_table.shape[0]
            state = int(np.digitize(current_stock, self._trainer._stock_bins))
            state = min(state, n_stock_levels - 1)

            # Clamp demand_trend to valid range (0=decreasing, 1=stable, 2=increasing)
            n_demand_trends = self._trainer._q_table.shape[1]
            dt = max(0, min(demand_trend, n_demand_trends - 1))

            # Index Q-table with both dimensions to get per-action values
            action = int(np.argmax(self._trainer._q_table[state, dt]))

            return PolicyDecision(
                action=action,
                order_quantity=float(self._trainer._order_qtys[action]),
                stock_level=current_stock,
                # Heuristic confidence: Q-table is coarsely discretized, so
                # lower trust than PPO — 0.7 is a conservative estimate
                confidence=0.7,
                method="qtable",
            )

        else:
            # Fallback: simple (s,S) heuristic
            s = self.config.demand_mean * self.config.lead_time_days * 0.5
            S = self.config.demand_mean * self.config.lead_time_days * 1.5

            if current_stock <= s:
                order_qty = S - current_stock
                # Map the continuous order_qty to the closest discrete action
                diffs = [abs(order_qty - q) for q in order_qtys]
                action = int(np.argmin(diffs))
                order_qty = float(order_qtys[action])
            elif current_stock <= s * 1.5:
                order_qty = float(order_qtys[2])
                action = 2  # medium
            else:
                order_qty = 0.0
                action = 0

            return PolicyDecision(
                action=action,
                order_quantity=order_qty,
                stock_level=current_stock,
                # Heuristic confidence: simple rule-based fallback with no
                # learned parameters — 0.5 reflects low certainty
                confidence=0.5,
                method="heuristic_fallback",
            )
