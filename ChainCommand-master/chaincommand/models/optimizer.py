"""Inventory optimization — Genetic Algorithm, DQN, and Hybrid."""

from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np

from ..config import settings
from ..data.schemas import ForecastResult, OptimizationResult, Product
from ..utils.logging_config import get_logger

log = get_logger(__name__)


class GeneticOptimizer:
    """GA optimization for reorder point + safety stock + order quantity."""

    # Fitness function constants
    HOLDING_COST_PCT = 0.25    # Annual holding cost as fraction of unit cost
    ORDERING_COST_MULT = 10    # Ordering cost = unit_cost * this multiplier
    STOCKOUT_PENALTY = 0.1     # Fraction of monthly lost sales treated as penalty
    STOCKOUT_Z = 1.65          # Z-score for service level in stockout probability
    ORDERING_FREQ_COST = 5     # Per-order cost multiplier
    MUTATION_RATE = 0.1        # Probability of mutating each gene

    def __init__(self) -> None:
        self._pop_size = settings.ga_population_size
        self._generations = settings.ga_generations

    def optimize(
        self,
        product: Product,
        demand_forecast: List[ForecastResult],
        seed: int | None = None,
    ) -> OptimizationResult:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        avg_demand = product.daily_demand_avg
        std_demand = product.daily_demand_std
        lead_time = product.lead_time_days
        unit_cost = product.unit_cost
        holding_cost_pct = self.HOLDING_COST_PCT

        if demand_forecast:
            forecast_demands = [f.predicted_demand for f in demand_forecast]
            avg_demand = float(np.mean(forecast_demands))
            std_demand = float(np.std(forecast_demands, ddof=1))

        # Initialize population: [reorder_point, safety_stock, order_qty]
        population = []
        for _ in range(self._pop_size):
            ss = rng.uniform(0.5, 3.0) * std_demand * math.sqrt(lead_time)
            rop = avg_demand * lead_time + ss
            # EOQ approximation with noise
            ordering_cost = unit_cost * self.ORDERING_COST_MULT
            annual_demand = avg_demand * 365
            eoq = math.sqrt(2 * annual_demand * ordering_cost / (unit_cost * holding_cost_pct))
            oq = eoq * rng.uniform(0.7, 1.3)
            population.append([rop, ss, max(oq, product.min_order_qty)])

        # Evolution — track best individual across all generations
        best = population[0]
        best_fitness_val = -1.0

        for _gen in range(self._generations):
            # Fitness = minimize total cost (holding + stockout penalty + ordering)
            fitness = []
            for individual in population:
                rop, ss, oq = individual
                holding = (ss + oq / 2) * unit_cost * holding_cost_pct / 365  # daily
                stockout_prob = max(0, 1 - ss / (std_demand * math.sqrt(lead_time) * self.STOCKOUT_Z + 0.01))
                stockout_cost = stockout_prob * avg_demand * product.selling_price * 30 * self.STOCKOUT_PENALTY
                ordering_freq = max(avg_demand * 30 / max(oq, 1), 0.1)
                ordering_total = ordering_freq * unit_cost * self.ORDERING_FREQ_COST

                cost = holding + stockout_cost + ordering_total
                fitness.append(1 / (cost + 1))

            # Track best individual before population is replaced
            gen_best_idx = int(np.argmax(fitness))
            if fitness[gen_best_idx] > best_fitness_val:
                best_fitness_val = fitness[gen_best_idx]
                best = list(population[gen_best_idx])

            # Selection + crossover
            total_fitness = sum(fitness)
            probs = [f / total_fitness for f in fitness]
            new_pop = []
            for _ in range(self._pop_size):
                # Roulette-wheel (fitness-proportionate) selection
                idx = int(np_rng.choice(len(population), p=probs))
                parent1 = population[idx]
                idx2 = int(np_rng.choice(len(population), p=probs))
                parent2 = population[idx2]

                # Single-point crossover
                crossover_point = rng.randint(1, 2)
                child = list(parent1[:crossover_point] + parent2[crossover_point:])
                # Mutation
                for j in range(3):
                    if rng.random() < self.MUTATION_RATE:
                        child[j] += rng.gauss(0, std_demand * 0.2)
                child = [max(1, c) for c in child]
                child[2] = max(child[2], product.min_order_qty)
                new_pop.append(child)

            # Elitism: preserve best individual from previous generation
            new_pop[0] = list(best)
            population = new_pop

        # Estimate savings vs current
        current_holding = (product.safety_stock + product.current_stock / 2) * unit_cost * holding_cost_pct / 365
        new_holding = (best[1] + best[2] / 2) * unit_cost * holding_cost_pct / 365
        saving = max(0, current_holding - new_holding)

        result = OptimizationResult(
            product_id=product.product_id,
            recommended_reorder_point=round(best[0], 1),
            recommended_safety_stock=round(best[1], 1),
            recommended_order_qty=round(best[2], 0),
            expected_cost_saving=round(saving, 2),
            method="genetic_algorithm",
        )
        log.debug("ga_optimized", product_id=product.product_id, rop=best[0])
        return result


class DQNOptimizer:
    """DQN reinforcement learning for dynamic inventory decisions.

    Simplified state-action model that works without deep learning frameworks.
    """

    def __init__(self) -> None:
        self._epsilon = settings.dqn_epsilon_start
        self._epsilon_end = settings.dqn_epsilon_end
        self._episodes = settings.dqn_episodes
        # Compute decay rate so that epsilon reaches epsilon_end by the last episode:
        #   epsilon_start * decay^episodes = epsilon_end
        #   decay = (epsilon_end / epsilon_start) ^ (1 / episodes)
        if self._episodes > 0 and self._epsilon > 0:
            self._epsilon_decay = (self._epsilon_end / self._epsilon) ** (1.0 / self._episodes)
        else:
            self._epsilon_decay = settings.dqn_epsilon_decay
        self._q_tables: Dict[str, Dict[str, Dict[str, float]]] = {}  # product_id -> state_key -> action -> value
        self._trained = False

    def train(self, product: Product, seed: int | None = None) -> None:
        """Train DQN on simulated inventory environment."""
        rng = random.Random(seed)

        pid = product.product_id
        avg_demand = product.daily_demand_avg
        std_demand = product.daily_demand_std

        # Per-product Q-table
        if pid not in self._q_tables:
            self._q_tables[pid] = {}
        q_table = self._q_tables[pid]

        # Actions: order_nothing, order_small, order_medium, order_large
        actions = ["none", "small", "medium", "large"]
        action_qtys = {
            "none": 0,
            "small": product.min_order_qty,
            "medium": product.min_order_qty * 2,
            "large": product.min_order_qty * 4,
        }

        epsilon = self._epsilon
        for _ep in range(self._episodes):
            stock = product.current_stock
            total_reward = 0

            for _day in range(30):
                # State discretization
                stock_level = "critical" if stock < product.safety_stock else \
                              "low" if stock < product.reorder_point else \
                              "normal" if stock < product.reorder_point * 2 else "high"
                state_key = stock_level

                # Epsilon-greedy action
                if rng.random() < epsilon:
                    action = rng.choice(actions)
                else:
                    q_vals = q_table.get(state_key, {a: 0.0 for a in actions})
                    action = max(q_vals, key=q_vals.get)

                # Simulate
                demand = max(0, rng.gauss(avg_demand, std_demand))
                order_qty = action_qtys[action]
                stock = stock + order_qty - demand

                # Reward: penalize stockouts and excess holding
                reward = 0
                if stock < 0:
                    reward -= abs(stock) * product.selling_price * 0.5  # stockout penalty
                    stock = 0
                else:
                    reward -= stock * product.unit_cost * 0.001  # holding cost

                if order_qty > 0:
                    reward -= order_qty * product.unit_cost * 0.01  # ordering cost

                # Update Q-table (Bellman equation)
                if state_key not in q_table:
                    q_table[state_key] = {a: 0.0 for a in actions}
                lr = 0.1
                gamma = 0.95
                old_q = q_table[state_key].get(action, 0)

                # Compute next state for bootstrapping
                next_stock_level = (
                    "critical" if stock < product.safety_stock else
                    "low" if stock < product.reorder_point else
                    "normal" if stock < product.reorder_point * 2 else "high"
                )
                next_q_vals = q_table.get(
                    next_stock_level, {a: 0.0 for a in actions},
                )
                max_next_q = max(next_q_vals.values()) if next_q_vals else 0.0

                q_table[state_key][action] = (
                    old_q + lr * (reward + gamma * max_next_q - old_q)
                )

                total_reward += reward

            epsilon = max(self._epsilon_end, epsilon * self._epsilon_decay)

        self._trained = True
        log.info("dqn_trained", product_id=pid, episodes=self._episodes)

    def decide(self, product: Product) -> OptimizationResult:
        pid = product.product_id
        stock = product.current_stock
        stock_level = "critical" if stock < product.safety_stock else \
                      "low" if stock < product.reorder_point else \
                      "normal" if stock < product.reorder_point * 2 else "high"

        q_table = self._q_tables.get(pid, {})
        q_vals = q_table.get(stock_level, {})
        if not q_vals:
            # Fallback
            return OptimizationResult(
                product_id=product.product_id,
                recommended_reorder_point=product.reorder_point,
                recommended_safety_stock=product.safety_stock,
                recommended_order_qty=float(product.min_order_qty),
            )

        best_action = max(q_vals, key=q_vals.get)
        qty_map = {
            "none": 0,
            "small": product.min_order_qty,
            "medium": product.min_order_qty * 2,
            "large": product.min_order_qty * 4,
        }

        return OptimizationResult(
            product_id=product.product_id,
            recommended_reorder_point=product.reorder_point,
            recommended_safety_stock=product.safety_stock,
            recommended_order_qty=float(qty_map.get(best_action, product.min_order_qty)),
            method="dqn",
        )


class HybridOptimizer:
    """GA provides initial parameters, DQN fine-tunes dynamically."""

    def __init__(self) -> None:
        self._ga = GeneticOptimizer()
        self._dqn = DQNOptimizer()
        self._dqn_trained: set = set()

    def optimize(
        self,
        product: Product,
        demand_forecast: List[ForecastResult],
        seed: int | None = None,
    ) -> OptimizationResult:
        # Step 1: GA for global optimization
        ga_result = self._ga.optimize(product, demand_forecast, seed=seed)

        # Step 2: DQN for dynamic adjustment — use GA-optimized params
        ga_product = product.model_copy(update={
            "reorder_point": ga_result.recommended_reorder_point,
            "safety_stock": ga_result.recommended_safety_stock,
        })
        if product.product_id not in self._dqn_trained:
            self._dqn.train(ga_product, seed=seed)
            self._dqn_trained.add(product.product_id)

        dqn_result = self._dqn.decide(ga_product)

        # Blend: GA for structural params, DQN for order qty
        blended = OptimizationResult(
            product_id=product.product_id,
            recommended_reorder_point=ga_result.recommended_reorder_point,
            recommended_safety_stock=ga_result.recommended_safety_stock,
            recommended_order_qty=round(
                0.6 * ga_result.recommended_order_qty + 0.4 * dqn_result.recommended_order_qty, 0
            ),
            expected_cost_saving=ga_result.expected_cost_saving,
            method="hybrid_ga_dqn",
        )
        log.debug("hybrid_optimized", product_id=product.product_id)
        return blended
