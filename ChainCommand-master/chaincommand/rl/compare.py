"""Rule-based vs tuned (s,S) vs PPO on identical held-out demand."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chaincommand.rl.environment import InventoryEnv, InventoryEnvConfig

Z = 1.645  # 95% service level


def simulate(decide, cfg, demands):
    """One episode, same dynamics as InventoryEnv, on a fixed demand sequence."""
    L = cfg.lead_time_days
    stock, pending, recent, since = cfg.demand_mean * L, [], [], 30
    r = dict(holding=0.0, stockout=0.0, ordering=0.0, short=0.0, demand=0.0,
             stockout_days=0, orders=0, stock_sum=0.0)
    for day, d in enumerate(demands):
        pre_stock, pre_pending = stock, sum(q for _, q in pending)  # what the env's obs shows
        stock += sum(q for a, q in pending if a <= day)
        pending = [(a, q) for a, q in pending if a > day]
        state = dict(stock=stock, pending=sum(q for _, q in pending), recent=recent,
                     day=day, since=since, pre_stock=pre_stock, pre_pending=pre_pending)
        qty = decide(state)
        if qty > 0:
            pending.append((day + L, float(qty)))
            r["ordering"] += cfg.ordering_cost_fixed + qty * cfg.ordering_cost_per_unit
            r["orders"] += 1
            since = 0
        else:
            since += 1
        recent = (recent + [d])[-30:]
        short = max(0.0, d - stock)
        stock = min(max(0.0, stock - d), cfg.max_stock)
        r["holding"] += stock * cfg.holding_cost_per_unit / 365
        r["stockout"] += short * cfg.stockout_cost_per_unit
        r["short"] += short
        r["demand"] += d
        r["stockout_days"] += short > 0
        r["stock_sum"] += stock
    r["total"] = r["holding"] + r["stockout"] + r["ordering"]
    return r


def run(decide, cfg, all_demands):
    return pd.DataFrame([simulate(decide, cfg, d) for d in all_demands])


def rule_based(cfg):
    """Report formula: forecast over lead time + safety stock - stock - incoming.
    A 7-day cycle stock is added to the order so it does not reorder daily."""
    L = cfg.lead_time_days

    def f(st):
        rec = st["recent"]
        mu = np.mean(rec) if len(rec) >= 7 else cfg.demand_mean
        sd = np.std(rec) if len(rec) >= 7 else cfg.demand_std
        need = mu * L + Z * sd * np.sqrt(L)
        pos = st["stock"] + st["pending"]
        return 0 if pos >= need else need + 7 * mu - pos
    return f


def ss_policy(s, S):
    def f(st):
        pos = st["stock"] + st["pending"]
        return max(0.0, S - pos) if pos <= s else 0
    return f


def tune_ss(cfg, demands):
    best = None
    for s in range(300, 1601, 100):
        for gap in range(300, 2101, 300):
            cost = run(ss_policy(s, s + gap), cfg, demands)["total"].mean()
            if best is None or cost < best[0]:
                best = (cost, s, s + gap)
    return best[1], best[2]


def rl_policy(model, cfg, qtys):
    def f(st):
        rec = st["recent"]
        avg = np.mean(rec) if rec else cfg.demand_mean
        obs = np.array([min(1.0, st["pre_stock"] / cfg.max_stock),
                        min(1.0, avg / cfg.max_demand),
                        (st["day"] % 7) / 7.0,
                        min(1.0, st["pre_pending"] / cfg.max_stock),
                        min(1.0, st["since"] / 30.0)], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        return qtys[int(action)]
    return f


def summarise(name, df, cfg):
    return {
        "policy": name,
        "total_cost": df["total"].mean(),
        "cost_std": df["total"].std(),
        "holding": df["holding"].mean(),
        "stockout_cost": df["stockout"].mean(),
        "ordering": df["ordering"].mean(),
        "service_level_%": 100 * (1 - df["short"].sum() / df["demand"].sum()),
        "stockout_day_%": 100 * df["stockout_days"].mean() / cfg.episode_length,
        "orders_per_ep": df["orders"].mean(),
        "avg_stock": df["stock_sum"].mean() / cfg.episode_length,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--skip-rl", action="store_true")
    args = ap.parse_args()

    cfg = InventoryEnvConfig()
    T = cfg.episode_length

    def demands(seed, n):
        return np.maximum(0, np.random.default_rng(seed).normal(cfg.demand_mean, cfg.demand_std, (n, T)))

    tune_d, test_d = demands(1, 100), demands(999, args.episodes)  # RL trained with seed 42
    s, S = tune_ss(cfg, tune_d)
    print(f"(s,S) tuned on separate demand: s={s}, S={S}")

    rows = [summarise("rule_based (forecast+safety-stock-incoming)", run(rule_based(cfg), cfg, test_d), cfg),
            summarise(f"(s,S) tuned s={s} S={S}", run(ss_policy(s, S), cfg, test_d), cfg)]

    if not args.skip_rl:
        from chaincommand.rl.policy import RLInventoryPolicy
        pol = RLInventoryPolicy(cfg)
        pol.train(total_timesteps=args.steps, seed=42)
        model = getattr(pol._trainer, "_model", None)
        if model is None:
            print("WARNING: trainer produced no PPO model; RL row skipped")
        else:
            qtys = list(InventoryEnv(cfg)._order_quantities)
            rows.append(summarise(f"PPO ({args.steps} steps)", run(rl_policy(model, cfg, qtys), cfg, test_d), cfg))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    Path("artifacts").mkdir(exist_ok=True)
    out.to_csv("artifacts/policy_comparison.csv", index=False)


if __name__ == "__main__":
    main()