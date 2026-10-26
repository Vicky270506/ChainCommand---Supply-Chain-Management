"""Command line for the MLOps layer.

    python -m chaincommand.mlops bootstrap          # seed + train first models (fresh install)
    python -m chaincommand.mlops train-all          # forecaster + stockout (champion/challenger)
    python -m chaincommand.mlops train-forecaster
    python -m chaincommand.mlops train-stockout
    python -m chaincommand.mlops forecast           # forecast with the champion and save to the database
    python -m chaincommand.mlops plan               # forecasts + pending recommendations
    python -m chaincommand.mlops monitor [--retrain-if-needed]
    python -m chaincommand.mlops scheduler [--once]
    python -m chaincommand.mlops status
"""
from __future__ import annotations

import argparse
import json
import logging

from ..config import settings


def _short(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "metrics"}


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m chaincommand.mlops", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bootstrap")
    sub.add_parser("train-all")
    f = sub.add_parser("train-forecaster")
    f.add_argument("--horizon", type=int)
    sub.add_parser("train-stockout")
    sub.add_parser("forecast")
    sub.add_parser("plan")
    m = sub.add_parser("monitor")
    m.add_argument("--retrain-if-needed", action="store_true")
    s = sub.add_parser("scheduler")
    s.add_argument("--once", action="store_true", help="run the due jobs once and exit")
    s.add_argument("--poll-minutes", type=float, default=15.0)
    sub.add_parser("status")
    args = p.parse_args()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")

    # imported late so `--help` is instant
    from . import retrain, training
    from .tracking import wait_for_server

    if args.cmd == "bootstrap":
        from .bootstrap import bootstrap
        bootstrap()
        return
    wait_for_server()
    if args.cmd == "train-all":
        print(json.dumps({"forecaster": _short(training.train_forecaster()),
                          "stockout": _short(training.train_stockout())}, indent=2, default=str))
    elif args.cmd == "train-forecaster":
        print(json.dumps(_short(training.train_forecaster(horizon=args.horizon)), indent=2, default=str))
    elif args.cmd == "train-stockout":
        print(json.dumps(_short(training.train_stockout()), indent=2, default=str))
    elif args.cmd == "forecast":
        version, out = retrain.forecast_with_champion()
        print(f"saved {len(out)} forecast rows as {version}")
    elif args.cmd == "plan":
        print(json.dumps(retrain.refresh_plan(), indent=2))
    elif args.cmd == "monitor":
        from .monitoring import run_monitoring
        report = run_monitoring()
        print(json.dumps({k: v for k, v in report.items() if k != "drift"}, indent=2, default=str))
        if args.retrain_if_needed and report["retrain_recommended"]:
            print(json.dumps(_short(retrain.retrain(trigger="monitor")["forecaster"]), indent=2, default=str))
    elif args.cmd == "scheduler":
        if args.once:
            print(json.dumps(retrain.tick(), indent=2, default=str))
        else:
            retrain.run_forever(args.poll_minutes)
    elif args.cmd == "status":
        from .status import registry_status
        print(json.dumps(registry_status(), indent=2, default=str))


if __name__ == "__main__":
    main()
