"""CLI entry point for ChainCommand.

Usage:
    python -m chaincommand           Start API server
    python -m chaincommand --demo    Run one demo cycle and exit
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chaincommand",
        description="ChainCommand — Supply Chain Risk & Inventory Ops",
    )
    parser.add_argument("--demo", action="store_true", help="Run a single demo cycle (no server)")
    parser.add_argument("--host", default=None, help="Server host (default from config)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default from config)")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(_run_demo())
    else:
        _run_server(host=args.host, port=args.port)


async def _run_demo() -> None:
    """Run one full optimization cycle."""
    from .orchestrator import ChainCommandOrchestrator

    print("=" * 70)
    print("  ChainCommand — Demo Mode")
    print("  Running optimization cycle: ML → BOM → RL → Risk → CP-SAT → CTB")
    print("=" * 70)
    print()

    orchestrator = ChainCommandOrchestrator()
    result = await orchestrator.run_demo()

    print()
    print("=" * 70)
    print("  CYCLE RESULTS")
    print("=" * 70)
    print(f"  Cycle:           {result['cycle']}")
    print(f"  KPI Violations:  {result['violations']}")
    print()

    kpi = result.get("kpi", {})
    print("  KPI Snapshot:")
    print(f"    OTIF:              {kpi.get('otif', 0):.1%}")
    print(f"    Fill Rate:         {kpi.get('fill_rate', 0):.1%}")
    _mape = kpi.get('mape')
    print(f"    MAPE:              {_mape:.1f}%" if _mape is not None else "    MAPE:              N/A")
    print(f"    DSI:               {kpi.get('dsi', 0):.1f} days")
    print(f"    Stockout Count:    {kpi.get('stockout_count', 0)}")
    print(f"    Inventory Value:   ${kpi.get('total_inventory_value', 0):,.0f}")

    if "risk" in result:
        print()
        print("  Risk Assessment:")
        print(f"    Suppliers scored:  {result['risk']['scored']}")
        print(f"    High-risk:         {result['risk']['high_risk_count']}")

    if "rl_decisions" in result:
        print(f"  RL Decisions:        {result['rl_decisions']}")

    if "ctb" in result:
        print()
        print("  CTB Reports:")
        for ctb in result["ctb"]:
            status = "CLEAR" if ctb["is_clear"] else f"SHORT ({ctb['shortages']} parts)"
            print(f"    {ctb['assembly_id']:15s} {ctb['clear_pct']:5.1f}% — {status}")

    print()
    print("=" * 70)
    print("  Demo complete. Run without --demo to start the API server.")
    print("=" * 70)


def _run_server(host: str | None = None, port: int | None = None) -> None:
    """Start the FastAPI server with uvicorn."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn required for server mode: pip install uvicorn")
        sys.exit(1)

    from .config import settings

    uvicorn.run(
        "chaincommand.api.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
