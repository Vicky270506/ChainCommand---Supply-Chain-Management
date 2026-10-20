# ChainCommand — Claude Code Guide

## Project Overview
ChainCommand is a supply chain risk and inventory operations platform combining CP-SAT constraint optimization, RL inventory policy, BOM management, supplier risk scoring, and CTB (Clear-to-Build) analysis.

## Quick Start
```bash
pip install -e ".[dev]"           # Install with dev deps
python -m chaincommand --demo     # Run demo cycle
pytest tests/ -v                  # Run tests
ruff check chaincommand/ tests/   # Lint
```

## Architecture (v3.0)
- **CP-SAT Optimization**: OR-Tools MILP solver for multi-supplier allocation
- **RL Inventory Policy**: PPO (Stable-Baselines3) with (s,S) baseline comparison
- **BOM Management**: Multi-tier BOM tree, explosion, where-used, cost rollup
- **Supplier Risk Scoring**: 5-factor rule-based + RandomForest ML blending
- **CTB Dashboard**: Clear-to-Build analysis with shortage detection
- **ML Models**: LSTM + XGBoost ensemble forecaster, Isolation Forest anomaly detector
- **Event Bus**: Async pub/sub for KPI violations and alerts
- **AWS Backend**: Optional S3/Redshift/Athena/QuickSight persistence

## Key Directories
- `chaincommand/optimization/` — CP-SAT MILP optimizer + benchmark
- `chaincommand/rl/` — RL inventory environment, trainer, policy
- `chaincommand/bom/` — BOM tree models + manager
- `chaincommand/risk/` — Supplier risk scoring (rule-based + ML)
- `chaincommand/ctb/` — Clear-to-Build analyzer
- `chaincommand/models/` — ML models (forecaster, anomaly_detector, optimizer)
- `chaincommand/kpi/` — 12-metric KPI engine
- `chaincommand/events/` — Async pub/sub event bus
- `chaincommand/api/` — FastAPI REST + WebSocket
- `chaincommand/aws/` — AWS persistence (S3/Redshift/Athena/QuickSight)
- `tests/` — 100+ tests across test modules

## Configuration
All settings via `CC_` env prefix or `.env` file. Key settings:
- `CC_ORTOOLS_RISK_LAMBDA`: Risk-cost trade-off for CP-SAT (default: 0.3)
- `CC_RL_TOTAL_TIMESTEPS`: RL training steps (default: 50000)
- `CC_BOM_DEFAULT_ASSEMBLIES`: Synthetic BOM count (default: 5)
- `CC_CTB_DEFAULT_BUILD_QTY`: Default build quantity for CTB (default: 100)

## Testing
```bash
pytest tests/ -v --tb=short       # All tests
pytest tests/test_optimization/   # CP-SAT tests only
pytest tests/test_rl/             # RL inventory tests
pytest tests/test_bom/            # BOM tests
pytest tests/test_ctb/            # CTB tests
pytest tests/test_risk/           # Risk scoring tests
```

## Development Guidelines
- Always JSON-serialize datetime objects and numpy types before sending through WebSocket.
- After editing Python files, verify syntax with `python -m py_compile <file>`.
- Run `pytest tests/ -v` after code changes, not as a final step.

## Dependencies
Core deps always installed. Optional groups:
- `pip install -e ".[ortools]"` — OR-Tools CP-SAT
- `pip install -e ".[rl]"` — Gymnasium + Stable-Baselines3
- `pip install -e ".[all]"` — Everything
