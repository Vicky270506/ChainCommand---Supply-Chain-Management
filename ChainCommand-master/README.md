<div align="center">

# ChainCommand — Supply Chain Risk & Inventory Ops Platform

**CP-SAT Optimization | RL Inventory Policy | BOM Management | Supplier Risk Scoring | CTB Dashboard**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-E92063.svg)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<br>

<img src="https://img.shields.io/badge/Optimization-CP--SAT%20MILP-red?style=for-the-badge" />
<img src="https://img.shields.io/badge/RL-PPO%20Inventory%20Policy-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/ML-LSTM%20%2B%20XGBoost%20%2B%20RandomForest-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/BOM-Multi--Tier%20Explosion-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/AWS-S3%20%7C%20Redshift%20%7C%20Athena%20%7C%20QuickSight-FF9900?style=for-the-badge" />
<img src="https://img.shields.io/badge/Tests-181%20Passed-brightgreen?style=for-the-badge" />

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Pipeline Details](#pipeline-details)
- [API Reference](#api-reference)
- [Decision Cycle Walkthrough](#decision-cycle-walkthrough)
- [Research Foundations](#research-foundations)
- [AWS Integration (Optional)](#aws-integration-optional)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [Enterprise Deployment Infrastructure](#enterprise-deployment-infrastructure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

**ChainCommand** is a supply chain risk and inventory operations platform that combines constraint optimization, reinforcement learning, and multi-tier BOM analysis to automate procurement, inventory replenishment, and supplier risk assessment.

The system runs from a single command (`python -m chaincommand --demo`) with zero API keys: it generates a realistic supply chain scenario (50 products, 20 suppliers, 365-day demand history), trains ML models, builds BOM trees, trains an RL inventory policy, scores supplier risk, and executes a complete optimization cycle with CTB (Clear-to-Build) analysis.

### Why This Project?

| Challenge | Approach |
|-----------|----------|
| Supplier allocation is complex and multi-constraint | CP-SAT MILP optimization (cost + risk + capacity + MOQ + lead-time) |
| Inventory replenishment relies on static rules | PPO reinforcement learning with (s,S) heuristic baseline comparison |
| BOM management lacks visibility into shortages | Multi-tier BOM explosion + CTB analysis with shortage detection |
| Supplier switching decisions lack quantified risk | 5-factor rule-based scoring + RandomForest ML blending |
| High-cost decisions need oversight | HITL approval gates with configurable cost thresholds |
| Forecasting relies on single models | LSTM + XGBoost ensemble with dynamic MAPE-based weighting |
| Systems fail silently | Graceful fallback chains (PPO → Q-table → heuristic; CP-SAT → greedy) |
| Production data lacks persistence | AWS Strategy Pattern backend (S3 / Redshift / Athena / QuickSight) |

---

## Key Features

- **CP-SAT MILP Optimization** — OR-Tools constraint programming for optimal multi-supplier allocation with demand, capacity, MOQ, lead-time, and risk constraints
- **RL Inventory Policy** — PPO (Stable-Baselines3) for dynamic inventory replenishment with Gymnasium environment; falls back PPO → Q-table → (s,S) heuristic
- **Multi-Tier BOM Management** — BOM tree with explosion, where-used traversal, cost rollup, and critical path analysis; synthetic generation with 3 assembly templates
- **Supplier Risk Scoring** — 5-factor weighted scoring (delivery, quality, financial stability, geographic, concentration) + RandomForest ML blending (70% rule / 30% ML)
- **CTB (Clear-to-Build) Dashboard** — Component availability analysis, shortage detection with critical path tracing, only checks "buy" items
- **Ensemble Forecasting** — LSTM + XGBoost with dynamic inverse-MAPE weighting
- **Anomaly Detection** — Isolation Forest + Z-score for demand spikes, cost anomalies, and lead-time deviations
- **12 KPI Metrics** — OTIF, fill rate, MAPE, DSI, stockout count, inventory turnover, carrying cost, and more
- **Event-Driven Architecture** — Async pub/sub EventBus with proactive monitoring and tick-based health checks
- **HITL Approval Gates** — Orders ≥$50K require human approval; $10K–$50K pending review; <$10K auto-approved
- **REST API + WebSocket** — FastAPI dashboard with live event streaming and simulation control
- **AWS Persistence (Optional)** — Strategy Pattern backend with S3, Redshift, Athena, and QuickSight integration
- **181 Tests** — Unit, property-based (Hypothesis), and API tests across 15 test modules
- **Docker Deployment** — Dockerfile + docker-compose.yml for containerized deployment

---

## System Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Optimization Cycle                              │
│                                                                      │
│  Anomaly Detection ──→ Risk Scoring ──→ CP-SAT Allocation            │
│  ──→ RL Inventory Decisions ──→ CTB Analysis ──→ KPI Update          │
│                                                                      │
└────────────────────┬──────────────────┬──────────────────────────────┘
                     │                  │
        ┌────────────┼──────────────────┼────────────┐
        │            │                  │            │
  ┌──────────┐ ┌──────────┐  ┌──────────────┐ ┌──────────┐
  │  CP-SAT  │ │   PPO    │  │  BOM Engine  │ │  Risk    │
  │   MILP   │ │ Inventory│  │  + CTB       │ │  Scorer  │
  │Optimizer │ │  Policy  │  │  Analyzer    │ │ (RF+Rule)│
  └──────────┘ └──────────┘  └──────────────┘ └──────────┘
```

### CP-SAT MILP Formulation

```
min  Σ(cᵢ · xᵢ) + λ · Σ(rᵢ · xᵢ)

s.t. Σ xᵢ ≥ D                    (demand satisfaction)
     xᵢ ≥ MOQᵢ · yᵢ              (minimum order qty)
     xᵢ ≤ Capᵢ · yᵢ              (capacity limit)
     Σ yᵢ ≤ K                     (max suppliers)
     LTᵢ · yᵢ ≤ LT_max           (lead-time)
```

### RL Inventory Environment

```
Observation Space (Box[5]):
  [stock_ratio, demand_ratio, day_of_week, pending_ratio, days_since_order]

Action Space (Discrete[5]):
  0=none, 1=small(25%), 2=medium(50%), 3=large(100%), 4=xlarge(150%)

Reward: -holding_cost × stock - stockout_cost × max(0, demand-stock) - ordering_cost
Fallback: PPO → Q-table → (s,S) heuristic
```

### Supplier Risk Scoring

```
5-Factor Rule-Based Scoring:
  delivery_risk     = 1 - on_time_rate           (weight: 0.25)
  quality_risk      = defect_rate × 10           (weight: 0.25)
  financial_risk    = 1 - financial_stability     (weight: 0.20)
  geographic_risk   = base_risk × disruption_freq (weight: 0.15)
  concentration_risk = supply_share               (weight: 0.15)

ML Blending: 70% rule-based + 30% RandomForest (trained on historical data)
Risk Levels: low (< 0.3) | medium (0.3–0.6) | high (0.6–0.8) | critical (≥ 0.8)
```

---

## Project Structure

```
chaincommand/
│
├── __init__.py                          # Package metadata (v3.0.0)
├── __main__.py                          # CLI entry point (--demo / server)
├── config.py                            # Pydantic Settings (CC_ prefix, env-driven)
├── orchestrator.py                      # 6-step optimization cycle orchestrator
├── metrics.py                           # Prometheus metrics
├── auth.py                              # API key authentication
│
├── optimization/                        # Mathematical Optimization
│   ├── cpsat_optimizer.py               # OR-Tools CP-SAT MILP solver
│   └── benchmark.py                     # CP-SAT vs GA benchmark comparison
│
├── rl/                                  # Reinforcement Learning
│   ├── environment.py                   # Gymnasium inventory env (Discrete[5] × Box[5])
│   ├── trainer.py                       # PPO training + Q-table fallback + (s,S) baseline
│   └── policy.py                        # 3-tier inference: PPO → Q-table → (s,S) heuristic
│
├── bom/                                 # Bill of Materials
│   ├── models.py                        # BOMItem, BOMTree (explosion, where-used, cost rollup)
│   └── manager.py                       # BOMManager (synthetic generation, risk analysis)
│
├── risk/                                # Supplier Risk Scoring
│   └── scorer.py                        # 5-factor rule-based + RandomForest ML blending
│
├── ctb/                                 # Clear-to-Build
│   └── analyzer.py                      # CTB analysis with shortage detection & critical path
│
├── data/                                # Domain Data
│   ├── schemas.py                       # Pydantic models + enums
│   └── generator.py                     # Synthetic data: 50 products, 20 suppliers
│
├── models/                              # ML Models
│   ├── forecaster.py                    # LSTM + XGBoost ensemble forecaster
│   ├── anomaly_detector.py              # Isolation Forest + Z-score
│   └── optimizer.py                     # GA + DQN hybrid optimizer
│
├── kpi/                                 # KPI Engine
│   └── engine.py                        # 12 metrics, threshold checks, trends
│
├── events/                              # Event Engine
│   ├── bus.py                           # EventBus (async pub/sub)
│   └── monitor.py                       # ProactiveMonitor (tick-based health checks)
│
├── api/                                 # FastAPI Application
│   ├── app.py                           # FastAPI app with CORS & lifespan
│   └── routes/
│       ├── dashboard.py                 # KPI, inventory, BOM, risk, CTB, events, WebSocket
│       └── control.py                   # Simulation start/stop/speed
│
├── aws/                                 # AWS Persistence Backend (optional)
│   ├── backend.py                       # PersistenceBackend ABC, NullBackend, factory
│   ├── aws_backend.py                   # AWSBackend — assembles all clients
│   ├── s3_client.py                     # S3 upload/download (Parquet, JSONL, JSON)
│   ├── redshift_client.py              # Redshift DDL, COPY, queries
│   ├── athena_client.py                # Athena external tables, ad-hoc queries
│   └── quicksight_client.py            # QuickSight datasets + dashboards
│
└── utils/
    └── logging_config.py               # structlog configuration

tests/                                   # 181 tests across 15 modules
├── conftest.py                          # Shared fixtures
├── test_models/                         # ML model tests (23 tests)
├── test_kpi/                            # KPI engine tests (8 tests)
├── test_optimization/                   # CP-SAT optimizer tests (10 tests)
├── test_bom/                            # BOM tree + manager tests (17 tests)
├── test_rl/                             # RL inventory tests (12 tests)
├── test_ctb/                            # CTB analyzer tests (8 tests)
├── test_risk/                           # Risk scorer tests (11 tests)
├── test_property_based.py               # Hypothesis property tests (4 tests)
├── test_api/                            # API endpoint + security tests (28 tests)
├── test_aws/                            # AWS backend tests (52 tests, all mocked)
├── test_events/                         # Event bus tests (2 tests)
├── test_orchestrator_lifecycle.py       # Orchestrator lifecycle tests (3 tests)
└── test_mlflow_registry.py             # MLflow registry tests (3 tests)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/hsinnearth7/ChainCommand.git
cd ChainCommand

# Install core + dev dependencies (PEP 621)
pip install -e ".[dev]"

# Optional: install specific feature groups
pip install -e ".[ortools]"      # OR-Tools CP-SAT MILP
pip install -e ".[rl]"           # Gymnasium + Stable-Baselines3 (PPO)
pip install -e ".[all]"          # Everything
```

### Quick Start

```bash
# Run a single optimization cycle (no API keys needed)
python -m chaincommand --demo

# Start the FastAPI server
python -m chaincommand --host 0.0.0.0 --port 8000

# Docker
docker compose up --build
```

### Environment Variables

All settings are configurable via `CC_` prefixed environment variables or a `.env` file:

```bash
# Simulation
CC_NUM_PRODUCTS=50
CC_NUM_SUPPLIERS=20
CC_SIMULATION_SPEED=1.0

# CP-SAT Optimization
CC_ORTOOLS_RISK_LAMBDA=0.3         # Risk-cost trade-off weight

# RL Inventory Policy
CC_RL_TOTAL_TIMESTEPS=50000        # PPO training steps
CC_RL_EPISODE_LENGTH=90            # Episode length (days)
CC_RL_HOLDING_COST=0.5             # Daily holding cost per unit
CC_RL_STOCKOUT_COST=10.0           # Stockout penalty per unit
CC_RL_ORDERING_COST_FIXED=50.0     # Fixed cost per order

# BOM / CTB
CC_BOM_DEFAULT_ASSEMBLIES=5        # Synthetic BOM assemblies to generate
CC_CTB_DEFAULT_BUILD_QTY=100       # Default build quantity for CTB analysis

# KPI Thresholds
CC_OTIF_TARGET=0.95
CC_FILL_RATE_TARGET=0.97
CC_MAPE_THRESHOLD=15.0

# HITL Escalation
CC_COST_ESCALATION_THRESHOLD=50000
CC_AUTO_APPROVE_BELOW=10000
```

---

## Pipeline Details

### CP-SAT MILP Optimization

> `chaincommand/optimization/` — OR-Tools constraint programming for supplier allocation

Solves the multi-supplier allocation problem minimizing total cost + risk, subject to demand satisfaction, per-supplier capacity, minimum order quantity, lead-time, and max-supplier-count constraints. Includes a benchmark module comparing CP-SAT vs GA solutions on cost and optimality gap.

### RL Inventory Policy

> `chaincommand/rl/` — PPO-based inventory replenishment with Gymnasium environment

**Environment**: Custom `InventoryEnv` with 5D normalized observation space and 5 discrete actions (order nothing through extra-large order). Reward penalizes holding cost, stockout cost, and fixed ordering cost.

**Training**: PPO via Stable-Baselines3 with automatic (s,S) baseline comparison. Falls back to Q-table training if SB3 is unavailable.

**Inference**: 3-tier fallback chain ensures decisions are always available:
```
PPO model (if trained) → Q-table (if trained) → (s,S) heuristic (always available)
```

### BOM Management

> `chaincommand/bom/` — Multi-tier Bill of Materials

**BOMTree** — Tree structure supporting:
- `explode()` — Recursive multi-level explosion with quantity accumulation and scrap rate
- `where_used()` — Bottom-up search to find all assemblies using a part
- `cost_rollup()` — Aggregate material cost from leaf to root
- `critical_path()` — Longest-lead-time path through the BOM
- `validate()` — Cycle detection and structural integrity checks

**BOMManager** — 3 synthetic assembly templates (PCB Assembly, Sensor Module, Motor Assembly), single-source risk identification, and long-lead-time item detection.

### CTB (Clear-to-Build) Analysis

> `chaincommand/ctb/` — Component availability for production planning

Explodes a BOM, checks current inventory + on-order quantities against requirements for a target build quantity, and reports:
- Whether the assembly is clear to build
- Shortage details per part (required vs. available vs. shortfall)
- Estimated longest wait time for material arrival
- Total material cost

Only "buy" items are checked — sub-assemblies marked as "make" are skipped since they are internally produced.

### Supplier Risk Scoring

> `chaincommand/risk/` — Quantified supplier risk with actionable recommendations

**5-Factor Rule-Based**: Delivery reliability, quality (defect rate), financial stability, geographic risk, and supply concentration — each weighted and combined into an overall score.

**ML Enhancement**: Optional RandomForest model trained on synthetic historical data, blended 70/30 with rule-based scores for improved accuracy.

**Output**: Risk level (low/medium/high/critical), per-factor scores, and specific recommendations (e.g., "Request corrective action plan for defect rate > 3%").

### ML Models

> `chaincommand/models/` — Forecasting, anomaly detection, and optimization

**Ensemble Forecaster** — Dynamic-weighted LSTM + XGBoost:
```
Weight_LSTM = (1/MAPE_LSTM) / ((1/MAPE_LSTM) + (1/MAPE_XGB))
Weight_XGB  = (1/MAPE_XGB)  / ((1/MAPE_LSTM) + (1/MAPE_XGB))
```

**Anomaly Detector** — Isolation Forest with Z-score fallback:
- Demand spike detection (|z| > 2.5)
- Overstock detection (DSI > 60 days)
- Understock detection (DSI < 10 days)

### KPI Engine

> `chaincommand/kpi/engine.py` — 12 real-time supply chain metrics

| KPI | Formula | Threshold |
|-----|---------|-----------|
| **OTIF** | On-Time In-Full deliveries / Total deliveries | ≥ 95% |
| **Fill Rate** | Fulfilled demand / Total demand | ≥ 97% |
| **MAPE** | Mean Absolute Percentage Error of forecasts | ≤ 15% |
| **DSI** | Total Stock / Average Daily Demand | 10–60 days |
| **Stockout Count** | Products with zero stock | ≤ 3 |
| **Inventory Value** | Σ (stock × unit cost) | — |
| **Carrying Cost** | 25% of inventory value / 365 (daily) | — |
| **Order Cycle Time** | Average days from PO creation to delivery | — |
| **Perfect Order Rate** | Perfect deliveries / Total orders | — |
| **Inventory Turnover** | Annual COGS / Average inventory value | — |
| **Backorder Rate** | Backordered products / Total products | — |
| **Supplier Defect Rate** | Average defect rate across active suppliers | — |

### Event Engine

> `chaincommand/events/` — Async pub/sub and proactive monitoring

**EventBus** — Asynchronous publish/subscribe with error isolation:
```
publish(event) → dispatches to type-specific + wildcard subscribers
subscribe(event_type, handler) → register for specific events
subscribe_all(handler) → register for ALL events
```

**ProactiveMonitor** — Tick-based health scanning:
1. Inventory water level checks → `stockout_alert`, `low_stock_alert`, `overstock_alert`
2. KPI threshold violations → `kpi_threshold_violated`
3. Delivery delay detection → `delivery_delayed`
4. Anomaly detection batch → `anomaly_detected`

---

## API Reference

### Dashboard Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/kpi/current` | Latest KPI snapshot (12 metrics) |
| `GET` | `/api/kpi/history?periods=30` | KPI trend data |
| `GET` | `/api/inventory/status` | All products with stock status |
| `GET` | `/api/inventory/status?product_id=PRD-0001` | Single product detail |
| `GET` | `/api/bom/summary` | BOM summary (assembly count, component count, risks) |
| `GET` | `/api/bom/risks` | Single-source and long-lead-time risks |
| `GET` | `/api/risk/scores` | All supplier risk scores and recommendations |
| `GET` | `/api/ctb/status` | CTB analysis for all assemblies |
| `GET` | `/api/events/recent?limit=50` | Recent supply chain events |
| `GET` | `/api/approvals/pending` | Pending HITL approval requests |
| `POST` | `/api/approval/{id}/decide` | Approve or reject a request |
| `GET` | `/api/aws/status` | AWS backend status and configuration |
| `GET` | `/api/aws/kpi-trend/{metric}` | KPI trend from Redshift |
| `WS` | `/ws/live` | Real-time event stream |

### Control Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/simulation/start` | Start continuous optimization loop |
| `POST` | `/api/simulation/stop` | Stop simulation |
| `POST` | `/api/simulation/speed?speed=5.0` | Adjust simulation speed (0.1–100x) |
| `GET` | `/api/simulation/status` | Running state, cycle count, stats |

---

## Decision Cycle Walkthrough

Each cycle follows a 6-step optimization sequence:

```
Step 1: ANOMALY DETECTION
  └── Isolation Forest + Z-score scan across all products
      → demand spikes, cost anomalies, stock-level warnings

Step 2: SUPPLIER RISK SCORING
  └── 5-factor rule-based + RandomForest ML blending
      → risk scores, risk levels, actionable recommendations

Step 3: CP-SAT SUPPLIER ALLOCATION
  └── OR-Tools MILP optimization
      → optimal allocation across suppliers (cost + risk minimized)
      → HITL gate: ≥$50K requires approval

Step 4: RL INVENTORY DECISIONS
  └── PPO policy (or Q-table / (s,S) fallback)
      → per-product order quantities based on current state

Step 5: CTB (CLEAR-TO-BUILD) ANALYSIS
  └── BOM explosion → component availability check
      → shortage reports, longest-wait estimates, material costs

Step 6: KPI UPDATE
  └── 12-metric snapshot with threshold violation detection
      → events published to EventBus for downstream consumption
```

---

## Research Foundations

| Research | Source | Applied Concept |
|----------|--------|----------------|
| JD.com Two-Layer Architecture | ArXiv 2509.03811 | Strategic + Tactical separation in decision hierarchy |
| MARL Inventory Replenishment | ArXiv 2511.23366 | RL-based inventory decisions (adapted to PPO single-agent) |
| Temporal Hierarchical MAS | ArXiv 2508.12683 | Three temporal layers (strategic/tactical/operational) |
| CP-SAT Best Practices | OR-Tools Documentation | MILP formulation for supplier allocation constraints |
| PPO Algorithm | Schulman et al. 2017 | Proximal Policy Optimization for inventory policy |

---

## AWS Integration (Optional)

ChainCommand supports an optional AWS persistence backend using the **Strategy Pattern**. When enabled, cycle data is persisted to S3/Redshift for durable storage and analytics via Athena and QuickSight — defaulting to a zero-overhead `NullBackend` when disabled.

### Architecture

```
PersistenceBackend (ABC)
  ├── NullBackend        # Default — no-op, zero overhead
  └── AWSBackend         # S3 + Redshift + Athena + QuickSight
        ├── S3Client         # Upload Parquet/JSONL/JSON
        ├── RedshiftClient   # COPY from S3, SQL queries
        ├── AthenaClient     # External tables on S3, ad-hoc queries
        └── QuickSightClient # Datasets + dashboards
```

### Environment Variables

```bash
CC_AWS_ENABLED=true
CC_AWS_REGION=ap-northeast-1
CC_AWS_S3_BUCKET=chaincommand-data
CC_AWS_S3_PREFIX=supply-chain/
CC_AWS_REDSHIFT_HOST=my-cluster.abc123.redshift.amazonaws.com
CC_AWS_REDSHIFT_PORT=5439
CC_AWS_REDSHIFT_DB=chaincommand
CC_AWS_REDSHIFT_USER=admin
CC_AWS_REDSHIFT_PASSWORD=secret
CC_AWS_REDSHIFT_IAM_ROLE=arn:aws:iam::123456789012:role/RedshiftS3Access
CC_AWS_ATHENA_DATABASE=chaincommand
CC_AWS_ATHENA_OUTPUT=s3://chaincommand-data/athena-results/
CC_AWS_QUICKSIGHT_ACCOUNT_ID=123456789012
```

---

## Testing

```bash
# Run all tests (181)
pytest tests/ -v

# Run by module
pytest tests/test_models/          # ML model tests (23 tests)
pytest tests/test_kpi/             # KPI engine tests (8 tests)
pytest tests/test_optimization/    # CP-SAT optimizer tests (10 tests)
pytest tests/test_bom/             # BOM tree + manager tests (17 tests)
pytest tests/test_rl/              # RL inventory tests (12 tests)
pytest tests/test_ctb/             # CTB analyzer tests (8 tests)
pytest tests/test_risk/            # Risk scorer tests (11 tests)
pytest tests/test_property_based.py  # Hypothesis property tests (4 tests)
pytest tests/test_api/ -v          # API + security tests (28 tests)
pytest tests/test_aws/ -v          # AWS backend tests (52 tests)
pytest tests/test_events/ -v       # Event bus tests (2 tests)
pytest tests/test_orchestrator_lifecycle.py  # Orchestrator tests (3 tests)
pytest tests/test_mlflow_registry.py         # MLflow registry tests (3 tests)
```

| Test Module | Tests | Coverage |
|-------------|-------|----------|
| `test_models/` | 23 | LSTM, XGBoost, Ensemble, AnomalyDetector, GA, DQN, Hybrid |
| `test_kpi/` | 8 | KPI calculations, thresholds, violations |
| `test_optimization/` | 10 | CP-SAT allocation, constraints, sensitivity analysis |
| `test_bom/` | 17 | BOM tree operations, explosion, where-used, cost rollup, critical path |
| `test_rl/` | 12 | Gymnasium env, (s,S) baseline, Q-table trainer, PPO fallback policy |
| `test_ctb/` | 8 | Full availability, shortages, on-order, material costs |
| `test_risk/` | 11 | Rule-based scoring, ML training, recommendations, ranking |
| `test_property_based.py` | 4 | Hypothesis: KPI ranges, optimizer constraints, risk scores, BOM generation |
| `test_api/` | 28 | Dashboard, control, security, CORS, rate limiting |
| `test_aws/` | 52 | All AWS clients fully mocked |
| `test_events/` | 2 | Event bus pub/sub |
| `test_orchestrator_lifecycle.py` | 3 | Orchestrator lifecycle management |
| `test_mlflow_registry.py` | 3 | MLflow model registry |
| **Total** | **181** | Full coverage across all modules |

All tests use mocked dependencies — no real AWS credentials, GPU, or external services required.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.11+ |
| **Data Models** | Pydantic 2.0+, Pydantic Settings |
| **Data Processing** | pandas, numpy |
| **ML/Statistics** | scikit-learn (Isolation Forest, RandomForest), custom LSTM/XGB |
| **Constraint Optimization** | OR-Tools CP-SAT (MILP) |
| **Reinforcement Learning** | Gymnasium, Stable-Baselines3 (PPO) |
| **BOM/CTB** | Custom multi-tier BOM engine |
| **API Server** | FastAPI, uvicorn |
| **Async Runtime** | asyncio (native Python) |
| **Logging** | structlog (structured, ISO 8601) |
| **Metrics** | prometheus-client |
| **AWS (optional)** | boto3, redshift-connector, pyarrow (S3, Redshift, Athena, QuickSight) |
| **Testing** | pytest, hypothesis, pytest-asyncio, pytest-cov |
| **Deployment** | Docker, docker-compose, Kubernetes, Helm, Terraform |
| **Configuration** | Environment variables (CC_ prefix), .env file |

---

## Enterprise Deployment Infrastructure

ChainCommand includes production-grade deployment infrastructure spanning three maturity phases.

### Infrastructure Structure

```
├── k8s/                        # Kubernetes manifests
│   ├── namespace.yaml          # Namespace isolation
│   ├── configmap.yaml          # Application configuration
│   ├── secret.yaml             # Sensitive credentials
│   ├── deployment.yaml         # 2-replica deployment with health probes
│   ├── service.yaml            # ClusterIP service
│   ├── hpa.yaml                # Horizontal Pod Autoscaler (2-10 pods)
│   ├── ingress.yaml            # Nginx ingress controller
│   ├── postgres.yaml           # PostgreSQL 16 StatefulSet
│   ├── redis.yaml              # Redis 7 deployment
│   └── canary/                 # Istio + Flagger canary deployment
├── helm/chaincommand/          # Helm chart
├── serving/                    # BentoML model serving
├── monitoring/                 # Prometheus + Grafana
├── pipelines/                  # Airflow DAGs (training + monitoring)
├── mlflow/                     # Model registry
├── terraform/                  # AWS IaC (VPC + EKS + RDS + S3)
├── loadtests/                  # k6 performance tests
└── data_quality/               # Great Expectations validation
```

### Phase 1 — Minimum Viable Deployment

| Component | Technology | Details |
|-----------|-----------|---------|
| **Container Orchestration** | Kubernetes | 2-replica deployment, liveness/readiness probes, HPA (2-10 pods) |
| **Helm Chart** | Helm v3 | Parameterized values for all environments |
| **Model Serving** | BentoML | Inference service: `forecast`, `detect_anomalies`, `optimize` |
| **Database** | PostgreSQL 16 | StatefulSet with persistent volume |
| **Cache** | Redis 7 | In-memory cache for feature store & sessions |

### Phase 2 — Production Ready

| Component | Technology | Details |
|-----------|-----------|---------|
| **Model Registry** | MLflow | Version management, stage transitions |
| **Metrics** | Prometheus | Custom metrics: request rate, latency, KPIs, RL reward, risk scores |
| **Dashboards** | Grafana | Multi-panel dashboard: requests, latency, KPIs, simulation |
| **Canary Deployment** | Istio + Flagger | Stepped rollout with success rate and P99 gates |
| **Pipeline Orchestration** | Apache Airflow | Training DAG (weekly) + monitoring DAG (6-hourly drift detection) |

### Phase 3 — Enterprise Grade

| Component | Technology | Details |
|-----------|-----------|---------|
| **Infrastructure as Code** | Terraform | AWS: VPC, EKS, RDS, ElastiCache, S3 (dev + prod) |
| **Load Testing** | k6 | Ramp to 100 VUs, P95 < 500ms, error rate < 1% |
| **SLO** | YAML definitions | API availability 99.9%, latency P95, forecast accuracy |
| **Data Quality** | Great Expectations | Demand history + inventory status validation rules |

### Quick Start — Local Infrastructure

```bash
# Core services (app + PostgreSQL + Redis)
docker compose up -d

# Monitoring (Prometheus + Grafana)
docker compose -f monitoring/docker-compose.monitoring.yaml up -d

# MLflow Model Registry
docker compose -f mlflow/docker-compose.mlflow.yaml up -d

# Airflow Pipeline Orchestration
docker compose -f pipelines/docker-compose.airflow.yaml up -d

# Kubernetes (local)
kubectl apply -f k8s/
# Or with Helm:
helm install chaincommand helm/chaincommand/

# Load Testing
k6 run loadtests/k6_api.js
```

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- CP-SAT formulation based on OR-Tools constraint programming best practices
- PPO implementation via Stable-Baselines3
- Architecture informed by JD.com's autonomous supply chain research
- Built upon [ChainInsight](https://github.com/hsinnearth7/ChainInsight) — our supply chain analytics predecessor project

---

<div align="center">

**Optimized by algorithms, driven by data.**

</div>
