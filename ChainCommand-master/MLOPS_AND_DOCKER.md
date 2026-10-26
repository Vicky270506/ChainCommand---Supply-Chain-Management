# MLflow + Docker (Hybrid Inventory Management System)

## What was added
| Report requirement | Where it lives |
|---|---|
| Experiment tracking: params, metrics, datasets, artifacts | `chaincommand/mlops/training.py` (every training run) |
| Model versions, signature, input example | `mlflow.pyfunc` models in `chaincommand/mlops/pyfunc_models.py` |
| Registry + champion/challenger promotion | `chaincommand/mlops/registry.py` (aliases `champion`, `challenger`, `previous_champion`) |
| Deployment info in the database | `mlflow_metadata` table (written on every registration/promotion) |
| Retraining: daily / weekly / monthly / event-driven | `chaincommand/mlops/retrain.py` (`scheduler`) |
| Monitoring: MAE, RMSE, sMAPE, bias, interval coverage, drift (PSI/KS), fill rate, stockout rate, approval rate | `chaincommand/mlops/monitoring.py` |
| API | `GET /api/mlops/models`, `GET /api/mlops/monitoring` (manager+), `POST /api/mlops/retrain` (admin) |

Flow: train challenger -> evaluate on the same hold-out window as the current champion -> promote only if better ->
forecast with the champion -> pending recommendations (each stores the forecast version that produced it).

## Run with Docker (needs Docker Desktop, ~4 GB RAM)
```powershell
docker compose up --build
```
First start takes several minutes (image build, seeding, first training). When `bootstrap` has exited and `api` is up:

* MLflow UI: http://localhost:5000 (experiments, runs, Models tab)
* API docs: http://localhost:8000/docs  -> login `admin@example.com` / `Demo12345`
* Optional `.env`: copy `.env.docker.example` to `.env` and change passwords.

Useful commands:
```powershell
docker compose logs -f scheduler                                   # retraining decisions
docker compose run --rm bootstrap python -m chaincommand.mlops train-all    # train a challenger now
docker compose run --rm bootstrap python -m chaincommand.mlops monitor      # monitoring report (also logged in MLflow)
docker compose down            # stop (keeps data)      |   docker compose down -v   # stop and delete all data
```

## Run without Docker
```powershell
pip install -e ".[all]"
$env:CC_DATABASE_URL = "postgresql+psycopg://postgres:YOURPASSWORD@localhost:5432/chaincommand"
python -m chaincommand.db.seed --reset
python -m chaincommand.db.create_user --demo
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --serve-artifacts
```
In a second terminal (same `CC_DATABASE_URL`):
```powershell
$env:CC_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m chaincommand.mlops train-all     # run twice: the 2nd (identical) model is rejected, the champion stays
python -m chaincommand.mlops plan          # champion forecasts + pending recommendations
python -m chaincommand.mlops monitor
python -m chaincommand.mlops scheduler --once
python -m chaincommand.mlops status
```
Without a server, leave `CC_MLFLOW_TRACKING_URI` unset: it uses `sqlite:///mlflow.db` + `./mlruns` (view with `mlflow ui`).

## Demo script (5 min)
1. MLflow UI -> `chaincommand-inventory` -> open a run: params, metrics (baselines vs LightGBM), dataset, artifacts, signature.
2. Models tab -> `demand-forecaster-h7`: versions and the `champion` alias.
3. `train-all` again -> new version is **rejected** (not better); show the `decision_reason` tag.
4. `/docs` -> `GET /api/mlops/models`; then `GET /api/mlops/monitoring`.
5. Recommendations carry `forecast_model_version = mlflow-v1-...` -> traceable to the exact model.

## Honest notes (put these in the report)
* The Docker **image build** was not run in the sandbox where this was written (no Docker daemon). The compose file
  validates (`docker compose config`), and every container command (MLflow server on PostgreSQL with artifact proxy,
  bootstrap, scheduler, API) was run natively against PostgreSQL 16 and MLflow 3.17.
* The original ChainCommand simulator (PPO training) still starts with the API, which is why `api` takes ~1 min.
* Monitoring scores the champion on the latest days it was not trained on, and on saved forecasts once real days pass.
  Seasonal peaks can look like drift; thresholds are in `config.py` (`mlops_*`).
* Registered model = trained on data before the hold-out window (default 30 days), so the comparison is fair.
* The RL policy is not yet registered in MLflow (the report's RL part is still the rule/(s,S) comparison).
