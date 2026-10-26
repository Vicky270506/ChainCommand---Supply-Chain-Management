# One image, several roles (api / mlflow server / bootstrap / scheduler) - chosen by the compose command.

# ── Stage 1: build a virtualenv with every dependency ────────────
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build

# CPU-only PyTorch (the default wheel bundles CUDA and adds ~2 GB; RL here runs on CPU)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml .
COPY chaincommand/ chaincommand/
RUN pip install ".[serve]"

# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime
# libgomp1: OpenMP runtime needed by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app
COPY chaincommand/ /app/chaincommand/
COPY pyproject.toml /app/
# artifacts/ holds the scheduler state; /mlflow/artifacts is used by the MLflow server container
RUN mkdir -p /app/artifacts /mlflow/artifacts && chown -R appuser:appuser /app /mlflow
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"]

CMD ["uvicorn", "chaincommand.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
