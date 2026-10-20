# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed for compiling C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY chaincommand/ chaincommand/

# Install to a target directory (no editable install)
RUN pip install --no-cache-dir --target=/install ".[all]"

# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Upgrade base packages with known CVEs
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade setuptools wheel

# Create a non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/
COPY --from=builder /build/chaincommand /app/chaincommand
COPY --from=builder /build/pyproject.toml /app/

# Ensure the app directory is owned by appuser
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"]

CMD ["python", "-m", "chaincommand", "--host", "0.0.0.0", "--port", "8000"]
