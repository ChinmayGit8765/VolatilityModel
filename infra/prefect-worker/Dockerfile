FROM python:3.12-slim

# OpenMP runtime required by LightGBM on slim images (no OpenMP in slim).
# The daily flow's retrain/eval tasks import lightgbm in-process and via
# subprocess scripts, so libgomp1 must be present or those tasks crash with:
# libgomp.so.1: cannot open shared object file (Pattern 8 / STACK.md).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv, then install third-party deps in a cache-friendly layer.
# Same split-sync pattern as infra/api/Dockerfile (WR-07): --no-install-project
# first (deps only), then a full sync once src/ is present. Because pyproject.toml
# and uv.lock are identical to the api/dashboard images, this slow layer is a
# Docker cache hit when those images were built first.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --frozen --no-install-project

# Copy source tree and asset config so the project installs and assets resolve.
COPY src/ ./src/
COPY config/ ./config/

# Install the project itself now that src/ exists (cheap — deps cached).
RUN uv sync --no-dev --frozen

# Make the volforecast package importable without venv activation.
ENV PYTHONPATH=/app/src

# The Prefect process worker runs the flow entrypoint resolved from the
# bind-mounted repo at /repo (see infra/docker-compose.yml prefect-worker
# volumes), so the project root for config/data resolution is /repo, NOT /app.
# These match the compose volumes (../:/repo:ro and ../data:/repo/data) and the
# source path recorded by scripts/create_deployment.py.
ENV VOLFORECAST_ROOT=/repo
ENV VOLFORECAST_DATA_ROOT=/repo/data

# MLflow tracking URI (compose DNS name) — overridable at run time.
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Prediction log path inside the bind-mounted data volume (Pitfall 7).
ENV PREDICTION_LOG_PATH=/repo/data/predictions/predictions.parquet

# WR-08: run as a non-root user. uid 1000 matches the api/dashboard images so the
# /repo/data bind mount stays writable (ingest writes raw, labeller writes FVR).
RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Start a Prefect process worker on local-pool through the uv-managed venv so the
# flow run subprocess (sys.executable) inherits the full project environment.
# --no-sync: the env was fully built at image-build time; do not re-resolve.
# --type process auto-creates local-pool on first start (no separate create step).
CMD ["uv", "run", "--no-sync", "prefect", "worker", "start", "--pool", "local-pool", "--type", "process"]
