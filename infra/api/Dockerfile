FROM python:3.12-slim

# OpenMP runtime required by LightGBM on slim images (no OpenMP in slim).
# This MUST be installed before any LightGBM import or the container crashes
# with: libgomp.so.1: cannot open shared object file (Pattern 8 / STACK.md).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv (matches local dev + CI toolchain exactly)
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --frozen

# Copy source tree and asset config
COPY src/ ./src/
COPY config/ ./config/

# Python path so volforecast package is importable
ENV PYTHONPATH=/app/src

# VOLFORECAST_ROOT: used by load_assets to find config/assets.yaml at /app/config/assets.yaml
ENV VOLFORECAST_ROOT=/app

# VOLFORECAST_DATA_ROOT: the data directory (bind-mounted from the host).
# Processed parquets are read from /data/processed/; the prediction log is
# written to /data/predictions/ (via PREDICTION_LOG_PATH).
ENV VOLFORECAST_DATA_ROOT=/data

# MLflow tracking URI (compose DNS name) — overridable at run time
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Prediction log path inside the bind-mounted data volume (Pitfall 7)
ENV PREDICTION_LOG_PATH=/data/predictions/predictions.parquet

# Start the ASGI server bound to all interfaces on port 8000.
# The host-side compose binding (127.0.0.1:8000:8000) enforces loopback isolation.
CMD ["uv", "run", "uvicorn", "volforecast.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
