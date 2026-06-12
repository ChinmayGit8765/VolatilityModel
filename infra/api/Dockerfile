FROM python:3.12-slim

# OpenMP runtime required by LightGBM on slim images (no OpenMP in slim).
# This MUST be installed before any LightGBM import or the container crashes
# with: libgomp.so.1: cannot open shared object file (Pattern 8 / STACK.md).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv (matches local dev + CI toolchain exactly).
# WR-07: --no-install-project installs only third-party deps in this layer —
# the volforecast project itself is a hatchling root package that needs src/
# present, so a plain `uv sync --frozen` here breaks fresh builds.  Splitting
# also keeps the (slow) dependency layer cached across src/ changes.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --frozen --no-install-project

# Copy source tree and asset config
COPY src/ ./src/
COPY config/ ./config/

# Now that src/ exists, install the project itself (cheap — deps are cached).
RUN uv sync --no-dev --frozen

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

# WR-08: run as a non-root user.  A compromised service process can then no
# longer write as root through the /data bind mount into the host repo, nor
# modify the installed environment under /app.
# Tradeoff (documented): /data stays writable by design — the prediction log
# lives there (Phase 4 contract), and Docker Desktop's WSL2 file sharing maps
# host bind mounts world-writable anyway.  Acceptable for a loopback-only
# local dev stack; a cloud deploy should use a dedicated volume instead.
RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Start the ASGI server bound to all interfaces on port 8000.
# The host-side compose binding (127.0.0.1:8000:8000) enforces loopback isolation.
# WR-07: --no-sync prevents uv from re-resolving/re-installing the environment
# at container start — the env was fully built at image build time.
CMD ["uv", "run", "--no-sync", "uvicorn", "volforecast.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
