FROM python:3.12-slim

# OpenMP runtime required by LightGBM on slim images (LightGBM is a transitive
# import via data_access -> mlflow -> lightgbm model artifacts).
# Must be installed BEFORE any LightGBM import — identical to infra/api/Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv then install third-party deps in a separate layer from the project
# code — this keeps the (slow) dependency layer cached across src/ changes.
# WR-07 pattern: --no-install-project in the first sync, then full sync once src/ exists.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --frozen --no-install-project

# Copy source tree and asset config
COPY src/ ./src/
COPY config/ ./config/

# Now that src/ exists, install the project itself (cheap — deps already cached).
RUN uv sync --no-dev --frozen

# Python path so volforecast package is importable
ENV PYTHONPATH=/app/src

# VOLFORECAST_ROOT: used by load_assets() to find config/assets.yaml at /app/config/
ENV VOLFORECAST_ROOT=/app

# VOLFORECAST_DATA_ROOT: bind-mounted data directory (read by all loaders)
ENV VOLFORECAST_DATA_ROOT=/data

# MLflow tracking URI — overridable at compose runtime
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# API base URL — overridable at compose runtime
ENV API_BASE_URL=http://api:8000

# WR-08: run as a non-root user.  Mirrors infra/api/Dockerfile security posture.
# A compromised dashboard process can no longer write as root through the /data
# bind mount.  The dashboard is read-only by design, so this is belt-and-suspenders.
RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Start the Streamlit server bound to all interfaces on port 8501.
# The host-side compose binding (127.0.0.1:8501:8501) enforces loopback isolation.
# --no-sync prevents uv from re-resolving/installing on container start.
CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/volforecast/dashboard/app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]
