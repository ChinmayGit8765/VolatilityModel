#!/bin/bash
# MLflow server entrypoint for docker-compose.
# Installs psycopg2-binary and starts MLflow with --allowed-hosts '*'
# to allow connections from the Docker host on Windows 11.
# MLflow 3.x defaults to localhost-only; --allowed-hosts '*' opens it up.
set -e

pip install psycopg2-binary -q

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root /mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts '*'
