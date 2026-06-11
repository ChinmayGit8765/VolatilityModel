#!/bin/bash
# MLflow server entrypoint for docker-compose.
# Starts MLflow with --allowed-hosts '*' to allow connections from the Docker
# host on Windows 11 (MLflow 3.x defaults to localhost-only).  The host port is
# bound to 127.0.0.1 in docker-compose.yml, so this does not expose the server
# beyond the local machine.
# psycopg2-binary is baked into the image at build time (see mlflow/Dockerfile)
# rather than pip-installed here, so startup never depends on PyPI.
#
# --artifacts-destination + --serve-artifacts: artifact-proxy mode (Plan 03-01).
# Clients (training scripts, serving container) upload/download artifacts via
# HTTP through this tracking server.  No direct volume mount to mlflow_artifacts
# is required from client-side containers.  Resolves Open Question #1 / A4.
set -e

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --artifacts-destination /mlflow/artifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts '*'
