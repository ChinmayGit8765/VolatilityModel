#!/bin/bash
# MLflow server entrypoint for docker-compose.
# Starts MLflow with an EXPLICIT allowed-hosts list (WR-09): the wildcard '*'
# defeated MLflow 3.x's DNS-rebinding protection — a malicious website could
# point an attacker-controlled hostname at 127.0.0.1 and script requests
# against the unauthenticated loopback-bound server from the victim's browser.
# The explicit list covers host-side clients (localhost / 127.0.0.1) and
# compose-internal clients (mlflow-server DNS name), with and without port.
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
  --allowed-hosts 'localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow-server,mlflow-server:5000'
