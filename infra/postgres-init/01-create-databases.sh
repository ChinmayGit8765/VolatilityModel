#!/bin/bash
# Create the Prefect database alongside the MLflow database.
# Postgres entrypoint runs all *.sh scripts in /docker-entrypoint-initdb.d/ on first init.
# POSTGRES_DB is already created by the entrypoint (mlflowdb); we only need prefectdb.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE prefectdb OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prefectdb')\gexec
EOSQL
