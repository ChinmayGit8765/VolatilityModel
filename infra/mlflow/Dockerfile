FROM ghcr.io/mlflow/mlflow:v3.13.0

# psycopg2 is required for the postgresql:// backend-store URI.
# Install it at image BUILD time with a pinned version (WR-06):
#   - the tracking server's availability no longer depends on PyPI being
#     reachable at every container (re)start
#   - the driver version cannot silently drift between restarts
RUN pip install --no-cache-dir psycopg2-binary==2.9.10
