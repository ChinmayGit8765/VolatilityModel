# VolForecast — Crypto + Stock Volatility Forecasting MLOps Platform

A short-horizon realized volatility forecasting system covering crypto (BTC, ETH) and
equities (SPY + large caps), wrapped in a full production MLOps lifecycle: ingestion,
validation, feature engineering, classical + ML models, serving, monitoring, drift
detection, and automated retraining.

**Portfolio goal:** Honestly benchmark an ML volatility model against the correct
classical baselines (GARCH(1,1)/EWMA) under leak-free walk-forward evaluation, inside
a genuine end-to-end MLOps lifecycle — every screening question ("deploy", "CI/CD for ML",
"drift + retraining", "feature engineering at scale") gets a repo-backed answer.

---

## Running the stack

### Prerequisites

- Docker Desktop 28+ with WSL2 backend (Windows 11) or Docker Engine (Linux/macOS)
- [uv](https://docs.astral.sh/uv/) package manager
- Git + [DVC](https://dvc.org/) (installed via `uv sync`)

### 1. Start the infrastructure stack

```bash
# Copy credentials template and optionally edit
cp infra/.env.example infra/.env

# Start Postgres + MLflow + Prefect (all four services)
docker compose -f infra/docker-compose.yml up -d

# MLflow tracking UI:  http://localhost:5000
# Prefect orchestration UI:  http://localhost:4200
```

> **Note (Windows 11):** PostgreSQL binds to host port 5433 (not 5432) to avoid
> conflicts with other local Postgres instances. MLflow and Prefect communicate
> on port 5432 within the Docker network — this only affects host-side `psql` access.

Wait ~30 seconds for all services to become healthy, then verify:

```bash
docker compose -f infra/docker-compose.yml ps
# All four services should show Status: Up (healthy)
```

### 2. Install Python dependencies

```bash
uv sync --dev
```

This installs the `volforecast` package plus all dev dependencies from the locked
`uv.lock` file (reproducible across machines and CI).

### 3. Ingest OHLCV data

```bash
# Ingest all 5 assets (2 crypto + 3 equity) from 2022-01-01 to today
uv run volforecast ingest --start 2022-01-01

# Or ingest a single asset
uv run volforecast ingest --symbol BTC/USDT --start 2022-01-01
uv run volforecast ingest --symbol SPY --start 2022-01-01
```

The ingest pipeline:
1. Fetches raw OHLCV from ccxt/Binance (crypto) and yfinance (equity)
2. Writes raw parquet to `data/raw/{asset_class}/{symbol}.parquet`
3. Runs `validate_asset` gate (Pandera + calendar checks) for every asset
4. Writes validated data to `data/processed/{asset_class}/{symbol}.parquet`
5. Rejected assets are quarantined to `data/quarantine/` — no invalid data reaches processed

### 4. Reproduce data at any commit (DVC)

```bash
# Restore data to the exact state recorded at the current commit
dvc checkout

# Or pull from remote cache (if configured)
dvc pull
```

Raw and processed datasets are DVC-tracked (`data/raw.dvc`, `data/processed.dvc`).
The `.dvc` pointer files are committed to git; actual parquet files are gitignored.

### 5. Tear down

```bash
docker compose -f infra/docker-compose.yml down
# Add -v to also remove named volumes (postgres_data, mlflow_artifacts)
```

---

## Development

### Run tests

```bash
# All tests (fixture-only, no live API calls)
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/unit/test_pipeline.py -v
```

### Lint and format

```bash
uv run ruff check .          # lint
uv run ruff format --check . # format check
uv run ruff format .         # auto-format
```

### CI

Every push triggers GitHub Actions CI (`.github/workflows/ci.yml`):
- `ruff check .` — lint
- `ruff format --check .` — format check
- `pytest tests/ -x -q` with `VOLFORECAST_NO_LIVE_API=1` — fixture-only tests

No live API calls are permitted in CI.

---

## Architecture

```
config/assets.yaml
       |
       v
volforecast ingest CLI
   /             \
ccxt adapter    yfinance adapter
(BTC, ETH)      (SPY, AAPL, MSFT)
   \             /
    v           v
  data/raw/{asset_class}/{symbol}.parquet
       |
       v  (DVC tracks data/)
  validate_asset gate --FAIL--> data/quarantine/ + skip
       |
       v PASS
  data/processed/{asset_class}/{symbol}.parquet

docker-compose stack (always-on services):
  postgres:5433 (host) / 5432 (internal)
     |-- mlflow-server:5000  (--backend-store-uri postgresql://...)
     |-- prefect-server:4200 (PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://...)
     |-- prefect-worker (PREFECT_API_URL=http://prefect-server:4200/api)
  named volumes: postgres_data, mlflow_artifacts
```

---

## Tech Stack

| Component | Library/Version | Purpose |
|-----------|-----------------|---------|
| Ingestion | ccxt 4.5.x / yfinance 1.4.x | Crypto + equity OHLCV fetch |
| Validation | Pandera 0.31.x | Schema + calendar + OHLC gate |
| Calendar | exchange-calendars 4.13.x | NYSE session validation |
| Tracking | MLflow 3.13.0 | Experiment tracking + model registry |
| Orchestration | Prefect 3.7.x | DAG scheduling + event-driven retrain |
| Backend | PostgreSQL 16 | MLflow + Prefect metadata store |
| Data versioning | DVC 3.67.x | Parquet versioning alongside git |
| Models | arch (GARCH), LightGBM 4.6 | Classical baselines + ML regressor |
| Serving | FastAPI 0.136.x | Real-time inference endpoint |
| Monitoring | Evidently 0.7.x | Drift detection |
| CI/CD | GitHub Actions + uv | Lint + fixture-only tests on every push |

---

## Data

- **Raw:** `data/raw/crypto/BTC-USD.parquet`, `ETH-USD.parquet`; `data/raw/equity/SPY.parquet`, `AAPL.parquet`, `MSFT.parquet`
- **Processed:** `data/processed/` — validated subsets ready for feature engineering
- **Quarantine:** `data/quarantine/` — validation failures with row-level reasons
- All data directories are gitignored; DVC `.dvc` pointer files are committed

---

## License

MIT
