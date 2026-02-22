# Stock Pipeline Stage 1

Implementation of `docs/stage-1-data-foundation.md`:
- local raw/clean/manifests data lake
- NYSE trading calendar and `next_trading_day`
- schema-lock discovery against Nasdaq Data Link metadata
- master/ticker mapping ingestion
- equity + ETF price ingestion with ADV$20d
- fundamentals/insiders/institutions point-in-time ingestion (`knowledge_date`)
- daily tradable universe builder
- deterministic as-of join primitive and daily panel builder
- validation suite (leakage, survivorship, quality)
- validation includes panel coverage, date coverage, and universe/panel market-cap alignment checks

## Prerequisites

- Python 3.11+
- `NASDAQ_DATA_LINK_API_KEY` set in environment

## Install

```bash
pip install -e .[dev]
```

## Configuration

Default config is `config/stage1.yaml`.
It freezes:
- decision semantics (weekly, after-close decisions)
- execution semantics (next trading day)
- timezone (`US/Eastern`)
- local-zip-backed default start date (`2016-01-01`) aligned with bundled files
- universe thresholds and ETF allowlist
- base/conservative transaction costs
- dataset codes + field candidate mappings
- `prefer_local_zip` mode with explicit per-dataset `local_zip_path`
- per-dataset `enforce_lastupdated_cutoff` join policy (fundamentals default: `false`)

## CLI

```bash
stock-stage1 discover --config config/stage1.yaml
stock-stage1 build-calendar --config config/stage1.yaml
stock-stage1 ingest-master --config config/stage1.yaml
stock-stage1 ingest-prices --config config/stage1.yaml
stock-stage1 ingest-actions --config config/stage1.yaml
stock-stage1 ingest-facts --config config/stage1.yaml
stock-stage1 build-universe --config config/stage1.yaml
stock-stage1 build-panel --config config/stage1.yaml
stock-stage1 validate --config config/stage1.yaml
```

Run everything in one command:

```bash
stock-stage1 run-all --config config/stage1.yaml
```

## Output Layout

- `data/raw/`: immutable raw API pull pages (JSON)
- `data/clean/`: cleaned typed Parquet tables
- `data/manifests/pulls/`: pull manifests (params/page counts/rows)
- `data/manifests/schema/`: schema locks from metadata discovery
- `data/manifests/clean/`: clean table manifests
- `data/manifests/validation/`: Stage-1 validation report
- `data/manifests/runs/stage1_last_run.json`: last run status + failure stage if aborted

## Tests

```bash
pytest -q
```
