# Stage 1 Runbook

This runbook executes the implementation in `docs/stage-1-data-foundation.md`.

## 1) Configure

- Edit `config/stage1.yaml` dates/thresholds if needed.
- Set `nasdaq.prefer_local_zip: true` to use local `SHARADAR_*.zip` dumps first.
- Prefer explicit `nasdaq.datasets.<alias>.local_zip_path` for deterministic file selection.
- Export API key:

```bash
export NASDAQ_DATA_LINK_API_KEY="..."
```

## 2) Install

```bash
pip install -e .[dev]
```

## 3) Run steps

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

Or end-to-end:

```bash
stock-stage1 run-all --config config/stage1.yaml
```

## 4) Check outputs

- Raw pulls: `data/raw/`
- Clean tables: `data/clean/`
- Pull manifests: `data/manifests/pulls/`
- Schema locks: `data/manifests/schema/`
- Validation report: `data/manifests/validation/stage1_validation_report.json`
- Last run status: `data/manifests/runs/stage1_last_run.json`
