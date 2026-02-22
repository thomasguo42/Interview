# Stage 2 Runbook

This runbook executes the implementation in `docs/stage-2-backtest-baselines.md`.

Assumption: you will implement a `stock-stage2` CLI (mirroring the Stage 1 `stock-stage1` CLI) to run benchmarks/backtests and write artifacts.

## 1) Configure

- Create/edit `config/stage2.yaml`:
  - Capital (default: 100000)
  - Rebalance schedule (weekly, last trading day of week)
  - Target positions (default: 35, cap 50)
  - Turnover throttle (default: max 15 names changed per rebalance)
  - Liquidity limit (default: 1% of ADV$20 as-of decision_date)
  - Cost scenario selection: base vs conservative
  - Backtest date range + holdout split (recommend: last 2 full calendar years; if end_date is mid-year, treat that year as optional forward reporting only)

## 2) Install

```bash
pip install -e .[dev]
```

## 3) Run backtests

Recommended order:

1) Benchmarks:
```bash
stock-stage2 benchmark --config config/stage2.yaml --benchmark SPY
stock-stage2 benchmark --config config/stage2.yaml --benchmark BIL
```

2) Baselines (base costs first):
```bash
stock-stage2 backtest --config config/stage2.yaml --strategy MOM_126x21 --costs base
stock-stage2 backtest --config config/stage2.yaml --strategy INS_90     --costs base
stock-stage2 backtest --config config/stage2.yaml --strategy INST_CHG   --costs base
stock-stage2 backtest --config config/stage2.yaml --strategy COMBO_RAW  --costs base
stock-stage2 backtest --config config/stage2.yaml --strategy COMBO_OVERLAY --costs base
```

3) Re-run key baselines under conservative costs:
```bash
stock-stage2 backtest --config config/stage2.yaml --strategy MOM_126x21 --costs conservative
stock-stage2 backtest --config config/stage2.yaml --strategy COMBO_RAW  --costs conservative
stock-stage2 backtest --config config/stage2.yaml --strategy COMBO_OVERLAY --costs conservative
```

## 4) Check outputs

Suggested layout (from the Stage 2 checklist):
- `data/backtests/<strategy>/<cost_scenario>/nav_daily.parquet`
- `data/backtests/<strategy>/<cost_scenario>/trades.parquet`
- `data/backtests/<strategy>/<cost_scenario>/holdings.parquet`
- `data/backtests/<strategy>/<cost_scenario>/summary.json`
- `data/backtests/<strategy>/<cost_scenario>/report.md`
