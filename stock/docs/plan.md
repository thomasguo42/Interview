# Stock Prediction System Plan (Sharadar Core US Equities Bundle)

Last updated: 2026-02-20

## Objective

Build a robust, point-in-time stock selection system that produces actionable weekly trade suggestions (buy/sell + sizing) for a human trader.

Primary target:
- Long-only equities portfolio
- 30-40 positions (hard cap 50)
- Weekly rebalance
- Predict 20 trading-day (20d) forward returns (cross-sectional ranking)
- Must beat SPY total return net of realistic costs

Secondary target:
- A 60d horizon variant for robustness checks

## Inputs

Data source: Sharadar Core US Equities Bundle via Nasdaq Data Link API:
- Core US Fundamentals Data
- Core US Insiders Data
- Core US Institutional Investors Data
- Sharadar Equity Prices
- Sharadar Fund Prices (ETFs)

## Outputs

Weekly rebalance report (for a human):
- Universe + timestamp (as-of date) used for decisions
- Buy list and sell list (tickers + permaticker)
- Target weights and estimated trade sizes
- Expected return score / rank and confidence score
- Risk metrics (vol, max weight, sector exposure)
- Estimated transaction costs (base + conservative scenarios)
- Short rationale (“top drivers” / key features)

Optional daily watchlist:
- Best candidates (no forced trading) and any risk-off warnings

## Non-Negotiables (robustness contract)

- Point-in-time only: no lookahead from revisions, filings, or future knowledge.
- Survivorship-safe: delisted symbols must appear historically.
- Universe selection must be computed as-of each date.
- Costs must be modeled; results must survive a conservative cost scenario.
- Validation must be time-aware (walk-forward). Keep a final holdout period untouched.

## Default Trading Spec (frozen v0)

These defaults can be parameterized later, but treat them as fixed for the first implementation.

### Strategy
- Long-only, weekly rebalance
- 20d forward-return ranking model
- Portfolio size: 30-40 names (cap 50)

### Universe (stocks + ETFs)
- US-listed common stocks + ETFs
- Exclude OTC, preferreds, warrants, and most ADRs by default (based on master security metadata)
- Initial liquidity/quality screens (tunable):
  - Price > $5
  - 20d average daily dollar volume (ADV$) > $10M
  - Market cap > $300M (point-in-time if available)

### Execution (human-friendly)
- Decision time: after close on rebalance day
- Execution assumption: next trading day open (or a conservative fill proxy)
- Order style suggestion: limit/marketable-limit

### Costs (two scenarios)
- Base:
  - Equities: 10 bps per side
  - ETFs: 5 bps per side
- Conservative:
  - Equities: 25 bps per side
  - ETFs: 10 bps per side

### Risk overlay (don’t hide in cash)
- Two-sleeve design:
  - Alpha sleeve: stock picks
  - Defensive sleeve: SPY and/or BIL/SHY/IEF/TLT
- Bias: stay invested most of the time
- Use defensive sleeve primarily when:
  - Signal conviction is weak (prefer SPY over cash), or
  - Risk regime turns adverse (trend/vol/drawdown rules)

## Implementation Stages

### Stage 1: Data Foundation + Point-in-Time Contract
Goal: build a reproducible local dataset and the point-in-time join mechanism.

Deliverables:
- Local raw + cleaned data lake (prices, fundamentals, insiders, institutions, master tickers)
- Trading calendar (NYSE) and corporate-action/adjustment policy
- Daily universe table (as-of) with liquidity stats
- Daily point-in-time panel builder (as-of join for fundamentals/insiders/institutions)
- Validation suite: leakage checks + survivorship checks

Definition of done:
- Can build a daily panel for multi-year history with no missing-key failures
- Random audits confirm every joined record’s knowledge_date <= asof_date
- Universe filters remove microcaps and don’t rely on “today-only” membership
- SPY/BIL/SHY/IEF/TLT available with coherent histories (or inception handled explicitly)

### Stage 2: Backtest Engine + Baseline Strategies
Goal: a walk-forward simulator to validate the data contract and establish baselines.

Implementation checklist:
- `docs/stage-2-backtest-baselines.md`

Deliverables:
- Weekly rebalance backtester (time-aware, realistic fills, costs, constraints)
- Benchmarks: SPY total return (+ optional BIL cash proxy)
- Baselines (sanity): momentum/reversal + simple insider/institutional signals (value/quality optional once SF1 column set is expanded)

Definition of done:
- Backtests run end-to-end on baselines and match expected properties (no “too good to be true” results)
- Costs and liquidity constraints materially impact results (good sign of realism)

### Stage 3: Feature Store + Labels
Goal: standardize the feature matrix and label generation for ML.

Deliverables:
- Feature definitions (prices/volume, fundamentals, insiders, institutions, regime/ETF features)
- Labels: 20d forward returns (and 60d for robustness)
- Dataset splits: rolling walk-forward and final holdout

### Stage 4: ML Ranking Models (leakage-safe)
Goal: train and validate a cross-sectional ranking model.

Deliverables:
- Baseline ML models (regularized linear, gradient-boosted trees)
- Walk-forward evaluation with purging around label horizons
- Confidence calibration (for exposure and turnover control)

### Stage 5: Portfolio Construction + Risk Overlay
Goal: convert rankings into a tradable portfolio under constraints.

Deliverables:
- Weighting: equal-weight or volatility-scaled, max 5% per name
- Optional sector caps (e.g., 25%)
- Turnover throttles (target 10-15 name changes per rebalance)
- Defensive sleeve rules (SPY/BIL/SHY/IEF/TLT) with an exposure floor

### Stage 6: Reporting, Paper Trading, Monitoring
Goal: produce actionable reports and safety monitoring before real trading.

Deliverables:
- Weekly report (CSV/HTML) with buys/sells, weights, costs, rationale
- Paper-trade mode and performance tracking vs SPY
- Alerts: missing data spikes, drift, and unexpected turnover
