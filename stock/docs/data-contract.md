# Data Contract (Point-in-Time) for Sharadar Bundle

This document defines the “no lookahead” rules and joining semantics used throughout the system.

Last updated: 2026-02-20

## Core Definitions

- decision_date:
  - The date we compute signals, after market close.
  - Example: Friday close for a weekly rebalance schedule.

- execution_date:
  - The next trading day when trades are assumed to be executed.
  - Example: Monday open (or next open if holiday).

- asof_date:
  - The date at which all information must have been available.
  - For this system: asof_date = decision_date (after close).

- knowledge_date:
  - The earliest date a record is allowed to influence the model.
  - Conservative rule: knowledge_date is the next trading day after the record’s availability timestamp.
  - Rationale: avoids same-day “close-to-close” leakage and filing-time ambiguity.

- point-in-time join:
  - For each (security, asof_date), select the latest record with knowledge_date <= asof_date.

Operational defaults (v0):
- Timezone: US/Eastern
- Rebalance frequency: weekly
- decision_date: rebalance day after close
- execution_date: next NYSE trading day open

## General “No Leakage” Rules

1. Never use any record whose availability timestamp is after asof_date.
2. Never use “latest revision” of a historical record if it was updated after asof_date.
3. Universe membership must be computed as-of each date; never filter to “currently listed”.
4. All features must be computed using data with timestamps <= asof_date only.
5. Labels are computed strictly from future prices (e.g., 20d forward returns) and never used as inputs.

## Dataset-Specific Rules (Sharadar)

This section uses semantic names. The concrete resolved field names are locked from Nasdaq Data Link metadata in:
- `data/manifests/schema/master.json`
- `data/manifests/schema/equity_prices.json`
- `data/manifests/schema/etf_prices.json`
- `data/manifests/schema/fundamentals.json`
- `data/manifests/schema/insiders.json`
- `data/manifests/schema/institutions.json`

Initial configured dataset codes and candidate field mappings live in `config/stage1.yaml`; schema lock output is the source of truth used by the pipeline.

### Master / Tickers / Security Reference

Purpose:
- Determine security type (common stock vs ETF vs ADR, etc.)
- Handle ticker changes (map ticker -> permaticker by date)
- Identify listing venue and basic filters (exclude OTC, etc.)

Rules:
- Primary ID: permaticker
- Maintain a mapping table from (ticker, date) -> permaticker
- Never filter securities based on “today’s” status when building historical universes

### Prices (Sharadar Equity Prices, Sharadar Fund Prices)

Purpose:
- Compute returns, vol, liquidity, and technical features

Rules:
- Use adjusted close series for return calculations (consistent total-return proxy)
- Use close as the decision price (decision after close)
- Execution assumptions in backtests should be next open (or a conservative proxy)

Backtest fill price policy (recommended v0):
- Canonical PnL scale: adjusted prices (total-return-consistent).
- Fill price at execution_date open:
  - Compute `adj_factor = close_adj / close` for that same execution_date bar.
  - Compute `open_adj = open * adj_factor` and execute at `open_adj`.
- Mark-to-market:
  - Use `close_adj` for end-of-day NAV.

Notes:
- This avoids explicitly modeling splits/dividends (they are embedded in `close_adj`), but it is still an approximation around corporate-action dates.
- Do not mix raw and adjusted series inside the same PnL calculation.

Quality checks:
- Missing days, duplicate rows, zero volume, extreme jumps (splits), inconsistent adjustments

### Fundamentals (Core US Fundamentals)

Purpose:
- Value/quality/growth features and stable company descriptors

Availability timestamp (must pick one based on actual columns):
- Prefer: filing/acceptance/publication date
- Fallback: period-end date plus a conservative delay (not ideal; only if necessary)

knowledge_date rule:
- knowledge_date = next_trading_day(availability_date)

Revision handling:
- If multiple revisions exist for the same economic period, do NOT use a revision with lastupdated > asof_date.
- When selecting “latest-known fundamentals”, use:
  - max(availability_date) subject to knowledge_date <= asof_date, and
  - lastupdated <= asof_date if lastupdated exists
- Operational note:
  - `config/stage1.yaml` now supports per-dataset `enforce_lastupdated_cutoff`.
  - For the current bundled SF1 files, default is `false` because `lastupdated` behaves like a late backfill timestamp and can suppress nearly all historical joins.

### Insiders (Core US Insiders)

Purpose:
- Insider net buying/selling signals and clustering

Availability timestamp:
- Filing date/time (or the closest available field)

knowledge_date rule:
- knowledge_date = next_trading_day(availability_date_proxy)
- Preferred availability_date_proxy: filing_date
- Fallback when filing_date is unavailable in this dataset: calendardate + conservative lag

Aggregation:
- Compute trailing window aggregates later (e.g., 30d/90d net shares, net $ volume, buy/sell ratio, role-weighted signals)

### Institutional Investors / 13F (Core US Institutional)

Purpose:
- Ownership level and change features

Availability timestamp:
- Filing date for the 13F report

Current bundle caveat (verified via live metadata on 2026-02-16):
- `SHARADAR/SF3` does not expose a filing-date column in this entitlement; it exposes `calendardate` and holdings fields.
- Stage-1 fallback policy is active: treat `calendardate` as the proxy availability anchor and apply a conservative lag before `knowledge_date` (configured in `config/stage1.yaml` as `availability_lag_days: 60`).
- This is intentionally conservative and should be replaced with explicit filing timestamps if/when a filing-date field/table is available.

knowledge_date rule:
- knowledge_date = next_trading_day(availability_date_proxy)

Reporting-lag consideration:
- The positions reflect a quarter-end snapshot but become known only at filing.
- Use changes between sequential filings as features; avoid assuming real-time portfolio adjustments.

## Universe Construction Contract

Universe = all securities eligible to trade on each asof_date.

Equities (default):
- US common stocks only
- Exclude OTC, preferreds, warrants, and most ADRs (based on master metadata)
- Liquidity/quality filters (tunable):
  - price > 5
  - ADV$20d > 10M
  - market cap > 300M (point-in-time if possible)

ETFs:
- Include ETFs for defensive/risk overlay even if they fail equity-specific rules.
- Maintain a minimum allowlist: SPY, BIL, SHY, IEF, TLT (can expand later).

## Required Validation (must pass before modeling)

Leakage audit:
- Randomly sample (security, asof_date) joins and assert:
  - joined_record.knowledge_date <= asof_date
  - if lastupdated exists: joined_record.lastupdated <= asof_date

Survivorship audit:
- Confirm that delisted securities appear historically and can be selected by the universe builder.

Universe sanity:
- Confirm the filtered universe has reasonable liquidity distribution and is not dominated by microcaps.

ETF sanity:
- Confirm allowlisted ETFs have continuous (or explicitly handled) histories.
