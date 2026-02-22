# Stage 2: Backtest Engine + Baseline Strategies (Implementation Checklist)

Goal: build a leakage-safe, walk-forward backtest engine and baseline strategies to validate the Stage 1 point-in-time contract and establish realistic performance baselines (net of costs).

Last updated: 2026-02-20

Depends on:
- `docs/data-contract.md` (decision/execution/knowledge_date semantics)
- Stage 1 outputs in `data/clean/` (prices, universe, panel, calendar)

## Stage 2 Outputs (what you will have at the end)

1. Backtest configuration (new file recommended):
   - `config/stage2.yaml` (capital, rebalance schedule, constraints, costs, benchmark selection, strategy selection)

2. A reusable backtest engine:
   - Deterministic weekly rebalance simulator
   - Realistic fills (next open), costs, and liquidity constraints
   - Robust handling of missing prices, delistings, and ETFs/cash sleeve behavior

3. Baseline strategy implementations (non-ML):
   - Momentum and reversal baselines
   - Insider-activity baseline (from Stage 1 insider aggregates)
   - Institutional baseline (from Stage 1 institution aggregates; conservative lag already applied)
   - A simple blended baseline (z-score blend)

4. Backtest artifacts (persisted to disk):
   - Daily NAV time series
   - Rebalance snapshots (targets, fills, costs)
   - Trade blotter (orders/trades)
   - Holdings history (shares + weights)
   - Performance report vs benchmarks (SPY + optional BIL)

## Recommended Defaults (frozen v0)

These choices are the “recommended option” for Stage 2 to keep the simulator simple, conservative, and consistent with the Stage 1 contract.

### A) Timing and bars

- decision_date: weekly, after close
- execution_date: next NYSE trading day open
- Mark-to-market: each trading day close
- Trading calendar: use the Stage 1 NYSE calendar:
  - primary: `data/clean/trading_calendar_nyse.parquet`
  - fallback (older name): `data/clean/trading_calendar.parquet`
  - Use the same `next_trading_day` semantics as Stage 1.

### B) Price scale (recommended: adjusted-price backtest)

Use a single total-return-consistent price scale for fills and PnL:
- Use `close_adj` as the canonical close series.
- Create a synthetic adjusted open (and optionally high/low) using the same-day adjustment factor:
  - `adj_factor = close_adj / close`
  - `open_adj = open * adj_factor`

Notes:
- This avoids having to model splits/dividends explicitly and is typically “good enough” for daily-bar backtests.
- Keep the raw `open/close` fields for diagnostics only; do not mix raw and adjusted series in PnL.

### C) Portfolio constraints

- Positions: target 35 (cap 50)
- Weighting: equal-weight (Stage 2), with:
  - max weight per name: 5%
  - minimum weight per name: none (equal-weight implies a minimum given N)
- Turnover throttle (human-friendly):
  - max 15 name changes per rebalance (buys + sells; treat one “swap” as 2 orders)
- Liquidity constraint:
  - leakage-safe recommendation: cap by decision-date liquidity, not execution-date liquidity
  - do not trade more than 1% of `ADV$20` as-of decision_date (known after the decision close)

### D) Costs

Use the two scenarios already frozen in `config/stage1.yaml`:
- Base: equities 10 bps/side, ETFs 5 bps/side
- Conservative: equities 25 bps/side, ETFs 10 bps/side

Apply costs on executed notional:
- `cost = abs(trade_notional) * (bps_per_side / 10_000)`

### E) Benchmarks

- Primary benchmark: SPY total return (from SFP)
- Optional cash proxy: BIL total return (from SFP)

## Stage 2 Step-by-Step

### A) Freeze the backtest contract (do this first)

1. Write down the exact rebalance schedule rule:
   - Recommended: “last trading day of each week” as decision_date.
   - Practical definition: group NYSE trading days by ISO week and take the max date.

2. Lock the execution timing:
   - execution_date = `next_trading_day(decision_date)`
   - If execution_date is outside the data range, stop the simulation.

3. Lock the price policy:
   - Signals computed using prices up to decision_date close (`close_adj`).
   - Trades filled at execution_date open (`open_adj`).
   - Daily NAV marked on each trading day close (`close_adj`).

4. Document missing-price rules (recommended, deterministic):
   - If a security is missing `open_adj` on an execution_date, do not trade it that day.
   - If a held security is missing `close_adj` on a mark date:
     - Prefer marking it at the last available `close_adj` and flag as “stale price”.
     - If stale persists beyond a threshold (e.g., 5 trading days) or the security is past its last price date, force-liquidate at last available `close_adj` with an extra conservative cost penalty (e.g., +50 bps).
       - Deterministic proxy for “past last price date” (recommended): `data/clean/master_security.parquet:end_date` (resolved from `SHARADAR/TICKERS:lastpricedate` in the Stage 1 schema lock).
   - Persist counters/flags so you can audit how often this happens (it should be rare).

5. Keep the “no hiding in cash” intent explicit:
   - Stage 2 can allow a defensive sleeve, but require that the portfolio is invested (stocks and/or ETFs) most of the time.
   - Recommended: any unallocated capital goes to SPY by default (not cash).
   - In risk-off, allocate primarily to defensive ETFs, but keep a fixed SPY floor (e.g., 40%) so the system stays meaningfully invested.

6. Clarify turnover throttle semantics (make this precise before coding):
   - Apply the throttle to the *intended* target portfolio changes at decision time (pre-execution).
   - Execution constraints (missing `open_adj`, liquidity caps) can only reduce what actually trades; do not “replace” blocked trades with new names using execution-day information.
   - Initial rebalance (starting from all-cash) is exempt from the 15-name throttle; build directly to the target number of positions on the first execution.
   - Optional but recommended to reduce busywork: ignore tiny resize trades using a minimum trade-notional threshold (e.g., skip trades under $200 notional).
   - Throttle scope recommendation:
     - Apply the 15-name change limit to the *equity alpha sleeve* constituents only.
     - ETF sleeve adjustments (SPY/defensive) are allowed but should remain simple (at most a small number of ETF orders per rebalance) and are not counted toward the 15-name equity limit.

### B) Define Stage 2 inputs (use Stage 1 tables as-is)

7. Confirm required inputs exist in `data/clean/` and name them in one place (in the stage2 config or constants):
   - equity prices: `equity_prices_daily.parquet` (must include `open`, `close`, `close_adj`, `adv20_dollar`)
   - ETF prices: `etf_prices_daily.parquet` (same fields)
   - universe: `tradable_universe_daily.parquet` (eligible flag + screens)
   - panel: `daily_panel.parquet` (PIT features by `asof_date`)
   - insider aggregates: `insiders_daily_agg.parquet` (by `knowledge_date`)
   - institution aggregates: `institutions_daily_agg.parquet` (by `knowledge_date`)

8. Create a single “price accessor” abstraction (even if it’s just functions) that can:
   - fetch `open_adj/close_adj/adv20_dollar/vol` for (permaticker, date)
   - return “missing” deterministically (no silent forward-fills without flags)

9. Build the synthetic adjusted open fields once (recommended) and persist:
   - Create `equity_prices_daily_adj.parquet` and `etf_prices_daily_adj.parquet` OR add `open_adj` columns into the existing clean tables (choose one approach and be consistent).

### C) Implement the simulator core (accounting + execution)

10. Decide what state you will track:
   - cash balance
   - holdings: {permaticker -> shares}
   - last prices (for marking and stale detection)
   - cumulative costs

11. Define the rebalance loop:
   - For each decision_date:
     - Load the eligible universe (as-of decision_date)
     - Compute strategy scores using only data <= decision_date
     - Create target weights (respecting caps and max positions)
     - Apply turnover throttle (limit name changes)
     - Convert weights -> target shares using execution_date `open_adj`
     - Generate trades (delta shares), apply liquidity limit, apply costs
     - Update holdings and cash

12. Define the daily mark-to-market loop:
   - For each trading day (including execution dates):
     - If today is an execution_date: execute trades at open
     - Compute end-of-day NAV using `close_adj`
     - Record daily NAV, exposure, and any stale/missing flags

13. Round shares to realistic increments:
   - Recommended for Stage 2 realism: integer shares for equities/ETFs.
   - Keep any rounding residual in cash (small); optionally sweep residual into BIL at each rebalance if you want “always invested” behavior.

14. Keep the simulation deterministic:
   - stable sorting of tickers when scores tie
   - explicit random seeds if any randomness is introduced (prefer none in Stage 2)

### D) Implement benchmarks

15. SPY benchmark:
   - Buy SPY at the first execution_date open and hold.
   - Apply ETF transaction cost on entry (and optionally on exit if you close at end).

16. Optional BIL benchmark:
   - Same idea as SPY, used as a cash proxy in reports.

### E) Baseline strategies (start simple, then add)

Each strategy should follow a consistent interface:
- Input: decision_date, eligible universe, Stage 1 panel, price history up to decision_date
- Output: target weights for execution_date

Recommended baselines (in order):

17. MOM-126x21 (recommended first momentum baseline):
   - Score = 126-trading-day return excluding last 21 trading days.
   - Rank descending; pick top N; equal weight.
   - Rationale: standard momentum that avoids short-term reversal.

18. MOM-252x21 (longer horizon momentum):
   - Same as above but 252d lookback (if enough history exists).

19. REV-21 (short-term reversal):
   - Score = negative 21d return.
   - Rank descending; pick top N; equal weight.

20. INS-90 (insider activity baseline):
   - Use `insiders_daily_agg.parquet` keyed by `knowledge_date`.
   - Default definition: rolling 90 *calendar* day sum of `insider_net_dollar` per security, as-of decision_date.
   - Window anchor: use `knowledge_date` (availability-aware), not `transaction_date`.
   - Rank descending; pick top N; equal weight.
   - Implementation note (performance): compute the INS-90 signal at weekly decision dates only; avoid per-day scans.

21. INST-CHG (institutional change baseline):
   - Use `institutions_daily_agg.parquet` keyed by `knowledge_date`.
   - Default definition: percent change vs the previous observation on filing/knowledge dates:
     - Let `curr = institution_total_shares(t)` and `prev = institution_total_shares(prev_t)` where `prev_t` is the previous `knowledge_date` for that permaticker.
     - Compute `inst_chg = (curr - prev) / abs(prev)` with safe handling:
       - if `prev` is missing or `abs(prev) < 1`, set `inst_chg = NA`.
   - For standalone ranking, rank by `inst_chg` (descending) after dropping NAs.
   - For COMBO, winsorize `inst_chg` cross-sectionally at each decision_date before z-scoring (e.g., clip to [1st, 99th] percentile).
   - Rank descending; pick top N; equal weight.
   - Note: SF3 in this entitlement uses a conservative lag policy from Stage 1; keep it unchanged.

22. COMBO_RAW (simple blended baseline, no overlay):
   - z-score normalize each component cross-sectionally at decision_date:
     - momentum score
     - insider score
     - institution score
   - Winsorize (e.g., clip to [-3, +3]) before combining.
   - Combine weights (recommended):
     - 50% momentum
     - 25% insiders
     - 25% institutions
   - Rank by combined score; pick top N; equal weight.

### F) Risk overlay (keep it modest in Stage 2)

23. Implement a single conservative regime flag using only SPY prices:
   - Example (recommended): risk_off if:
     - SPY close_adj < SPY 200d moving average (computed up to decision_date), AND
     - SPY realized_vol20 is above its trailing 1-year median (also up to decision_date)

24. When risk_off is true:
   - Allocate some portion to defensive ETFs (choose one):
     - “Simple”: 100% BIL/SHY (capital preservation)
     - “Balanced”: 50% SPY, 50% IEF/TLT (stay invested but reduce risk)
   - Recommended for your “don’t sit in cash” requirement: keep a SPY floor (e.g., 30-50%) so the system doesn’t avoid equities for long stretches.

25. Scope recommendation:
   - For baseline comparisons, run each baseline strategy *without* the risk overlay first.
   - Implement the overlay as an optional wrapper and expose it as a separate strategy ID (e.g., `COMBO_OVERLAY` vs `COMBO_RAW`).

### G) Evaluation and reporting (must be part of Stage 2)

26. Produce metrics for each strategy and each cost scenario:
   - CAGR, annualized vol, Sharpe (and/or Sortino), max drawdown
   - turnover (% notional traded per year) and average weekly turnover
   - average number of positions and % time invested in equities vs ETFs vs cash
   - cost drag: total costs and bps/year

27. Split periods (industrial standard for this project; make it exact):
   - Holdout uses the last `holdout_years` *full calendar years* available.
   - If `end_date` is not year-end, exclude the partial final year from both dev and holdout (treat it as optional “forward” reporting only).
   - Concrete example with Stage 1 defaults:
     - Stage 1 `end_date`: 2026-02-16
     - Holdout years (2): 2024-01-01 to 2025-12-31
     - Development: 2016-01-01 to 2023-12-31
     - Optional forward (do not tune on it): 2026-01-01 to 2026-02-16
   - Report dev and holdout separately; forward is optional and must be labeled clearly.

28. Write artifacts to disk with a clear, repeatable naming convention:
   - `data/backtests/<strategy>/<cost_scenario>/nav_daily.parquet`
   - `data/backtests/<strategy>/<cost_scenario>/trades.parquet`
   - `data/backtests/<strategy>/<cost_scenario>/holdings.parquet`
   - `data/backtests/<strategy>/<cost_scenario>/summary.json`
   - `data/backtests/<strategy>/<cost_scenario>/report.md`

### H) Stage 2 validation (look for “too good to be true”)

29. Sanity checks that should hold:
   - Conservative costs reduce performance meaningfully.
   - Adding a 1-day delay (trade at execution_date+1 open) should generally reduce performance.
   - If a baseline has extreme Sharpe (>3) over long windows, investigate leakage immediately.

30. Spot audits:
   - Randomly sample 50 rebalances:
     - confirm every feature used has timestamp <= decision_date
     - confirm trades execute on execution_date (not on decision_date)
     - confirm `open_adj` and `close_adj` exist for executed trades

31. Revision-policy note (avoid confusion with SF1):
   - Stage 2 baselines do not require SF1 fundamentals.
   - If you use `daily_panel.parquet` for any SF1-derived features, you inherit the Stage 1 join policy:
     - `config/stage1.yaml` sets `fundamentals.enforce_lastupdated_cutoff: false` (bundle-specific behavior).
   - If/when you add fundamentals-based baselines, implement a strict variant explicitly and compare results.

### I) Performance notes (don’t accidentally build an O(days * rows) engine)

32. Avoid per-day full-table scans of multi-million-row Parquet:
   - Precompute the list of decision dates and execution dates once from the calendar.
   - Precompute score inputs (e.g., momentum returns) vectorized with groupby+shift, then slice only the decision dates.
   - Build fast lookups for prices on (permaticker, date) (e.g., set a MultiIndex or factorize to integer ids).
   - Load only needed columns from Parquet (avoid pulling full tables into memory unnecessarily).

## Stage 2 Definition of Done (copy/paste checklist)

- [ ] Rebalance schedule is deterministic and documented (weekly, decision after close, execute next open)
- [ ] Backtest uses a single coherent price scale (`close_adj` + synthetic `open_adj`)
- [ ] Simulator handles missing prices/delistings deterministically and logs exceptions
- [ ] Costs and liquidity constraints are implemented and materially affect results
- [ ] SPY (and optional BIL) benchmarks run with the same execution semantics
- [ ] At least 3 baselines (momentum, insiders, institutions) run end-to-end
- [ ] Reports compare dev vs holdout periods and base vs conservative costs
- [ ] No “too good to be true” results survive the leakage sanity checks
