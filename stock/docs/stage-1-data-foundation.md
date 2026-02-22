# Stage 1: Data Foundation + Point-in-Time Contract (Implementation Checklist)

Goal: create the local data lake and the point-in-time (“as-of”) panel builder used by all later stages.

Last updated: 2026-02-16

## Stage 1 Outputs (what you will have at the end)

1. Local data lake:
   - Raw pulls (immutable) from Nasdaq Data Link API
   - Cleaned, typed, deduped tables in a columnar format (Parquet recommended)
   - A manifest file capturing pull parameters and row counts

2. Core reference tables:
   - Trading calendar (NYSE)
   - Master security/ticker table
   - Ticker -> permaticker mapping by date

3. Price tables:
   - Equity daily OHLCV (+ adjusted fields if available)
   - ETF daily OHLCV (+ adjusted fields if available)

4. Point-in-time datasets:
   - Fundamentals with knowledge_date
   - Insiders with knowledge_date
   - Institutions (13F) with knowledge_date

5. Tradable universe table:
   - Daily membership + the values used for screening (price, ADV$, market cap if available)

6. Daily “as-of panel” builder:
   - For each asof_date, (security, features) where every feature respects knowledge_date <= asof_date

7. Validation suite:
   - Leakage checks
   - Survivorship checks
   - Data quality checks

## Stage 1 Step-by-Step

### A) Freeze global semantics (do this first)

1. Choose and document:
   - decision_date: rebalance day after close (weekly)
   - execution_date: next trading day
   - timezone: US/Eastern

2. Define knowledge_date policy (conservative):
   - knowledge_date = next_trading_day(availability_date)
   - availability_date comes from filing/acceptance/publication dates (verify via API metadata)

3. Write these definitions into `docs/data-contract.md` (already drafted, adjust field names once verified).

### B) Project structure and configuration

4. Create a config file capturing:
   - start_date / end_date
   - holdout definition (recommended: last 2 calendar years available)
   - universe screens: price, ADV$, market cap thresholds
   - ETF allowlist: SPY, BIL, SHY, IEF, TLT
   - cost scenarios (base vs conservative)

5. Store the API key in an environment variable:
   - NASDAQ_DATA_LINK_API_KEY

6. Decide local storage layout:
   - raw/ (immutable API responses, partitioned by dataset and date range)
   - clean/ (Parquet tables)
   - manifests/ (pull manifests, schema snapshots, row counts)

### C) API discovery and schema locking (avoid guessing)

7. Call the Nasdaq Data Link API metadata endpoints to retrieve:
   - dataset codes you have access to (for each Sharadar product)
   - columns and their types
   - primary identifiers available (permaticker, ticker, date fields)

8. For each table you will use, record:
   - dataset code
   - date column(s)
   - permaticker column(s)
   - availability timestamp field(s): filing/acceptance/lastupdated/etc.

9. Update `docs/data-contract.md` with the exact column names you will treat as:
   - availability_date
   - lastupdated (if present)
   - security type / exchange / category fields

### D) Ingest master reference tables (build your ID spine)

10. Ingest the master tickers/security reference table first.

11. Clean it:
   - enforce a canonical permaticker type
   - standardize ticker strings (upper-case)
   - normalize security type/category fields

12. Build (or ingest) a ticker history mapping:
   - (ticker, date) -> permaticker
   - handle symbol changes by date range
   - ensure you can resolve tickers used in price tables to permaticker reliably

13. Create a security classification table:
   - common stock vs ETF vs other
   - listing venue (exchange) and OTC flags
   - country and ADR flags if present

### E) Ingest prices (equity + fund) and compute basic derived columns

14. Ingest equity daily prices over your chosen period (e.g., 2004+ to present).

15. Ingest fund/ETF daily prices for:
   - SPY, BIL, SHY, IEF, TLT at minimum
   - optionally all ETFs if available, but start with the allowlist

16. Standardize:
   - date column as a proper date (no time component)
   - numeric dtypes (floats/ints)
   - dedupe on (permaticker, date)
   - sort by (permaticker, date)

17. Adjustment policy:
   - pick adjusted close (and adjusted volume if needed) as the source for return computations
   - keep raw close for reporting if you want (but do not mix series in calculations)

18. Build basic “price features” needed for Stage 1 universe screens:
   - dollar volume = close * volume
   - ADV$20d = rolling mean dollar volume over 20 trading days (per security)
   - realized volatility (optional at this stage, useful later)

19. Data quality checks:
   - missing dates in the calendar for active tickers
   - zero or negative prices/volumes
   - extreme single-day jumps that indicate splits (should be handled by adjusted series)

### F) Ingest fundamentals / insiders / institutions with knowledge_date

20. Fundamentals:
   - ingest table(s) needed for value/quality/growth features
   - define availability_date using filing/acceptance/publication fields
   - compute knowledge_date = next_trading_day(availability_date)
   - handle revisions: never allow lastupdated > asof_date in later joins
   - dedupe on the correct natural key (verify via metadata)

21. Insiders:
   - ingest insider transactions/filings table(s)
   - availability_date = filing date
   - knowledge_date = next_trading_day(filing date)
   - keep enough fields to aggregate later (transaction type, shares, price if available, insider role if available)

22. Institutions (13F):
   - ingest filings/holdings table(s)
   - availability_date = filing date
   - knowledge_date = next_trading_day(filing date)
   - plan to compute ownership change features later (needs consistent holder IDs and security IDs)

### G) Trading calendar (NYSE) and next_trading_day function

23. Build a trading calendar table:
   - one row per trading day
   - flags for holidays/early closes if needed

24. Implement next_trading_day(date):
   - returns the next row in the trading calendar after the given date
   - must handle weekends/holidays

25. Verify next_trading_day correctness with spot checks around known holidays.

### H) Build the daily tradable universe table (as-of)

26. For each trading day, compute:
   - whether each security is eligible (security type filters)
   - price screen
   - ADV$20d screen
   - market cap screen (only if you have point-in-time market cap)

27. Persist universe membership as a daily table:
   - (asof_date, permaticker, eligible_flag, reasons/explanations)

28. Add ETFs:
   - ensure allowlisted ETFs are eligible even if they fail equity-specific rules

### I) Implement the point-in-time join primitive

29. Implement a reusable “as-of join” function:
   - Inputs: (panel_index: permaticker x asof_date), (fact_table with knowledge_date), join key(s)
   - Output: for each panel row, the latest fact record with knowledge_date <= asof_date

30. Make the join deterministic:
   - if multiple rows tie for latest knowledge_date, use a stable tie-breaker (e.g., max(lastupdated), then a stable row id)

31. Apply this join to create a daily panel table:
   - prices + liquidity stats + “latest known” fundamentals snapshot
   - “latest known” insider aggregates (even simple counts at first)
   - “latest known” institutional snapshot markers

### J) Validation (don’t skip)

32. Leakage tests:
   - sample random (permaticker, asof_date) rows and assert:
     - joined_record.knowledge_date <= asof_date
     - if lastupdated exists: joined_record.lastupdated <= asof_date

33. Survivorship tests:
   - pick a known delisted name (if you can identify one) and confirm it appears historically
   - confirm universe membership changes over time (not static)

34. Sanity tests:
   - the universe is not dominated by microcaps after screens
   - ETFs in allowlist have expected histories or inception handling

## Stage 1 Definition of Done (copy/paste checklist)

- [ ] API dataset codes and columns are recorded and stable
- [ ] Master security reference and ticker->permaticker mapping works across time
- [ ] Equity and ETF prices are ingested, typed, deduped, and adjusted consistently
- [ ] Fundamentals/insiders/institutions have knowledge_date and revision rules defined
- [ ] Universe membership is computed as-of each date with liquidity screens
- [ ] As-of join primitive exists and is used to build daily panels
- [ ] Leakage and survivorship audits pass on spot checks

