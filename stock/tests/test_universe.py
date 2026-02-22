from pathlib import Path

import pandas as pd

from stock_pipeline.config import load_config
from stock_pipeline.universe import build_tradable_universe


def _write_config(path: Path) -> None:
    path.write_text(
        f"""
project:
  timezone: US/Eastern
  decision_timing: after_close
  rebalance_frequency: weekly
  execution_timing: next_trading_day_open
  start_date: 2024-01-01
  end_date: 2024-12-31
  holdout_years: 2
storage:
  raw_root: {path.parent / 'raw'}
  clean_root: {path.parent / 'clean'}
  manifest_root: {path.parent / 'manifests'}
universe:
  min_price: 5.0
  min_adv20_dollar: 10000000.0
  min_market_cap: 300000000.0
etf_allowlist: [SPY, BIL, SHY, IEF, TLT]
costs:
  base:
    equities_bps_per_side: 10
    etfs_bps_per_side: 5
  conservative:
    equities_bps_per_side: 25
    etfs_bps_per_side: 10
nasdaq:
  base_url: https://data.nasdaq.com/api/v3
  api_key_env: NASDAQ_DATA_LINK_API_KEY
  page_size: 10000
  datasets:
    master:
      datatable: SHARADAR/TICKERS
      filters: {{}}
      required_columns: []
      field_candidates: {{}}
    equity_prices:
      datatable: SHARADAR/SEP
      filters: {{}}
      required_columns: []
      field_candidates: {{}}
    etf_prices:
      datatable: SHARADAR/SFP
      filters: {{}}
      required_columns: []
      field_candidates: {{}}
    fundamentals:
      datatable: SHARADAR/SF1
      filters: {{}}
      required_columns: []
      field_candidates: {{}}
    insiders:
      datatable: SHARADAR/SF2
      filters: {{}}
      required_columns: []
      field_candidates: {{}}
    institutions:
      datatable: SHARADAR/SF3
      filters: {{}}
      required_columns: []
      field_candidates: {{}}
"""
    )


def test_universe_builder_applies_screens_and_etf_override(tmp_path: Path):
    cfg_path = tmp_path / "stage1.yaml"
    _write_config(cfg_path)

    clean_root = tmp_path / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)

    master_security = pd.DataFrame(
        {
            "permaticker": [1, 2, 3, 3],
            "ticker": ["AAA", "BBB", "CCC", "CCC"],
            "start_date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2024-01-01"),
            ],
            "end_date": [
                pd.Timestamp("2025-12-31"),
                pd.Timestamp("2025-12-31"),
                pd.Timestamp("2023-12-31"),
                pd.Timestamp("2025-12-31"),
            ],
            "security_type_raw": ["Common Stock", "Common Stock", "Common Stock", "ADR"],
            "exchange": ["NYSE", "NASDAQ", "NYSE", "NYSE"],
            "country": ["USA", "USA", "USA", "USA"],
            "is_otc": [False, False, False, False],
            "is_adr": [False, False, False, True],
            "is_etf": [False, False, False, False],
            "is_common_stock": [True, True, True, False],
        }
    )
    master_security.to_parquet(clean_root / "master_security.parquet", index=False)

    equity_prices = pd.DataFrame(
        {
            "permaticker": [1, 2, 3, 3],
            "ticker": ["AAA", "BBB", "CCC", "CCC"],
            "date": [
                pd.Timestamp("2024-01-10"),
                pd.Timestamp("2024-01-10"),
                pd.Timestamp("2023-06-01"),
                pd.Timestamp("2024-06-03"),
            ],
            "close_for_returns": [10.0, 4.0, 20.0, 20.0],
            "adv20_dollar": [20_000_000.0, 20_000_000.0, 30_000_000.0, 30_000_000.0],
            "dollar_volume": [20_000_000.0, 20_000_000.0, 30_000_000.0, 30_000_000.0],
            "open": [10.0, 4.0, 20.0, 20.0],
            "high": [10.0, 4.0, 20.0, 20.0],
            "low": [10.0, 4.0, 20.0, 20.0],
            "close": [10.0, 4.0, 20.0, 20.0],
            "close_adj": [10.0, 4.0, 20.0, 20.0],
            "volume": [2_000_000, 5_000_000, 1_500_000, 1_500_000],
            "volume_adj": [2_000_000, 5_000_000, 1_500_000, 1_500_000],
            "volume_for_liquidity": [2_000_000, 5_000_000, 1_500_000, 1_500_000],
            "realized_vol20": [0.02, 0.03, 0.02, 0.02],
        }
    )
    equity_prices.to_parquet(clean_root / "equity_prices_daily.parquet", index=False)

    etf_prices = pd.DataFrame(
        {
            "permaticker": [10],
            "ticker": ["SPY"],
            "date": [pd.Timestamp("2024-01-10")],
            "close_for_returns": [100.0],
            "adv20_dollar": [1_000_000.0],
            "dollar_volume": [1_000_000.0],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "close_adj": [100.0],
            "volume": [10_000],
            "volume_adj": [10_000],
            "volume_for_liquidity": [10_000],
            "realized_vol20": [0.01],
        }
    )
    etf_prices.to_parquet(clean_root / "etf_prices_daily.parquet", index=False)

    config = load_config(cfg_path)
    universe = build_tradable_universe(config)

    aaa = universe[(universe["permaticker"] == 1) & (universe["asof_date"] == pd.Timestamp("2024-01-10"))].iloc[0]
    assert bool(aaa["eligible_flag"])
    assert aaa["eligibility_reason"] == "eligible"

    bbb = universe[(universe["permaticker"] == 2) & (universe["asof_date"] == pd.Timestamp("2024-01-10"))].iloc[0]
    assert not bool(bbb["eligible_flag"])
    assert "price" in bbb["eligibility_reason"]

    spy = universe[(universe["ticker"] == "SPY") & (universe["asof_date"] == pd.Timestamp("2024-01-10"))].iloc[0]
    assert bool(spy["eligible_flag"])
    assert spy["eligibility_reason"] == "etf_allowlist"

    ccc_2023 = universe[(universe["permaticker"] == 3) & (universe["asof_date"] == pd.Timestamp("2023-06-01"))].iloc[0]
    assert bool(ccc_2023["eligible_flag"])
    assert bool(ccc_2023["screen_security_type_pass"])

    ccc_2024 = universe[(universe["permaticker"] == 3) & (universe["asof_date"] == pd.Timestamp("2024-06-03"))].iloc[0]
    assert not bool(ccc_2024["eligible_flag"])
    assert not bool(ccc_2024["screen_security_type_pass"])


def test_universe_market_cap_join_aligns_on_keys_not_row_order(tmp_path: Path):
    cfg_path = tmp_path / "stage1.yaml"
    _write_config(cfg_path)

    clean_root = tmp_path / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)

    master_security = pd.DataFrame(
        {
            "permaticker": [1, 2],
            "ticker": ["AAA", "BBB"],
            "start_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "end_date": [pd.Timestamp("2025-12-31"), pd.Timestamp("2025-12-31")],
            "security_type_raw": ["Common Stock", "Common Stock"],
            "exchange": ["NYSE", "NASDAQ"],
            "country": ["USA", "USA"],
            "is_otc": [False, False],
            "is_adr": [False, False],
            "is_etf": [False, False],
            "is_common_stock": [True, True],
        }
    )
    master_security.to_parquet(clean_root / "master_security.parquet", index=False)

    # Intentionally reverse row order vs permaticker order to catch index-based mis-assignment.
    equity_prices = pd.DataFrame(
        {
            "permaticker": [2, 1],
            "ticker": ["BBB", "AAA"],
            "date": [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-10")],
            "close_for_returns": [10.0, 10.0],
            "adv20_dollar": [20_000_000.0, 20_000_000.0],
            "dollar_volume": [20_000_000.0, 20_000_000.0],
            "open": [10.0, 10.0],
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "close_adj": [10.0, 10.0],
            "volume": [2_000_000, 2_000_000],
            "volume_adj": [2_000_000, 2_000_000],
            "volume_for_liquidity": [2_000_000, 2_000_000],
            "realized_vol20": [0.02, 0.02],
        }
    )
    equity_prices.to_parquet(clean_root / "equity_prices_daily.parquet", index=False)

    etf_prices = pd.DataFrame(
        {
            "permaticker": [],
            "ticker": [],
            "date": [],
            "close_for_returns": [],
            "adv20_dollar": [],
            "dollar_volume": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "close_adj": [],
            "volume": [],
            "volume_adj": [],
            "volume_for_liquidity": [],
            "realized_vol20": [],
        }
    )
    etf_prices.to_parquet(clean_root / "etf_prices_daily.parquet", index=False)

    fundamentals = pd.DataFrame(
        {
            "permaticker": [1, 2],
            "knowledge_date": [pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-09")],
            "lastupdated": [pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-09")],
            "market_cap": [100.0, 200.0],
        }
    )
    fundamentals.to_parquet(clean_root / "fundamentals_pit.parquet", index=False)

    config = load_config(cfg_path)
    universe = build_tradable_universe(config)

    a = universe[(universe["permaticker"] == 1) & (universe["asof_date"] == pd.Timestamp("2024-01-10"))].iloc[0]
    b = universe[(universe["permaticker"] == 2) & (universe["asof_date"] == pd.Timestamp("2024-01-10"))].iloc[0]
    assert float(a["market_cap"]) == 100.0
    assert float(b["market_cap"]) == 200.0
