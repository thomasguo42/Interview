from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from stock_pipeline.stage2_backtest import (
    _compute_insider_scores_at_decisions,
    _compute_period_split,
    _target_weights_for_strategy,
    run_benchmark_backtest,
    run_strategy_backtest,
)
from stock_pipeline.stage2_cli import build_parser
from stock_pipeline.stage2_config import load_stage2_config


def _write_stage2_fixture(tmp_path: Path) -> Path:
    clean_root = tmp_path / "clean"
    out_root = tmp_path / "backtests"
    clean_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    trading_days = pd.to_datetime(
        [
            "2023-12-29",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-12-30",
            "2024-12-31",
            "2025-01-02",
        ]
    )
    pd.DataFrame({"trade_date": trading_days}).to_parquet(clean_root / "trading_calendar_nyse.parquet", index=False)

    eq_rows = []
    spy_rows = []
    bil_rows = []
    for i, date in enumerate(trading_days):
        px = 10.0 + i * 0.1
        eq_rows.append(
            {
                "permaticker": 1,
                "ticker": "AAA",
                "date": date,
                "open": px,
                "close": px,
                "close_adj": px,
                "adv20_dollar": 2_000_000.0,
                "realized_vol20": 0.02,
            }
        )
        spy_px = 100.0 + i
        bil_px = 50.0 + i * 0.05
        spy_rows.append(
            {
                "permaticker": 100,
                "ticker": "SPY",
                "date": date,
                "open": spy_px,
                "close": spy_px,
                "close_adj": spy_px,
                "adv20_dollar": 100_000_000.0,
                "realized_vol20": 0.02,
            }
        )
        bil_rows.append(
            {
                "permaticker": 101,
                "ticker": "BIL",
                "date": date,
                "open": bil_px,
                "close": bil_px,
                "close_adj": bil_px,
                "adv20_dollar": 20_000_000.0,
                "realized_vol20": 0.01,
            }
        )

    pd.DataFrame(eq_rows).to_parquet(clean_root / "equity_prices_daily.parquet", index=False)
    pd.DataFrame(spy_rows + bil_rows).to_parquet(clean_root / "etf_prices_daily.parquet", index=False)

    # Universe only needs eligible equity rows on decision dates.
    universe_dates = pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08", "2024-12-31"])
    universe = pd.DataFrame(
        {
            "asof_date": universe_dates,
            "permaticker": [1, 1, 1, 1],
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "eligible_flag": [True, True, True, True],
            "is_etf_allowlist": [False, False, False, False],
            "adv20_dollar": [2_000_000.0] * 4,
        }
    )
    universe.to_parquet(clean_root / "tradable_universe_daily.parquet", index=False)

    insiders = pd.DataFrame(
        {
            "permaticker": [1, 1],
            "knowledge_date": pd.to_datetime(["2023-12-28", "2024-01-04"]),
            "insider_net_dollar": [1_000.0, 2_000.0],
        }
    )
    insiders.to_parquet(clean_root / "insiders_daily_agg.parquet", index=False)

    institutions = pd.DataFrame(
        {
            "permaticker": [1, 1],
            "knowledge_date": pd.to_datetime(["2023-12-28", "2024-01-04"]),
            "institution_total_shares": [100.0, 120.0],
        }
    )
    institutions.to_parquet(clean_root / "institutions_daily_agg.parquet", index=False)

    master = pd.DataFrame(
        {
            "permaticker": [1, 100, 101],
            "end_date": pd.to_datetime(["2030-01-01", "2030-01-01", "2030-01-01"]),
        }
    )
    master.to_parquet(clean_root / "master_security.parquet", index=False)

    cfg = {
        "stage1_config_path": "config/stage1.yaml",
        "period": {
            "start_date": "2023-12-29",
            "end_date": "2025-01-02",
            "holdout_years": 1,
            "include_forward_period": True,
        },
        "portfolio": {
            "initial_capital": 10_000.0,
            "target_positions": 1,
            "max_positions": 1,
            "max_weight_per_name": 1.0,
            "turnover_max_name_changes": 15,
            "turnover_initial_rebalance_exempt": True,
            "min_trade_notional": 0.0,
            "integer_shares": True,
        },
        "execution": {
            "liquidity_adv_fraction": 0.01,
            "stale_price_max_days": 5,
            "stale_liquidation_penalty_bps": 50.0,
            "calendar_primary": str(clean_root / "trading_calendar_nyse.parquet"),
            "calendar_fallback": str(clean_root / "trading_calendar.parquet"),
        },
        "risk_overlay": {
            "enabled_for_combo_overlay": True,
            "spy_floor_weight": 0.40,
            "defensive_ticker": "BIL",
        },
        "benchmarks": {
            "default_cost_scenario": "base",
            "symbols": ["SPY", "BIL"],
        },
        "storage": {
            "clean_root": str(clean_root),
            "output_root": str(out_root),
            "equity_prices": "equity_prices_daily.parquet",
            "etf_prices": "etf_prices_daily.parquet",
            "universe": "tradable_universe_daily.parquet",
            "panel": "daily_panel.parquet",
            "insiders_agg": "insiders_daily_agg.parquet",
            "institutions_agg": "institutions_daily_agg.parquet",
            "master_security": "master_security.parquet",
            "equity_prices_adj": "equity_prices_daily_adj.parquet",
            "etf_prices_adj": "etf_prices_daily_adj.parquet",
        },
        "strategies": {
            "combo_alias_to_raw": True,
            "insider_window_calendar_days": 90,
            "combo_weights": {
                "momentum": 0.5,
                "insiders": 0.25,
                "institutions": 0.25,
            },
            "combo_winsor_lower_q": 0.01,
            "combo_winsor_upper_q": 0.99,
        },
        "validation": {
            "run_delay_sanity_check": True,
            "delay_sanity_days": 1,
            "rebalance_audit_sample_size": 50,
            "rebalance_audit_seed": 17,
            "autorun_missing_benchmarks": True,
        },
    }
    cfg_path = tmp_path / "stage2.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_period_split_uses_last_full_calendar_years():
    dates = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-12-29",
                "2024-01-03",
                "2024-12-31",
                "2025-01-02",
            ]
        )
    )
    split = _compute_period_split(dates, holdout_years=1)
    assert split.holdout_start == pd.Timestamp("2024-01-01")
    assert split.holdout_end == pd.Timestamp("2024-12-31")
    assert split.forward_start == pd.Timestamp("2025-01-02")


def test_stage2_cli_parser_wired_commands():
    parser = build_parser()
    args = parser.parse_args(["prepare-prices"])
    assert args.func.__name__ == "cmd_prepare_prices"
    args = parser.parse_args(["benchmark", "--benchmark", "SPY"])
    assert args.func.__name__ == "cmd_benchmark"
    args = parser.parse_args(["backtest", "--strategy", "MOM_126x21", "--costs", "base"])
    assert args.func.__name__ == "cmd_backtest"


def test_stage2_smoke_backtests_write_artifacts(tmp_path: Path):
    cfg_path = _write_stage2_fixture(tmp_path)
    cfg = load_stage2_config(cfg_path)

    strategy_result = run_strategy_backtest(cfg, strategy="COMBO", cost_scenario="base")
    assert strategy_result["summary"]["strategy"] == "COMBO_RAW"
    assert Path(strategy_result["artifacts"]["nav_daily"]).exists()
    assert Path(strategy_result["artifacts"]["summary"]).exists()
    assert "benchmarks" in strategy_result["summary"]["comparisons"]
    assert "execution_delay" in strategy_result["summary"]["sensitivity_checks"]
    assert strategy_result["summary"]["sensitivity_checks"]["execution_delay"]["warning_only"] is True
    assert "rebalance_spot_audit" in strategy_result["summary"]["audits"]

    benchmark_result = run_benchmark_backtest(cfg, benchmark_ticker="SPY", cost_scenario="base")
    assert benchmark_result["summary"]["strategy"] == "BENCHMARK_SPY"
    assert Path(benchmark_result["artifacts"]["report"]).exists()

    # Adjusted-price tables are generated automatically when absent.
    assert (cfg.storage.clean_root / cfg.storage.equity_prices_adj).exists()
    assert (cfg.storage.clean_root / cfg.storage.etf_prices_adj).exists()


def test_benchmark_enters_even_when_adv_missing(tmp_path: Path):
    cfg_path = _write_stage2_fixture(tmp_path)
    cfg = load_stage2_config(cfg_path)

    # Force missing ADV on ETF benchmark source.
    etf_path = cfg.storage.clean_root / cfg.storage.etf_prices
    etf = pd.read_parquet(etf_path)
    etf["adv20_dollar"] = pd.NA
    etf.to_parquet(etf_path, index=False)

    result = run_benchmark_backtest(cfg, benchmark_ticker="SPY", cost_scenario="base")
    overall = result["summary"]["metrics"]["overall"]
    assert overall["end_nav"] != overall["start_nav"]


def test_insider_window_rolls_off_by_decision_date():
    insiders = pd.DataFrame(
        {
            "permaticker": [1],
            "knowledge_date": pd.to_datetime(["2023-01-01"]),
            "insider_net_dollar": [500.0],
        }
    )
    decision_index = pd.DataFrame(
        {
            "permaticker": [1, 1],
            "asof_date": pd.to_datetime(["2023-01-15", "2023-05-01"]),
        }
    )
    scores = _compute_insider_scores_at_decisions(decision_index, insiders, window_days=90).sort_values("asof_date")
    assert float(scores.iloc[0]["ins_90"]) == 500.0
    assert float(scores.iloc[1]["ins_90"]) == 0.0


def test_strategy_etf_sleeve_not_blocked_when_adv_missing(tmp_path: Path):
    cfg_path = _write_stage2_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text())
    raw["portfolio"]["max_weight_per_name"] = 0.50
    raw["validation"]["run_delay_sanity_check"] = False
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_stage2_config(cfg_path)

    etf_path = cfg.storage.clean_root / cfg.storage.etf_prices
    etf = pd.read_parquet(etf_path)
    etf["adv20_dollar"] = pd.NA
    etf.to_parquet(etf_path, index=False)

    result = run_strategy_backtest(cfg, strategy="MOM_126x21", cost_scenario="base")
    trades = pd.read_parquet(result["artifacts"]["trades"])
    etf_trades = trades[trades["asset_class"] == "etf"].copy()

    assert not etf_trades.empty
    assert not bool((etf_trades["reason"] == "missing_decision_adv20").any())
    assert bool((etf_trades["reason"] == "executed").any())


def test_combo_overlay_risk_off_respects_name_change_cap(tmp_path: Path):
    cfg_path = _write_stage2_fixture(tmp_path)
    cfg = load_stage2_config(cfg_path)

    day_scores = pd.DataFrame(
        {
            "permaticker": list(range(1, 101)),
            "ticker": [f"T{i}" for i in range(1, 101)],
            "score": list(range(100, 0, -1)),
        }
    )
    current_equities = set(range(1, 36))
    weights, selected, name_changes = _target_weights_for_strategy(
        strategy_id="COMBO_OVERLAY",
        day_scores=day_scores,
        current_equities=current_equities,
        is_initial_rebalance=False,
        risk_off=True,
        spy_permaticker=100,
        defensive_permaticker=101,
        config=cfg,
    )

    assert name_changes <= cfg.portfolio.turnover_max_name_changes
    assert len(selected) == len(current_equities) - cfg.portfolio.turnover_max_name_changes
    assert 100 in weights
    assert 101 in weights


def test_benchmark_metrics_flags_and_weekly_turnover(tmp_path: Path):
    cfg_path = _write_stage2_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text())
    raw["validation"]["run_delay_sanity_check"] = False
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_stage2_config(cfg_path)

    result = run_benchmark_backtest(cfg, benchmark_ticker="BIL", cost_scenario="base")
    summary = result["summary"]
    overall = summary["metrics"]["overall"]

    assert summary["sanity_flags"]["extreme_sharpe_over_3"] is False
    assert overall["avg_weekly_turnover"] > 0.0
    assert overall["avg_weekly_turnover_on_trade_weeks"] >= overall["avg_weekly_turnover"]
