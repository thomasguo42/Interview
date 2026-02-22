from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import Stage1Config, load_config as load_stage1_config


@dataclass(frozen=True)
class Stage2PeriodConfig:
    start_date: str
    end_date: str
    holdout_years: int
    include_forward_period: bool


@dataclass(frozen=True)
class Stage2PortfolioConfig:
    initial_capital: float
    target_positions: int
    max_positions: int
    max_weight_per_name: float
    turnover_max_name_changes: int
    turnover_initial_rebalance_exempt: bool
    min_trade_notional: float
    integer_shares: bool


@dataclass(frozen=True)
class Stage2ExecutionConfig:
    liquidity_adv_fraction: float
    stale_price_max_days: int
    stale_liquidation_penalty_bps: float
    calendar_primary: Path
    calendar_fallback: Path


@dataclass(frozen=True)
class Stage2RiskOverlayConfig:
    enabled_for_combo_overlay: bool
    spy_floor_weight: float
    defensive_ticker: str


@dataclass(frozen=True)
class Stage2BenchmarksConfig:
    default_cost_scenario: str
    symbols: list[str]


@dataclass(frozen=True)
class Stage2StorageConfig:
    clean_root: Path
    output_root: Path
    equity_prices: str
    etf_prices: str
    universe: str
    panel: str
    insiders_agg: str
    institutions_agg: str
    master_security: str
    equity_prices_adj: str
    etf_prices_adj: str


@dataclass(frozen=True)
class Stage2StrategyConfig:
    combo_alias_to_raw: bool
    insider_window_calendar_days: int
    combo_weights: dict[str, float]
    combo_winsor_lower_q: float
    combo_winsor_upper_q: float


@dataclass(frozen=True)
class Stage2ValidationConfig:
    run_delay_sanity_check: bool
    delay_sanity_days: int
    rebalance_audit_sample_size: int
    rebalance_audit_seed: int
    autorun_missing_benchmarks: bool


@dataclass(frozen=True)
class Stage2Config:
    stage1_config_path: Path
    period: Stage2PeriodConfig
    portfolio: Stage2PortfolioConfig
    execution: Stage2ExecutionConfig
    risk_overlay: Stage2RiskOverlayConfig
    benchmarks: Stage2BenchmarksConfig
    storage: Stage2StorageConfig
    strategies: Stage2StrategyConfig
    validation: Stage2ValidationConfig
    stage1: Stage1Config


def _require(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required mapping section: {key}")
    return value


def load_stage2_config(path: str | Path) -> Stage2Config:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text())

    stage1_cfg_path = Path(raw.get("stage1_config_path", "config/stage1.yaml"))
    stage1_cfg = load_stage1_config(stage1_cfg_path)

    period_raw = _require(raw, "period")
    period = Stage2PeriodConfig(
        start_date=str(period_raw.get("start_date", stage1_cfg.project.start_date)),
        end_date=str(period_raw.get("end_date", stage1_cfg.project.end_date)),
        holdout_years=int(period_raw.get("holdout_years", stage1_cfg.project.holdout_years)),
        include_forward_period=bool(period_raw.get("include_forward_period", True)),
    )

    portfolio_raw = _require(raw, "portfolio")
    portfolio = Stage2PortfolioConfig(
        initial_capital=float(portfolio_raw.get("initial_capital", 100000.0)),
        target_positions=int(portfolio_raw.get("target_positions", 35)),
        max_positions=int(portfolio_raw.get("max_positions", 50)),
        max_weight_per_name=float(portfolio_raw.get("max_weight_per_name", 0.05)),
        turnover_max_name_changes=int(portfolio_raw.get("turnover_max_name_changes", 15)),
        turnover_initial_rebalance_exempt=bool(portfolio_raw.get("turnover_initial_rebalance_exempt", True)),
        min_trade_notional=float(portfolio_raw.get("min_trade_notional", 200.0)),
        integer_shares=bool(portfolio_raw.get("integer_shares", True)),
    )

    execution_raw = _require(raw, "execution")
    execution = Stage2ExecutionConfig(
        liquidity_adv_fraction=float(execution_raw.get("liquidity_adv_fraction", 0.01)),
        stale_price_max_days=int(execution_raw.get("stale_price_max_days", 5)),
        stale_liquidation_penalty_bps=float(execution_raw.get("stale_liquidation_penalty_bps", 50.0)),
        calendar_primary=Path(execution_raw.get("calendar_primary", "data/clean/trading_calendar_nyse.parquet")),
        calendar_fallback=Path(execution_raw.get("calendar_fallback", "data/clean/trading_calendar.parquet")),
    )

    risk_raw = _require(raw, "risk_overlay")
    risk_overlay = Stage2RiskOverlayConfig(
        enabled_for_combo_overlay=bool(risk_raw.get("enabled_for_combo_overlay", True)),
        spy_floor_weight=float(risk_raw.get("spy_floor_weight", 0.40)),
        defensive_ticker=str(risk_raw.get("defensive_ticker", "BIL")).upper(),
    )

    bench_raw = _require(raw, "benchmarks")
    benchmarks = Stage2BenchmarksConfig(
        default_cost_scenario=str(bench_raw.get("default_cost_scenario", "base")).lower(),
        symbols=[str(x).upper() for x in bench_raw.get("symbols", ["SPY", "BIL"])],
    )

    storage_raw = _require(raw, "storage")
    storage = Stage2StorageConfig(
        clean_root=Path(storage_raw.get("clean_root", "data/clean")),
        output_root=Path(storage_raw.get("output_root", "data/backtests")),
        equity_prices=str(storage_raw.get("equity_prices", "equity_prices_daily.parquet")),
        etf_prices=str(storage_raw.get("etf_prices", "etf_prices_daily.parquet")),
        universe=str(storage_raw.get("universe", "tradable_universe_daily.parquet")),
        panel=str(storage_raw.get("panel", "daily_panel.parquet")),
        insiders_agg=str(storage_raw.get("insiders_agg", "insiders_daily_agg.parquet")),
        institutions_agg=str(storage_raw.get("institutions_agg", "institutions_daily_agg.parquet")),
        master_security=str(storage_raw.get("master_security", "master_security.parquet")),
        equity_prices_adj=str(storage_raw.get("equity_prices_adj", "equity_prices_daily_adj.parquet")),
        etf_prices_adj=str(storage_raw.get("etf_prices_adj", "etf_prices_daily_adj.parquet")),
    )

    strat_raw = _require(raw, "strategies")
    combo_weights = strat_raw.get("combo_weights", {"momentum": 0.5, "insiders": 0.25, "institutions": 0.25})
    strategies = Stage2StrategyConfig(
        combo_alias_to_raw=bool(strat_raw.get("combo_alias_to_raw", True)),
        insider_window_calendar_days=int(strat_raw.get("insider_window_calendar_days", 90)),
        combo_weights={
            "momentum": float(combo_weights.get("momentum", 0.5)),
            "insiders": float(combo_weights.get("insiders", 0.25)),
            "institutions": float(combo_weights.get("institutions", 0.25)),
        },
        combo_winsor_lower_q=float(strat_raw.get("combo_winsor_lower_q", 0.01)),
        combo_winsor_upper_q=float(strat_raw.get("combo_winsor_upper_q", 0.99)),
    )

    validation_raw = raw.get("validation", {}) if isinstance(raw.get("validation", {}), dict) else {}
    validation = Stage2ValidationConfig(
        run_delay_sanity_check=bool(validation_raw.get("run_delay_sanity_check", True)),
        delay_sanity_days=int(validation_raw.get("delay_sanity_days", 1)),
        rebalance_audit_sample_size=int(validation_raw.get("rebalance_audit_sample_size", 50)),
        rebalance_audit_seed=int(validation_raw.get("rebalance_audit_seed", 17)),
        autorun_missing_benchmarks=bool(validation_raw.get("autorun_missing_benchmarks", True)),
    )

    return Stage2Config(
        stage1_config_path=stage1_cfg_path,
        period=period,
        portfolio=portfolio,
        execution=execution,
        risk_overlay=risk_overlay,
        benchmarks=benchmarks,
        storage=storage,
        strategies=strategies,
        validation=validation,
        stage1=stage1_cfg,
    )


def resolve_cost_bps(stage2: Stage2Config, scenario: str, *, is_etf: bool) -> float:
    key = scenario.lower()
    if key not in {"base", "conservative"}:
        raise ValueError(f"Unknown cost scenario: {scenario}")
    costs = stage2.stage1.costs.base if key == "base" else stage2.stage1.costs.conservative
    return float(costs.etfs_bps_per_side if is_etf else costs.equities_bps_per_side)
