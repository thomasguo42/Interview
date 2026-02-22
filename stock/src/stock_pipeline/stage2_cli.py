from __future__ import annotations

import argparse
import json

from .stage2_backtest import (
    prepare_adjusted_price_tables,
    run_benchmark_backtest,
    run_strategy_backtest,
)
from .stage2_config import load_stage2_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2 backtest pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/stage2.yaml", help="Path to stage-2 YAML config")

    prep = sub.add_parser("prepare-prices", parents=[common], help="Build adjusted-open price tables for stage 2")
    prep.set_defaults(func=cmd_prepare_prices)

    benchmark = sub.add_parser("benchmark", parents=[common], help="Run benchmark backtest (SPY/BIL)")
    benchmark.add_argument("--benchmark", required=True, help="Benchmark symbol, e.g. SPY or BIL")
    benchmark.add_argument("--costs", default=None, help="Cost scenario override: base|conservative")
    benchmark.set_defaults(func=cmd_benchmark)

    backtest = sub.add_parser("backtest", parents=[common], help="Run baseline strategy backtest")
    backtest.add_argument(
        "--strategy",
        required=True,
        help="Strategy ID: MOM_126x21|MOM_252x21|REV_21|INS_90|INST_CHG|COMBO_RAW|COMBO_OVERLAY|COMBO",
    )
    backtest.add_argument("--costs", required=True, help="Cost scenario: base|conservative")
    backtest.set_defaults(func=cmd_backtest)

    return parser


def cmd_prepare_prices(args: argparse.Namespace) -> None:
    config = load_stage2_config(args.config)
    outputs = prepare_adjusted_price_tables(config)
    print(json.dumps(outputs, indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    config = load_stage2_config(args.config)
    costs = str(args.costs or config.benchmarks.default_cost_scenario).lower()
    result = run_benchmark_backtest(config, benchmark_ticker=args.benchmark, cost_scenario=costs)
    payload = {
        "summary_path": result["artifacts"]["summary"],
        "output_dir": result["artifacts"]["output_dir"],
        "strategy": result["summary"]["strategy"],
        "cost_scenario": result["summary"]["cost_scenario"],
        "date_range": result["summary"]["date_range"],
        "metrics_overall": result["summary"]["metrics"]["overall"],
    }
    print(json.dumps(payload, indent=2, default=str))


def cmd_backtest(args: argparse.Namespace) -> None:
    config = load_stage2_config(args.config)
    result = run_strategy_backtest(config, strategy=args.strategy, cost_scenario=args.costs)
    payload = {
        "summary_path": result["artifacts"]["summary"],
        "output_dir": result["artifacts"]["output_dir"],
        "strategy": result["summary"]["strategy"],
        "cost_scenario": result["summary"]["cost_scenario"],
        "date_range": result["summary"]["date_range"],
        "metrics_overall": result["summary"]["metrics"]["overall"],
    }
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
