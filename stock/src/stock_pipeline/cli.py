from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from .config import load_config
from .ingest_actions import ingest_actions_table
from .io_utils import read_json, write_json
from .pipeline import (
    build_calendar_artifact,
    ensure_storage_layout,
    load_schema_locks,
    lock_schemas,
    make_client,
    run_full_stage1,
)
from .ingest_master import ingest_master_tables
from .ingest_prices import ingest_price_tables
from .ingest_facts import ingest_fact_tables
from .universe import build_tradable_universe
from .panel import build_daily_panel
from .validation import run_stage1_validation


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 data foundation pipeline")
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/stage1.yaml", help="Path to stage-1 YAML config")


def cmd_discover(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    client = make_client(config)

    if args.list_datatables:
        payload = client.discover_datatables(database_code=args.database_code)
        print(json.dumps(payload, indent=2)[:20000])

    locks = lock_schemas(config, client)
    print(f"Schema lock completed for {len(locks)} datasets -> {config.storage.manifest_root / 'schema'}")


def cmd_build_calendar(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    calendar = build_calendar_artifact(config)
    print(f"Trading calendar rows: {len(calendar.dates)}")


def cmd_ingest_master(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    client = make_client(config)
    locks = load_schema_locks(config)
    outputs = ingest_master_tables(config, client, locks["master"])
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


def cmd_ingest_prices(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    client = make_client(config)
    locks = load_schema_locks(config)
    outputs = ingest_price_tables(config, client, locks["equity_prices"], locks["etf_prices"])
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


def cmd_ingest_actions(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    client = make_client(config)
    locks = load_schema_locks(config)
    if "actions" not in locks:
        raise KeyError("Schema lock does not include actions dataset")
    path = ingest_actions_table(config, client, locks["actions"])
    print(f"Corporate actions path: {path}")


def cmd_ingest_facts(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    client = make_client(config)
    locks = load_schema_locks(config)
    calendar = build_calendar_artifact(config)
    outputs = ingest_fact_tables(
        config,
        client,
        calendar,
        locks["fundamentals"],
        locks["insiders"],
        locks["institutions"],
    )
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))


def cmd_build_universe(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    universe = build_tradable_universe(config)
    print(f"Universe rows: {len(universe)}")


def cmd_build_panel(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    path = build_daily_panel(config)
    print(f"Daily panel path: {path}")


def cmd_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_storage_layout(config)
    report = run_stage1_validation(config)
    run_status_path = config.storage.manifest_root / "runs" / "stage1_last_run.json"
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    if run_status_path.exists():
        existing = read_json(run_status_path)
        run_id = str(existing.get("run_id", run_id))
    write_json(
        run_status_path,
        {
            "run_id": run_id,
            "updated_at_utc": datetime.now(tz=UTC).isoformat(),
            "stage": "validate",
            "status": "completed",
            "validation_passed": bool(report.get("passed")),
        },
    )
    print(json.dumps(report, indent=2, default=str))


def cmd_run_all(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    report = run_full_stage1(config)
    print(json.dumps(report, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = _common_parser()
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="Discover available datatables and lock schemas")
    _add_config_argument(discover)
    discover.add_argument("--list-datatables", action="store_true", help="Print database datatable listing")
    discover.add_argument("--database-code", default="SHARADAR", help="Database code for discovery listing")
    discover.set_defaults(func=cmd_discover)

    build_calendar = sub.add_parser("build-calendar", help="Build NYSE trading calendar artifact")
    _add_config_argument(build_calendar)
    build_calendar.set_defaults(func=cmd_build_calendar)

    ingest_master = sub.add_parser("ingest-master", help="Ingest master security reference tables")
    _add_config_argument(ingest_master)
    ingest_master.set_defaults(func=cmd_ingest_master)

    ingest_prices = sub.add_parser("ingest-prices", help="Ingest equity and ETF daily prices")
    _add_config_argument(ingest_prices)
    ingest_prices.set_defaults(func=cmd_ingest_prices)

    ingest_actions = sub.add_parser("ingest-actions", help="Ingest corporate actions table")
    _add_config_argument(ingest_actions)
    ingest_actions.set_defaults(func=cmd_ingest_actions)

    ingest_facts = sub.add_parser("ingest-facts", help="Ingest fundamentals/insiders/institutions")
    _add_config_argument(ingest_facts)
    ingest_facts.set_defaults(func=cmd_ingest_facts)

    build_universe = sub.add_parser("build-universe", help="Build daily tradable universe")
    _add_config_argument(build_universe)
    build_universe.set_defaults(func=cmd_build_universe)

    build_panel = sub.add_parser("build-panel", help="Build daily as-of panel")
    _add_config_argument(build_panel)
    build_panel.set_defaults(func=cmd_build_panel)

    validate = sub.add_parser("validate", help="Run stage-1 validation checks")
    _add_config_argument(validate)
    validate.set_defaults(func=cmd_validate)

    run_all = sub.add_parser("run-all", help="Run full stage-1 pipeline end-to-end")
    _add_config_argument(run_all)
    run_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
