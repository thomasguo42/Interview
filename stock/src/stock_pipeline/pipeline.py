from __future__ import annotations

from datetime import UTC, datetime
from datetime import timedelta

import pandas as pd

from .calendar import TradingCalendar, build_nyse_calendar
from .config import Stage1Config
from .ingest_actions import ingest_actions_table
from .ingest_facts import ingest_fact_tables
from .ingest_master import ingest_master_tables
from .ingest_prices import ingest_price_tables
from .io_utils import ensure_dir, read_json, write_json, write_parquet
from .manifest import write_clean_manifest
from .nasdaq import NasdaqDataLinkClient
from .panel import build_daily_panel
from .schema_lock import DatasetSchemaLock, lock_dataset_schemas
from .universe import build_tradable_universe
from .validation import run_stage1_validation


def ensure_storage_layout(config: Stage1Config) -> None:
    ensure_dir(config.storage.raw_root)
    ensure_dir(config.storage.clean_root)
    ensure_dir(config.storage.manifest_root)


def make_client(config: Stage1Config) -> NasdaqDataLinkClient:
    return NasdaqDataLinkClient(
        base_url=config.nasdaq.base_url,
        api_key_env=config.nasdaq.api_key_env,
        page_size=config.nasdaq.page_size,
        timeout=config.nasdaq.request_timeout_seconds,
        max_retries=config.nasdaq.max_retries,
        retry_backoff_seconds=config.nasdaq.retry_backoff_seconds,
        prefer_local_zip=config.nasdaq.prefer_local_zip,
    )


def build_calendar_artifact(config: Stage1Config) -> TradingCalendar:
    end_dt = pd.Timestamp(config.project.end_date) + timedelta(days=10)
    calendar_df = build_nyse_calendar(config.project.start_date, end_dt.strftime("%Y-%m-%d"))

    out_path = config.storage.clean_root / "trading_calendar_nyse.parquet"
    write_parquet(calendar_df, out_path)
    write_clean_manifest(
        config.storage.manifest_root,
        alias="trading_calendar_nyse",
        clean_path=out_path,
        row_count=len(calendar_df),
        columns=list(calendar_df.columns),
    )
    return TradingCalendar.from_dataframe(calendar_df, date_col="trade_date")


def lock_schemas(config: Stage1Config, client: NasdaqDataLinkClient) -> dict[str, DatasetSchemaLock]:
    return lock_dataset_schemas(config, client, config.storage.manifest_root)


def load_schema_locks(config: Stage1Config) -> dict[str, DatasetSchemaLock]:
    out: dict[str, DatasetSchemaLock] = {}
    schema_dir = config.storage.manifest_root / "schema"
    for alias in config.nasdaq.datasets:
        path = schema_dir / f"{alias}.json"
        if not path.exists():
            raise FileNotFoundError(f"Schema lock missing for {alias}: {path}")
        payload = read_json(path)
        out[alias] = DatasetSchemaLock(
            alias=payload["alias"],
            datatable=payload["datatable"],
            columns=payload.get("columns", []),
            resolved_fields=payload.get("resolved_fields", {}),
            required_columns=payload.get("required_columns", []),
        )
    return out


def run_full_stage1(config: Stage1Config) -> dict[str, object]:
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

    def stage_log(message: str) -> None:
        print(f"[stage1][{run_id}] {message}", flush=True)

    def update_run_status(stage: str, status: str, extra: dict[str, object] | None = None) -> None:
        payload: dict[str, object] = {
            "run_id": run_id,
            "updated_at_utc": datetime.now(tz=UTC).isoformat(),
            "stage": stage,
            "status": status,
        }
        if extra:
            payload.update(extra)
        write_json(config.storage.manifest_root / "runs" / "stage1_last_run.json", payload)

    ensure_storage_layout(config)
    update_run_status("bootstrap", "in_progress")
    client = make_client(config)

    try:
        stage_log("locking schemas from Nasdaq metadata")
        update_run_status("schema_lock", "in_progress")
        schema_locks = lock_schemas(config, client)
        update_run_status("schema_lock", "completed", {"datasets": sorted(schema_locks.keys())})

        stage_log("building NYSE trading calendar")
        update_run_status("calendar", "in_progress")
        calendar = build_calendar_artifact(config)
        update_run_status("calendar", "completed", {"trading_days": int(len(calendar.dates))})

        stage_log("ingesting master reference tables")
        update_run_status("ingest_master", "in_progress")
        ingest_master_tables(config, client, schema_locks["master"])
        update_run_status("ingest_master", "completed")

        if "actions" in schema_locks:
            stage_log("ingesting corporate actions table")
            update_run_status("ingest_actions", "in_progress")
            ingest_actions_table(config, client, schema_locks["actions"])
            update_run_status("ingest_actions", "completed")

        stage_log("ingesting equity and ETF prices")
        update_run_status("ingest_prices", "in_progress")
        ingest_price_tables(config, client, schema_locks["equity_prices"], schema_locks["etf_prices"])
        update_run_status("ingest_prices", "completed")

        stage_log("ingesting fundamentals, insiders, and institutions")
        update_run_status("ingest_facts", "in_progress")
        ingest_fact_tables(
            config,
            client,
            calendar,
            schema_locks["fundamentals"],
            schema_locks["insiders"],
            schema_locks["institutions"],
        )
        update_run_status("ingest_facts", "completed")

        stage_log("building tradable universe")
        update_run_status("build_universe", "in_progress")
        build_tradable_universe(config)
        update_run_status("build_universe", "completed")

        stage_log("building daily as-of panel")
        update_run_status("build_panel", "in_progress")
        build_daily_panel(config)
        update_run_status("build_panel", "completed")

        stage_log("running validation suite")
        update_run_status("validate", "in_progress")
        report = run_stage1_validation(config)
        update_run_status("validate", "completed", {"validation_passed": bool(report.get("passed"))})

        stage_log(f"run finished with passed={report.get('passed')}")
        return report
    except Exception as exc:
        update_run_status("failed", "error", {"error": str(exc)})
        raise
