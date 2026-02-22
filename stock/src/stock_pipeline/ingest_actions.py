from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Stage1Config, resolve_filters
from .io_utils import write_parquet
from .manifest import write_clean_manifest
from .nasdaq import NasdaqDataLinkClient, pull_datatable_to_raw
from .schema_lock import DatasetSchemaLock, required_pull_columns
from .transform_utils import ensure_datetime, normalize_ticker


def ingest_actions_table(
    config: Stage1Config,
    client: NasdaqDataLinkClient,
    schema_lock: DatasetSchemaLock,
) -> Path:
    dataset = config.dataset("actions")
    params = resolve_filters(
        dataset.filters,
        start_date=config.project.start_date,
        end_date=config.project.end_date,
    )

    raw_df, _ = pull_datatable_to_raw(
        client=client,
        alias="actions",
        datatable=dataset.datatable,
        params=params,
        raw_root=config.storage.raw_root,
        manifest_root=config.storage.manifest_root,
        select_columns=required_pull_columns(schema_lock),
        local_zip_path=dataset.local_zip_path,
    )

    resolved = schema_lock.resolved_fields
    date_col = resolved.get("date")
    ticker_col = resolved.get("ticker")
    if not date_col or not ticker_col:
        raise ValueError("Actions dataset must resolve date and ticker columns")

    ensure_datetime(raw_df, [date_col])

    out = pd.DataFrame(
        {
            "date": raw_df[date_col],
            "ticker": normalize_ticker(raw_df[ticker_col]),
            "action": raw_df[resolved["action"]].astype("string") if resolved.get("action") else pd.NA,
            "value": pd.to_numeric(raw_df[resolved["value"]], errors="coerce") if resolved.get("value") else pd.NA,
            "contraticker": normalize_ticker(raw_df[resolved["contraticker"]]) if resolved.get("contraticker") else pd.NA,
            "contraname": raw_df[resolved["contraname"]].astype("string") if resolved.get("contraname") else pd.NA,
        }
    )

    out = (
        out.dropna(subset=["date", "ticker"])
        .sort_values(["ticker", "date", "action"])
        .drop_duplicates(["ticker", "date", "action"], keep="last")
        .reset_index(drop=True)
    )

    out_path = config.storage.clean_root / "corporate_actions.parquet"
    write_parquet(out, out_path)
    write_clean_manifest(
        config.storage.manifest_root,
        alias="corporate_actions",
        clean_path=out_path,
        row_count=len(out),
        columns=list(out.columns),
    )
    return out_path
