from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Stage1Config, resolve_filters
from .io_utils import write_parquet
from .manifest import write_clean_manifest
from .nasdaq import NasdaqDataLinkClient, pull_datatable_to_raw
from .schema_lock import DatasetSchemaLock, required_pull_columns
from .transform_utils import canonical_permaticker, ensure_datetime, normalize_ticker


def _as_bool(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "t", "yes", "y"}
    return series.astype("string").str.lower().str.strip().isin(truthy)


def ingest_master_tables(
    config: Stage1Config,
    client: NasdaqDataLinkClient,
    schema_lock: DatasetSchemaLock,
) -> dict[str, Path]:
    dataset = config.dataset("master")
    params = resolve_filters(
        dataset.filters,
        start_date=config.project.start_date,
        end_date=config.project.end_date,
    )

    raw_df, _ = pull_datatable_to_raw(
        client=client,
        alias="master",
        datatable=dataset.datatable,
        params=params,
        raw_root=config.storage.raw_root,
        manifest_root=config.storage.manifest_root,
        select_columns=required_pull_columns(schema_lock),
        local_zip_path=dataset.local_zip_path,
    )

    resolved = schema_lock.resolved_fields
    perm_col = resolved.get("permaticker")
    ticker_col = resolved.get("ticker")
    if not perm_col or not ticker_col:
        raise ValueError("Master dataset must resolve permaticker and ticker")

    date_cols = [resolved.get("start_date"), resolved.get("end_date")]
    date_cols = [col for col in date_cols if col]
    ensure_datetime(raw_df, date_cols)

    master = pd.DataFrame(
        {
            "permaticker": canonical_permaticker(raw_df[perm_col]),
            "ticker": normalize_ticker(raw_df[ticker_col]),
        }
    )

    if resolved.get("start_date"):
        master["start_date"] = raw_df[resolved["start_date"]]
    else:
        master["start_date"] = pd.Timestamp(config.project.start_date)

    if resolved.get("end_date"):
        master["end_date"] = raw_df[resolved["end_date"]]
    else:
        master["end_date"] = pd.Timestamp(config.project.end_date)

    if resolved.get("security_type"):
        master["security_type_raw"] = raw_df[resolved["security_type"]].astype("string")
    else:
        master["security_type_raw"] = pd.Series(pd.NA, index=raw_df.index, dtype="string")

    if resolved.get("exchange"):
        master["exchange"] = raw_df[resolved["exchange"]].astype("string")
    else:
        master["exchange"] = pd.Series(pd.NA, index=raw_df.index, dtype="string")

    if resolved.get("country"):
        master["country"] = raw_df[resolved["country"]].astype("string")
    else:
        master["country"] = pd.Series(pd.NA, index=raw_df.index, dtype="string")

    if resolved.get("is_adr"):
        master["is_adr"] = _as_bool(raw_df[resolved["is_adr"]])
    else:
        security_type = master["security_type_raw"].str.lower()
        master["is_adr"] = security_type.str.contains("adr", na=False)

    if resolved.get("is_otc"):
        master["is_otc"] = _as_bool(raw_df[resolved["is_otc"]])
    else:
        exchange = master["exchange"].str.upper()
        master["is_otc"] = exchange.eq("OTC")

    st = master["security_type_raw"].str.lower()
    master["is_etf"] = st.str.contains("etf", na=False)
    master["is_common_stock"] = st.str.contains("stock", na=False) & ~master["is_etf"]

    master = (
        master.dropna(subset=["permaticker", "ticker"])
        .drop_duplicates(["permaticker", "ticker", "start_date", "end_date"], keep="last")
        .sort_values(["permaticker", "start_date", "end_date", "ticker"])
        .reset_index(drop=True)
    )

    master_path = config.storage.clean_root / "master_security.parquet"
    write_parquet(master, master_path)
    write_clean_manifest(
        config.storage.manifest_root,
        alias="master_security",
        clean_path=master_path,
        row_count=len(master),
        columns=list(master.columns),
    )

    ticker_history = master[["ticker", "permaticker", "start_date", "end_date"]].copy()
    ticker_history = ticker_history.sort_values(["ticker", "start_date", "end_date", "permaticker"]).reset_index(drop=True)

    ticker_history_path = config.storage.clean_root / "ticker_permaticker_history.parquet"
    write_parquet(ticker_history, ticker_history_path)
    write_clean_manifest(
        config.storage.manifest_root,
        alias="ticker_permaticker_history",
        clean_path=ticker_history_path,
        row_count=len(ticker_history),
        columns=list(ticker_history.columns),
    )

    classification = master[
        ["permaticker", "ticker", "security_type_raw", "exchange", "country", "is_otc", "is_adr", "is_etf", "is_common_stock"]
    ].drop_duplicates(["permaticker", "ticker"], keep="last")

    classification_path = config.storage.clean_root / "security_classification.parquet"
    write_parquet(classification, classification_path)
    write_clean_manifest(
        config.storage.manifest_root,
        alias="security_classification",
        clean_path=classification_path,
        row_count=len(classification),
        columns=list(classification.columns),
    )

    return {
        "master_security": master_path,
        "ticker_permaticker_history": ticker_history_path,
        "security_classification": classification_path,
    }
