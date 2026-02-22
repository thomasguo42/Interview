from __future__ import annotations

from pathlib import Path

import pandas as pd

from .calendar import TradingCalendar
from .config import Stage1Config, resolve_filters
from .io_utils import read_parquet, write_parquet
from .manifest import write_clean_manifest
from .nasdaq import NasdaqDataLinkClient, pull_datatable_to_raw
from .schema_lock import DatasetSchemaLock, required_pull_columns
from .transform_utils import canonical_permaticker, ensure_datetime, ensure_numeric, normalize_ticker


class FactIngestionError(ValueError):
    """Raised when a fact table cannot be converted into point-in-time format."""


def _availability_column(schema_lock: DatasetSchemaLock) -> str:
    availability_col = schema_lock.resolved_fields.get("availability_date")
    if not availability_col:
        raise FactIngestionError(f"Dataset {schema_lock.alias} is missing an availability_date mapping")
    return availability_col


def _resolve_permaticker_from_history(
    table: pd.DataFrame, ticker_history: pd.DataFrame, date_col: str
) -> pd.Series:
    work = table[["ticker", date_col]].copy()
    work = work.reset_index().rename(columns={"index": "_source_index", date_col: "_lookup_date"})
    merged = work.merge(ticker_history, on="ticker", how="left")
    in_range = (merged["_lookup_date"] >= merged["start_date"]) & (merged["_lookup_date"] <= merged["end_date"])
    matched = merged[in_range].copy()
    if matched.empty:
        return pd.Series(pd.NA, index=table.index, dtype="Int64")

    matched = matched.sort_values(["_source_index", "start_date", "end_date"]).drop_duplicates(
        "_source_index", keep="last"
    )
    out = pd.Series(pd.NA, index=table.index, dtype="Int64")
    out.loc[matched["_source_index"].to_numpy()] = matched["permaticker"].astype("Int64").to_numpy()
    return out


def _prepare_fact_base(
    raw_df: pd.DataFrame,
    schema_lock: DatasetSchemaLock,
    calendar: TradingCalendar,
    availability_lag_days: int = 0,
) -> pd.DataFrame:
    resolved = schema_lock.resolved_fields

    availability_col = _availability_column(schema_lock)
    date_candidates = [
        availability_col,
        resolved.get("lastupdated"),
        resolved.get("report_period"),
        resolved.get("transaction_date"),
    ]
    ensure_datetime(raw_df, [col for col in date_candidates if col])

    numeric_candidates = [resolved.get("shares"), resolved.get("transaction_price"), resolved.get("market_cap")]
    ensure_numeric(raw_df, [col for col in numeric_candidates if col])

    perm_col = resolved.get("permaticker")
    ticker_col = resolved.get("ticker")
    if not perm_col and not ticker_col:
        raise FactIngestionError(
            f"Dataset {schema_lock.alias} must resolve at least one security identifier (permaticker or ticker)"
        )

    base = pd.DataFrame(
        {
            "permaticker": canonical_permaticker(raw_df[perm_col])
            if perm_col
            else pd.Series(pd.NA, index=raw_df.index, dtype="Int64"),
            "ticker": normalize_ticker(raw_df[ticker_col]) if ticker_col else pd.Series(pd.NA, index=raw_df.index, dtype="string"),
            "availability_date": raw_df[availability_col],
        }
    )
    base["availability_date_effective"] = base["availability_date"]
    if availability_lag_days > 0:
        base["availability_date_effective"] = base["availability_date_effective"] + pd.to_timedelta(
            availability_lag_days, unit="D"
        )

    base["knowledge_date"] = calendar.next_trading_day(base["availability_date_effective"])

    if resolved.get("lastupdated"):
        base["lastupdated"] = raw_df[resolved["lastupdated"]]
    else:
        base["lastupdated"] = pd.NaT

    if resolved.get("report_period"):
        base["report_period"] = raw_df[resolved["report_period"]]

    if resolved.get("market_cap"):
        base["market_cap"] = raw_df[resolved["market_cap"]]

    if resolved.get("transaction_date"):
        base["transaction_date"] = raw_df[resolved["transaction_date"]]
    if resolved.get("shares"):
        base["shares"] = raw_df[resolved["shares"]]
    if resolved.get("transaction_price"):
        base["transaction_price"] = raw_df[resolved["transaction_price"]]
    if resolved.get("transaction_code"):
        base["transaction_code"] = raw_df[resolved["transaction_code"]].astype("string")
    if resolved.get("insider_role"):
        base["insider_role"] = raw_df[resolved["insider_role"]].astype("string")
    if resolved.get("holder_id"):
        base["holder_id"] = raw_df[resolved["holder_id"]].astype("string")

    return base


def _dedupe_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["permaticker"]
    if "report_period" in df.columns:
        key_cols.append("report_period")
    key_cols.extend(["availability_date", "knowledge_date"])

    sort_cols = ["permaticker", "availability_date", "knowledge_date"]
    if "lastupdated" in df.columns:
        sort_cols.append("lastupdated")

    return (
        df.sort_values(sort_cols)
        .drop_duplicates(key_cols, keep="last")
        .reset_index(drop=True)
    )


def _dedupe_generic(df: pd.DataFrame, extra_keys: list[str] | None = None) -> pd.DataFrame:
    key_cols = ["permaticker", "availability_date", "knowledge_date"]
    if extra_keys:
        key_cols.extend([col for col in extra_keys if col in df.columns])
    sort_cols = key_cols + (["lastupdated"] if "lastupdated" in df.columns else [])
    return df.sort_values(sort_cols).drop_duplicates(key_cols, keep="last").reset_index(drop=True)


def _aggregate_insiders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["permaticker", "knowledge_date", "insider_txn_count", "insider_net_shares", "insider_net_dollar"])

    out = df.copy()
    code = out.get("transaction_code", pd.Series(pd.NA, index=out.index, dtype="string")).astype("string").str.upper()
    sign = pd.Series(0.0, index=out.index)
    sign = sign.mask(code.str.startswith(("P", "B"), na=False), 1.0)
    sign = sign.mask(code.str.startswith("S", na=False), -1.0)

    shares = pd.to_numeric(out.get("shares", 0.0), errors="coerce").fillna(0.0)
    txn_price = pd.to_numeric(out.get("transaction_price", 0.0), errors="coerce").fillna(0.0)

    out["signed_shares"] = sign * shares
    out["signed_dollar"] = sign * shares * txn_price

    agg = (
        out.groupby(["permaticker", "knowledge_date"], as_index=False)
        .agg(
            insider_txn_count=("permaticker", "size"),
            insider_net_shares=("signed_shares", "sum"),
            insider_net_dollar=("signed_dollar", "sum"),
        )
        .sort_values(["permaticker", "knowledge_date"])
        .reset_index(drop=True)
    )
    return agg


def _aggregate_institutions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "permaticker",
                "knowledge_date",
                "institution_record_count",
                "institution_total_shares",
                "institution_filing_count",
            ]
        )

    shares = pd.to_numeric(df.get("shares", 0.0), errors="coerce").fillna(0.0)
    work = df.copy()
    work["shares"] = shares

    agg = (
        work.groupby(["permaticker", "knowledge_date"], as_index=False)
        .agg(
            institution_record_count=("permaticker", "size"),
            institution_total_shares=("shares", "sum"),
        )
        .sort_values(["permaticker", "knowledge_date"])
        .reset_index(drop=True)
    )
    # Backward-compatible alias; this is a holder-record count, not literal filing count.
    agg["institution_filing_count"] = agg["institution_record_count"]
    return agg


def ingest_fact_tables(
    config: Stage1Config,
    client: NasdaqDataLinkClient,
    calendar: TradingCalendar,
    fundamentals_schema: DatasetSchemaLock,
    insiders_schema: DatasetSchemaLock,
    institutions_schema: DatasetSchemaLock,
) -> dict[str, Path]:
    ticker_history = read_parquet(config.storage.clean_root / "ticker_permaticker_history.parquet")
    ticker_history["ticker"] = normalize_ticker(ticker_history["ticker"])
    ticker_history["start_date"] = pd.to_datetime(ticker_history["start_date"]).dt.normalize()
    ticker_history["end_date"] = pd.to_datetime(ticker_history["end_date"]).dt.normalize()

    outputs: dict[str, Path] = {}

    for alias, schema_lock in (
        ("fundamentals", fundamentals_schema),
        ("insiders", insiders_schema),
        ("institutions", institutions_schema),
    ):
        dataset = config.dataset(alias)
        params = resolve_filters(
            dataset.filters,
            start_date=config.project.start_date,
            end_date=config.project.end_date,
        )
        raw_df, _ = pull_datatable_to_raw(
            client=client,
            alias=alias,
            datatable=dataset.datatable,
            params=params,
            raw_root=config.storage.raw_root,
            manifest_root=config.storage.manifest_root,
            select_columns=required_pull_columns(schema_lock),
            local_zip_path=dataset.local_zip_path,
        )

        base = _prepare_fact_base(
            raw_df,
            schema_lock,
            calendar,
            availability_lag_days=dataset.availability_lag_days,
        )

        missing_perm = base["permaticker"].isna() & base["ticker"].notna() & base["availability_date"].notna()
        if missing_perm.any():
            resolved_perm = _resolve_permaticker_from_history(
                base.loc[missing_perm, ["ticker", "availability_date"]],
                ticker_history,
                date_col="availability_date",
            )
            base.loc[missing_perm, "permaticker"] = resolved_perm.values

        base = base.dropna(subset=["permaticker", "availability_date", "knowledge_date"]).copy()
        base["permaticker"] = base["permaticker"].astype("Int64")

        if alias == "fundamentals":
            clean_df = _dedupe_fundamentals(base)
        elif alias == "insiders":
            clean_df = _dedupe_generic(base, extra_keys=["transaction_date", "transaction_code", "shares", "transaction_price"])
        else:
            clean_df = _dedupe_generic(base, extra_keys=["report_period", "holder_id", "shares"])

        out_path = config.storage.clean_root / f"{alias}_pit.parquet"
        write_parquet(clean_df, out_path)
        write_clean_manifest(
            config.storage.manifest_root,
            alias=alias,
            clean_path=out_path,
            row_count=len(clean_df),
            columns=list(clean_df.columns),
            notes="knowledge_date computed as next_trading_day(availability_date)",
        )
        outputs[alias] = out_path

        if alias == "insiders":
            insiders_agg = _aggregate_insiders(clean_df)
            insiders_agg_path = config.storage.clean_root / "insiders_daily_agg.parquet"
            write_parquet(insiders_agg, insiders_agg_path)
            write_clean_manifest(
                config.storage.manifest_root,
                alias="insiders_daily_agg",
                clean_path=insiders_agg_path,
                row_count=len(insiders_agg),
                columns=list(insiders_agg.columns),
            )
            outputs["insiders_daily_agg"] = insiders_agg_path

        if alias == "institutions":
            institutions_agg = _aggregate_institutions(clean_df)
            institutions_agg_path = config.storage.clean_root / "institutions_daily_agg.parquet"
            write_parquet(institutions_agg, institutions_agg_path)
            write_clean_manifest(
                config.storage.manifest_root,
                alias="institutions_daily_agg",
                clean_path=institutions_agg_path,
                row_count=len(institutions_agg),
                columns=list(institutions_agg.columns),
            )
            outputs["institutions_daily_agg"] = institutions_agg_path

    return outputs
