from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Stage1Config, resolve_filters
from .io_utils import read_parquet, write_parquet
from .manifest import write_clean_manifest
from .nasdaq import NasdaqDataLinkClient, pull_datatable_to_raw
from .schema_lock import DatasetSchemaLock, required_pull_columns
from .transform_utils import canonical_permaticker, ensure_datetime, ensure_numeric, normalize_ticker


def _resolve_permaticker_from_history(price_df: pd.DataFrame, ticker_history: pd.DataFrame) -> pd.Series:
    if "ticker" not in price_df.columns:
        return pd.Series(pd.NA, index=price_df.index, dtype="Int64")

    work = price_df[["ticker", "date"]].copy()
    work = work.reset_index().rename(columns={"index": "_source_index"})
    merged = work.merge(ticker_history, on="ticker", how="left")

    in_range = (merged["date"] >= merged["start_date"]) & (merged["date"] <= merged["end_date"])
    matched = merged[in_range].copy()
    if matched.empty:
        return pd.Series(pd.NA, index=price_df.index, dtype="Int64")

    matched = matched.sort_values(["_source_index", "start_date", "end_date"]).drop_duplicates(
        "_source_index", keep="last"
    )
    out = pd.Series(pd.NA, index=price_df.index, dtype="Int64")
    out.loc[matched["_source_index"].to_numpy()] = matched["permaticker"].astype("Int64").to_numpy()
    return out


def _standardize_prices(
    raw_df: pd.DataFrame,
    schema_lock: DatasetSchemaLock,
    ticker_history: pd.DataFrame,
) -> pd.DataFrame:
    resolved = schema_lock.resolved_fields
    date_col = resolved.get("date")
    if not date_col:
        raise ValueError(f"Price dataset {schema_lock.alias} could not resolve date column")

    ensure_datetime(raw_df, [date_col])
    numeric_cols = [
        resolved.get("open"),
        resolved.get("high"),
        resolved.get("low"),
        resolved.get("close"),
        resolved.get("close_adj"),
        resolved.get("volume"),
        resolved.get("volume_adj"),
    ]
    ensure_numeric(raw_df, [col for col in numeric_cols if col])

    if resolved.get("ticker"):
        ticker = normalize_ticker(raw_df[resolved["ticker"]])
    else:
        ticker = pd.Series(pd.NA, index=raw_df.index, dtype="string")

    perm_col = resolved.get("permaticker")
    if perm_col:
        permaticker = canonical_permaticker(raw_df[perm_col])
    else:
        permaticker = pd.Series(pd.NA, index=raw_df.index, dtype="Int64")

    standard = pd.DataFrame(
        {
            "permaticker": permaticker,
            "ticker": ticker,
            "date": raw_df[date_col],
            "open": raw_df[resolved["open"]] if resolved.get("open") else pd.NA,
            "high": raw_df[resolved["high"]] if resolved.get("high") else pd.NA,
            "low": raw_df[resolved["low"]] if resolved.get("low") else pd.NA,
            "close": raw_df[resolved["close"]] if resolved.get("close") else pd.NA,
            "close_adj": raw_df[resolved["close_adj"]] if resolved.get("close_adj") else pd.NA,
            "volume": raw_df[resolved["volume"]] if resolved.get("volume") else pd.NA,
            "volume_adj": raw_df[resolved["volume_adj"]] if resolved.get("volume_adj") else pd.NA,
        }
    )

    # Resolve missing permaticker values through ticker history ranges.
    missing = standard["permaticker"].isna()
    if missing.any():
        resolved_perm = _resolve_permaticker_from_history(standard.loc[missing, ["ticker", "date"]], ticker_history)
        standard.loc[missing, "permaticker"] = resolved_perm.values

    standard["close_for_returns"] = standard["close_adj"].fillna(standard["close"])
    standard["volume_for_liquidity"] = standard["volume_adj"].fillna(standard["volume"])
    standard["dollar_volume"] = standard["close_for_returns"] * standard["volume_for_liquidity"]

    standard = standard.dropna(subset=["permaticker", "date"]).copy()
    standard["permaticker"] = standard["permaticker"].astype("Int64")

    standard = standard.sort_values(["permaticker", "date", "ticker"]).drop_duplicates(
        ["permaticker", "date"], keep="last"
    )

    standard["adv20_dollar"] = (
        standard.groupby("permaticker", group_keys=False)["dollar_volume"]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    standard["realized_vol20"] = (
        standard.groupby("permaticker", group_keys=False)["close_for_returns"]
        .pct_change()
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    return standard.reset_index(drop=True)


def ingest_price_tables(
    config: Stage1Config,
    client: NasdaqDataLinkClient,
    equity_schema: DatasetSchemaLock,
    etf_schema: DatasetSchemaLock,
) -> dict[str, Path]:
    ticker_history = read_parquet(config.storage.clean_root / "ticker_permaticker_history.parquet")
    ticker_history["ticker"] = normalize_ticker(ticker_history["ticker"])
    ticker_history["start_date"] = pd.to_datetime(ticker_history["start_date"]).dt.normalize()
    ticker_history["end_date"] = pd.to_datetime(ticker_history["end_date"]).dt.normalize()

    outputs: dict[str, Path] = {}

    for alias, schema_lock in (("equity_prices", equity_schema), ("etf_prices", etf_schema)):
        dataset = config.dataset(alias)
        params = resolve_filters(
            dataset.filters,
            start_date=config.project.start_date,
            end_date=config.project.end_date,
        )

        # Restrict ETF pulls to allowlist for stage 1 defensive sleeve requirements.
        if alias == "etf_prices" and dataset.ticker_filter_column:
            params[dataset.ticker_filter_column] = ",".join(config.etf_allowlist)

        raw_df, _ = pull_datatable_to_raw(
            client=client,
            alias=alias,
            datatable=dataset.datatable,
            params=params,
            raw_root=config.storage.raw_root,
            manifest_root=config.storage.manifest_root,
            select_columns=required_pull_columns(
                schema_lock,
                extra_columns=[dataset.ticker_filter_column] if dataset.ticker_filter_column else None,
            ),
            local_zip_path=dataset.local_zip_path,
        )

        clean_df = _standardize_prices(raw_df, schema_lock, ticker_history)

        out_path = config.storage.clean_root / f"{alias}_daily.parquet"
        write_parquet(clean_df, out_path)
        write_clean_manifest(
            config.storage.manifest_root,
            alias=alias,
            clean_path=out_path,
            row_count=len(clean_df),
            columns=list(clean_df.columns),
        )
        outputs[alias] = out_path

    return outputs
