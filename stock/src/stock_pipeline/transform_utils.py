from __future__ import annotations

from typing import Iterable

import pandas as pd


def normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.strip()


def canonical_permaticker(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def ensure_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def dedupe_sorted(df: pd.DataFrame, key_cols: list[str], sort_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    sort_use = [col for col in sort_cols if col in df.columns]
    if sort_use:
        df = df.sort_values(sort_use)
    keep_keys = [col for col in key_cols if col in df.columns]
    if keep_keys:
        df = df.drop_duplicates(keep_keys, keep="last")
    return df
