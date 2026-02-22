from __future__ import annotations

import numpy as np
import pandas as pd


class AsofJoinError(ValueError):
    """Raised when as-of join preconditions are not met."""


def _coerce_key_dtype(left: pd.DataFrame, right: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_key = left[key_col]
    right_key = right[key_col]

    if pd.api.types.is_numeric_dtype(left_key) or pd.api.types.is_numeric_dtype(right_key):
        left[key_col] = pd.to_numeric(left_key, errors="coerce").astype("Int64")
        right[key_col] = pd.to_numeric(right_key, errors="coerce").astype("Int64")
    else:
        left[key_col] = left_key.astype("string")
        right[key_col] = right_key.astype("string")
    return left, right


def deterministic_asof_join(
    panel_index: pd.DataFrame,
    fact_table: pd.DataFrame,
    *,
    key_col: str = "permaticker",
    asof_col: str = "asof_date",
    knowledge_col: str = "knowledge_date",
    lastupdated_col: str | None = "lastupdated",
    enforce_lastupdated_cutoff: bool = True,
    prefix: str,
) -> pd.DataFrame:
    if key_col not in panel_index.columns or asof_col not in panel_index.columns:
        raise AsofJoinError(f"Panel index must contain {key_col} and {asof_col}")
    if key_col not in fact_table.columns or knowledge_col not in fact_table.columns:
        raise AsofJoinError(f"Fact table must contain {key_col} and {knowledge_col}")

    left = panel_index.copy()
    left[asof_col] = pd.to_datetime(left[asof_col]).dt.normalize()

    right = fact_table.copy()
    right[knowledge_col] = pd.to_datetime(right[knowledge_col]).dt.normalize()
    left, right = _coerce_key_dtype(left, right, key_col)

    if enforce_lastupdated_cutoff and lastupdated_col and lastupdated_col in right.columns:
        right[lastupdated_col] = pd.to_datetime(right[lastupdated_col], errors="coerce").dt.normalize()
        right["_effective_date"] = right[[knowledge_col, lastupdated_col]].max(axis=1)
    else:
        right["_effective_date"] = right[knowledge_col]

    left = left.dropna(subset=[key_col, asof_col]).copy()
    right = right.dropna(subset=[key_col, "_effective_date"]).copy()

    right["_stable_row_id"] = np.arange(len(right), dtype="int64")

    tie_cols = ["_effective_date", key_col, knowledge_col]
    if enforce_lastupdated_cutoff and lastupdated_col and lastupdated_col in right.columns:
        tie_cols.append(lastupdated_col)
    tie_cols.append("_stable_row_id")

    left = left.sort_values([asof_col, key_col]).reset_index(drop=True)
    right = right.sort_values(tie_cols).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on=asof_col,
        right_on="_effective_date",
        by=key_col,
        direction="backward",
        suffixes=("", "_fact"),
        allow_exact_matches=True,
    )

    # Prefix only right-side fact columns. Keep left-side panel columns unchanged.
    right_fact_cols = [
        col
        for col in right.columns
        if col not in {key_col, "_effective_date", "_stable_row_id"}
    ]
    rename_map: dict[str, str] = {}
    left_cols = set(left.columns)
    for col in right_fact_cols:
        merged_col = f"{col}_fact" if col in left_cols else col
        if merged_col in merged.columns:
            rename_map[merged_col] = f"{prefix}_{col}"
    return merged.rename(columns=rename_map)
