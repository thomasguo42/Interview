from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Stage1Config
from .io_utils import ensure_dir, write_json
from .nasdaq import NasdaqDataLinkClient


@dataclass(frozen=True)
class DatasetSchemaLock:
    alias: str
    datatable: str
    columns: list[dict[str, Any]]
    resolved_fields: dict[str, str | None]
    required_columns: list[str]


class SchemaResolutionError(ValueError):
    """Raised when required schema columns cannot be resolved."""


def required_pull_columns(schema_lock: DatasetSchemaLock, extra_columns: list[str] | None = None) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()

    def add(column: str | None) -> None:
        if not column:
            return
        col_norm = column.strip()
        if not col_norm:
            return
        key = col_norm.lower()
        if key in seen:
            return
        seen.add(key)
        cols.append(col_norm)

    for col in schema_lock.required_columns:
        add(col)
    for col in schema_lock.resolved_fields.values():
        add(col)
    if extra_columns:
        for col in extra_columns:
            add(col)
    return cols


def resolve_column_name(available_columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    available_lookup = {col.lower(): col for col in available_columns}
    for candidate in candidates:
        match = available_lookup.get(candidate.lower())
        if match:
            return match
    if required:
        raise SchemaResolutionError(
            f"Could not resolve column from candidates={candidates}; available sample={available_columns[:20]}"
        )
    return None


def resolve_columns_from_frame(df: pd.DataFrame, field_candidates: dict[str, list[str]]) -> dict[str, str | None]:
    available = list(df.columns)
    resolved: dict[str, str | None] = {}
    for semantic_field, candidates in field_candidates.items():
        resolved[semantic_field] = resolve_column_name(available, candidates, required=False)
    return resolved


def _extract_metadata_columns(metadata_payload: dict[str, Any]) -> list[dict[str, Any]]:
    datatable = metadata_payload.get("datatable", {})
    cols = datatable.get("columns", [])
    if not isinstance(cols, list):
        return []
    normalized: list[dict[str, Any]] = []
    for col in cols:
        if isinstance(col, dict):
            normalized.append(col)
        else:
            normalized.append({"name": str(col), "type": None})
    return normalized


def lock_dataset_schemas(config: Stage1Config, client: NasdaqDataLinkClient, manifest_root: Path) -> dict[str, DatasetSchemaLock]:
    locks: dict[str, DatasetSchemaLock] = {}
    target_dir = ensure_dir(manifest_root / "schema")

    for alias, dataset_cfg in config.nasdaq.datasets.items():
        metadata = client.datatable_metadata(dataset_cfg.datatable)
        columns = _extract_metadata_columns(metadata)
        available = [col.get("name", "") for col in columns]
        resolved: dict[str, str | None] = {}

        for semantic_field, candidates in dataset_cfg.field_candidates.items():
            resolved[semantic_field] = resolve_column_name(available, candidates, required=False)

        missing_required = [c for c in dataset_cfg.required_columns if c.lower() not in {a.lower() for a in available}]
        if missing_required:
            raise SchemaResolutionError(
                f"Dataset {alias} ({dataset_cfg.datatable}) missing required columns: {missing_required}"
            )

        lock = DatasetSchemaLock(
            alias=alias,
            datatable=dataset_cfg.datatable,
            columns=columns,
            resolved_fields=resolved,
            required_columns=dataset_cfg.required_columns,
        )
        locks[alias] = lock

        write_json(
            target_dir / f"{alias}.json",
            {
                "alias": alias,
                "datatable": dataset_cfg.datatable,
                "columns": columns,
                "resolved_fields": resolved,
                "required_columns": dataset_cfg.required_columns,
            },
        )

    return locks
