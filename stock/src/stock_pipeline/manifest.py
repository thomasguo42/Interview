from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .io_utils import write_json


def write_clean_manifest(
    manifest_root: Path,
    *,
    alias: str,
    clean_path: Path,
    row_count: int,
    columns: list[str],
    notes: str | None = None,
) -> None:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    write_json(
        manifest_root / "clean" / alias / f"{timestamp}.json",
        {
            "alias": alias,
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "clean_path": str(clean_path),
            "row_count": int(row_count),
            "columns": columns,
            "notes": notes,
        },
    )
