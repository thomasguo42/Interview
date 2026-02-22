from __future__ import annotations

import glob
import os
import random
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .io_utils import ensure_dir, write_json


class NasdaqApiError(RuntimeError):
    """Raised when Nasdaq Data Link requests fail."""


@dataclass(frozen=True)
class PullManifest:
    pull_id: str
    alias: str
    datatable: str
    started_at_utc: str
    params: dict[str, Any]
    pages: int
    rows: int
    raw_path: str
    clean_path: str | None = None


class NasdaqDataLinkClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        page_size: int = 10000,
        timeout: int = 60,
        max_retries: int = 5,
        retry_backoff_seconds: float = 1.5,
        prefer_local_zip: bool = False,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Missing required API key environment variable: {api_key_env}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.page_size = page_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.prefer_local_zip = prefer_local_zip
        self.session = requests.Session()

    def _request_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        all_params = {"api_key": self.api_key}
        if params:
            all_params.update(params)
        url = f"{self.base_url}/{path.lstrip('/')}"
        attempt = 0
        while True:
            try:
                resp = self.session.get(url, params=all_params, timeout=self.timeout)
                retryable_http = resp.status_code == 429 or 500 <= resp.status_code < 600
                if resp.status_code >= 400:
                    if retryable_http and attempt < self.max_retries:
                        print(
                            f"[stage1][nasdaq] retryable_http attempt={attempt + 1}/{self.max_retries + 1} "
                            f"status={resp.status_code} path={path}",
                            flush=True,
                        )
                        self._sleep_with_backoff(attempt)
                        attempt += 1
                        continue
                    raise NasdaqApiError(
                        f"Request failed: {resp.status_code} {resp.reason} for {url} params={all_params} body={resp.text[:500]}"
                    )
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt >= self.max_retries:
                    raise NasdaqApiError(
                        f"Request failed after {self.max_retries + 1} attempts for {url} params={all_params}: {exc}"
                    ) from exc
                print(
                    f"[stage1][nasdaq] retry_exception attempt={attempt + 1}/{self.max_retries + 1} "
                    f"type={exc.__class__.__name__} path={path}",
                    flush=True,
                )
                self._sleep_with_backoff(attempt)
                attempt += 1

    def _sleep_with_backoff(self, attempt: int) -> None:
        jitter = random.uniform(0.0, 0.25)
        delay = self.retry_backoff_seconds * (2**attempt) + jitter
        time.sleep(delay)

    def discover_datatables(self, database_code: str = "SHARADAR") -> dict[str, Any]:
        # Endpoint supports filtering by database_code for available datatables.
        return self._request_json("datatables.json", params={"database_code": database_code})

    def datatable_metadata(self, datatable: str) -> dict[str, Any]:
        return self._request_json(f"datatables/{datatable}/metadata.json")

    def iter_datatable_pages(
        self,
        datatable: str,
        params: dict[str, Any] | None = None,
        select_columns: list[str] | None = None,
    ) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
        query: dict[str, Any] = {"qopts.per_page": self.page_size}
        if select_columns:
            query["qopts.columns"] = ",".join(select_columns)
        if params:
            query.update(params)

        cursor_id: str | None = None
        page_payloads: list[dict[str, Any]] = []
        frames: list[pd.DataFrame] = []

        while True:
            page_params = dict(query)
            if cursor_id:
                page_params["qopts.cursor_id"] = cursor_id

            payload = self._request_json(f"datatables/{datatable}.json", page_params)
            page_payloads.append(payload)
            frames.append(self.payload_to_frame(payload))

            cursor_id = self._extract_next_cursor_id(payload)
            if not cursor_id:
                break

        return frames, page_payloads

    @staticmethod
    def _extract_next_cursor_id(payload: dict[str, Any]) -> str | None:
        if "meta" in payload and isinstance(payload["meta"], dict):
            return payload["meta"].get("next_cursor_id")
        datatable = payload.get("datatable", {})
        if isinstance(datatable, dict):
            return datatable.get("next_cursor_id")
        return None

    @staticmethod
    def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
        datatable = payload.get("datatable")
        if not isinstance(datatable, dict):
            raise NasdaqApiError("Unexpected payload format: missing datatable")

        columns_raw = datatable.get("columns", [])
        columns = [column["name"] if isinstance(column, dict) else str(column) for column in columns_raw]
        data = datatable.get("data", [])
        return pd.DataFrame(data, columns=columns)


def pull_datatable_to_raw(
    *,
    client: NasdaqDataLinkClient,
    alias: str,
    datatable: str,
    params: dict[str, Any],
    raw_root: Path,
    manifest_root: Path,
    select_columns: list[str] | None = None,
    local_zip_path: str | None = None,
) -> tuple[pd.DataFrame, PullManifest]:
    started = datetime.now(tz=UTC)
    pull_hash = sha1(f"{alias}|{datatable}|{sorted(params.items())}|{started.isoformat()}".encode()).hexdigest()[:12]
    pull_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{pull_hash}"
    raw_dir = ensure_dir(raw_root / alias / pull_id)

    rows = 0
    payloads: list[dict[str, Any]] = []
    used_local_fallback = False
    local_meta: dict[str, Any] | None = None

    if client.prefer_local_zip:
        local_df, local_meta = _load_local_zip_fallback(
            datatable,
            params,
            select_columns=select_columns,
            local_zip_path=local_zip_path,
        )
        if local_df is not None:
            df = local_df
            rows = int(len(df))
            used_local_fallback = True
            write_json(raw_dir / "local_zip_source.json", local_meta)
            print(
                f"[stage1] prefer_local_zip active for {datatable}; using local zip: {local_meta['zip_path']}",
                flush=True,
            )
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if not used_local_fallback:
        try:
            frames, payloads = client.iter_datatable_pages(datatable, params, select_columns=select_columns)
            for idx, payload in enumerate(payloads, start=1):
                rows += len(payload.get("datatable", {}).get("data", []))
                write_json(raw_dir / f"page_{idx:04d}.json", payload)
                if idx == 1 or idx % 100 == 0:
                    print(f"[stage1][pull:{alias}] wrote page {idx} rows_so_far={rows}", flush=True)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        except Exception:
            local_df, local_meta = _load_local_zip_fallback(
                datatable,
                params,
                select_columns=select_columns,
                local_zip_path=local_zip_path,
            )
            if local_df is None:
                raise
            df = local_df
            rows = int(len(df))
            write_json(raw_dir / "local_zip_source.json", local_meta)
            print(
                f"[stage1] API pull failed for {datatable}; used local zip fallback: {local_meta['zip_path']}",
                flush=True,
            )
    manifest = PullManifest(
        pull_id=pull_id,
        alias=alias,
        datatable=datatable,
        started_at_utc=started.isoformat(),
        params=params,
        pages=len(payloads),
        rows=int(rows),
        raw_path=str(raw_dir),
    )

    write_json(
        manifest_root / "pulls" / alias / f"{pull_id}.json",
        {
            "pull_id": manifest.pull_id,
            "alias": manifest.alias,
            "datatable": manifest.datatable,
            "started_at_utc": manifest.started_at_utc,
            "params": manifest.params,
            "pages": manifest.pages,
            "rows": manifest.rows,
            "raw_path": manifest.raw_path,
            "clean_path": manifest.clean_path,
        },
    )
    return df, manifest


def _datatable_to_zip_glob(datatable: str) -> str:
    # SHARADAR/SEP -> SHARADAR_SEP_*.zip (avoid accidental matches like SF3 -> SF3A/SF3B)
    code = datatable.split("/")[-1].upper()
    return f"SHARADAR_{code}_*.zip"


def _apply_local_filters(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    filtered = df
    for key, value in params.items():
        if key.startswith("qopts.") or key == "api_key":
            continue

        if key.endswith(".gte"):
            col = key[: -len(".gte")]
            if col in filtered.columns:
                filtered = filtered[filtered[col].astype("string") >= str(value)]
            continue

        if key.endswith(".lte"):
            col = key[: -len(".lte")]
            if col in filtered.columns:
                filtered = filtered[filtered[col].astype("string") <= str(value)]
            continue

        if key not in filtered.columns:
            continue

        if isinstance(value, str) and "," in value:
            allowed = [item.strip() for item in value.split(",") if item.strip()]
            series = filtered[key].astype("string")
            if key.lower() == "ticker":
                allowed = [item.upper() for item in allowed]
                series = series.str.upper()
            filtered = filtered[series.isin(allowed)]
        else:
            series = filtered[key]
            if key.lower() == "ticker":
                filtered = filtered[series.astype("string").str.upper() == str(value).upper()]
            else:
                filtered = filtered[series.astype("string") == str(value)]
    return filtered


def _filter_columns_from_params(params: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in params:
        if key.startswith("qopts.") or key == "api_key":
            continue
        base = key
        for suffix in (".gte", ".lte", ".gt", ".lt", ".eq"):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                break
        col = base.strip()
        if not col:
            continue
        low = col.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(col)
    return out


def _resolve_fallback_usecols(
    csv_columns: list[str],
    params: dict[str, Any],
    select_columns: list[str] | None,
) -> list[str] | None:
    if not select_columns:
        return None

    lookup = {col.lower(): col for col in csv_columns}
    merged = list(select_columns) + _filter_columns_from_params(params)
    out: list[str] = []
    seen: set[str] = set()
    for col in merged:
        resolved = lookup.get(col.lower())
        if not resolved:
            continue
        low = resolved.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(resolved)
    return out if out else None


def _load_local_zip_fallback(
    datatable: str,
    params: dict[str, Any],
    select_columns: list[str] | None = None,
    local_zip_path: str | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    if local_zip_path:
        explicit_path = Path(local_zip_path).expanduser()
        if not explicit_path.exists():
            raise NasdaqApiError(f"Configured local_zip_path for {datatable} not found: {explicit_path}")
        matches = [str(explicit_path)]
    else:
        pattern = _datatable_to_zip_glob(datatable)
        matches = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        if len(matches) > 1:
            shown = ", ".join(matches[:5])
            raise NasdaqApiError(
                f"Multiple local zip matches for {datatable}: {shown}. "
                f"Set dataset.local_zip_path to an explicit file."
            )
    if not matches:
        return None, None

    zip_path = matches[0]
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not members:
            return None, None
        member = members[0]

        with zf.open(member) as preview:
            header = pd.read_csv(preview, nrows=0)
        usecols = _resolve_fallback_usecols(list(header.columns), params, select_columns)

        total_in = 0
        total_out = 0
        chunk_idx = 0
        with zf.open(member) as csv_file:
            chunks: list[pd.DataFrame] = []
            for chunk in pd.read_csv(csv_file, chunksize=250000, low_memory=False, usecols=usecols):
                chunk_idx += 1
                total_in += len(chunk)
                reduced = _apply_local_filters(chunk, params)
                if not reduced.empty:
                    chunks.append(reduced)
                    total_out += len(reduced)
                if chunk_idx == 1 or chunk_idx % 20 == 0:
                    print(
                        f"[stage1][fallback:{datatable}] chunks={chunk_idx} rows_read={total_in} rows_kept={total_out}",
                        flush=True,
                    )

    data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    meta = {
        "datatable": datatable,
        "zip_path": str(Path(zip_path).resolve()),
        "zip_path_configured": local_zip_path is not None,
        "zip_member": member,
        "selected_columns": usecols,
        "rows_after_filter": int(len(data)),
        "params": params,
    }
    return data, meta
