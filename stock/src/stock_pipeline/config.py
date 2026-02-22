from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    timezone: str
    decision_timing: str
    rebalance_frequency: str
    execution_timing: str
    start_date: str
    end_date: str
    holdout_years: int


@dataclass(frozen=True)
class StorageConfig:
    raw_root: Path
    clean_root: Path
    manifest_root: Path


@dataclass(frozen=True)
class UniverseConfig:
    min_price: float
    min_adv20_dollar: float
    min_market_cap: float


@dataclass(frozen=True)
class CostScenario:
    equities_bps_per_side: int
    etfs_bps_per_side: int


@dataclass(frozen=True)
class CostsConfig:
    base: CostScenario
    conservative: CostScenario


@dataclass(frozen=True)
class DatasetConfig:
    datatable: str
    filters: dict[str, Any]
    field_candidates: dict[str, list[str]]
    required_columns: list[str]
    ticker_filter_column: str | None = None
    availability_lag_days: int = 0
    enforce_lastupdated_cutoff: bool = True
    local_zip_path: str | None = None


@dataclass(frozen=True)
class NasdaqConfig:
    base_url: str
    api_key_env: str
    page_size: int
    request_timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float
    prefer_local_zip: bool
    datasets: dict[str, DatasetConfig]


@dataclass(frozen=True)
class ValidationConfig:
    min_equity_eligible_unique: int
    min_equity_eligible_rows: int
    extreme_jump_threshold: float
    max_extreme_jump_ratio: float
    max_missing_active_ratio: float
    max_zero_volume_ratio: float
    min_fundamental_coverage_ratio: float
    max_start_date_gap_days: int
    market_cap_alignment_sample_size: int
    max_market_cap_missing_ratio: float
    max_market_cap_mismatch_ratio: float
    market_cap_match_rtol: float
    market_cap_match_atol: float


@dataclass(frozen=True)
class Stage1Config:
    project: ProjectConfig
    storage: StorageConfig
    universe: UniverseConfig
    etf_allowlist: list[str]
    costs: CostsConfig
    nasdaq: NasdaqConfig
    validation: ValidationConfig

    def dataset(self, alias: str) -> DatasetConfig:
        try:
            return self.nasdaq.datasets[alias]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset alias: {alias}") from exc


def _load_dataset_config(value: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        datatable=value["datatable"],
        filters=dict(value.get("filters", {})),
        field_candidates={
            key: list(candidates) for key, candidates in value.get("field_candidates", {}).items()
        },
        required_columns=list(value.get("required_columns", [])),
        ticker_filter_column=value.get("ticker_filter_column"),
        availability_lag_days=int(value.get("availability_lag_days", 0)),
        enforce_lastupdated_cutoff=bool(value.get("enforce_lastupdated_cutoff", True)),
        local_zip_path=value.get("local_zip_path"),
    )


def load_config(path: str | Path) -> Stage1Config:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())

    project = ProjectConfig(**raw["project"])
    storage = StorageConfig(
        raw_root=Path(raw["storage"]["raw_root"]),
        clean_root=Path(raw["storage"]["clean_root"]),
        manifest_root=Path(raw["storage"]["manifest_root"]),
    )
    universe = UniverseConfig(**raw["universe"])

    costs = CostsConfig(
        base=CostScenario(**raw["costs"]["base"]),
        conservative=CostScenario(**raw["costs"]["conservative"]),
    )

    nasdaq_datasets = {
        alias: _load_dataset_config(dataset_cfg)
        for alias, dataset_cfg in raw["nasdaq"]["datasets"].items()
    }
    nasdaq = NasdaqConfig(
        base_url=raw["nasdaq"]["base_url"],
        api_key_env=raw["nasdaq"]["api_key_env"],
        page_size=int(raw["nasdaq"]["page_size"]),
        request_timeout_seconds=int(raw["nasdaq"].get("request_timeout_seconds", 60)),
        max_retries=int(raw["nasdaq"].get("max_retries", 5)),
        retry_backoff_seconds=float(raw["nasdaq"].get("retry_backoff_seconds", 1.5)),
        prefer_local_zip=bool(raw["nasdaq"].get("prefer_local_zip", False)),
        datasets=nasdaq_datasets,
    )

    validation = ValidationConfig(
        min_equity_eligible_unique=int(raw.get("validation", {}).get("min_equity_eligible_unique", 50)),
        min_equity_eligible_rows=int(raw.get("validation", {}).get("min_equity_eligible_rows", 1000)),
        extreme_jump_threshold=float(raw.get("validation", {}).get("extreme_jump_threshold", 0.8)),
        max_extreme_jump_ratio=float(raw.get("validation", {}).get("max_extreme_jump_ratio", 0.01)),
        max_missing_active_ratio=float(raw.get("validation", {}).get("max_missing_active_ratio", 0.05)),
        max_zero_volume_ratio=float(raw.get("validation", {}).get("max_zero_volume_ratio", 0.05)),
        min_fundamental_coverage_ratio=float(raw.get("validation", {}).get("min_fundamental_coverage_ratio", 0.5)),
        max_start_date_gap_days=int(raw.get("validation", {}).get("max_start_date_gap_days", 90)),
        market_cap_alignment_sample_size=int(raw.get("validation", {}).get("market_cap_alignment_sample_size", 200000)),
        max_market_cap_missing_ratio=float(raw.get("validation", {}).get("max_market_cap_missing_ratio", 0.01)),
        max_market_cap_mismatch_ratio=float(raw.get("validation", {}).get("max_market_cap_mismatch_ratio", 0.01)),
        market_cap_match_rtol=float(raw.get("validation", {}).get("market_cap_match_rtol", 1e-6)),
        market_cap_match_atol=float(raw.get("validation", {}).get("market_cap_match_atol", 1e-3)),
    )

    return Stage1Config(
        project=project,
        storage=storage,
        universe=universe,
        etf_allowlist=[symbol.upper() for symbol in raw.get("etf_allowlist", [])],
        costs=costs,
        nasdaq=nasdaq,
        validation=validation,
    )


def resolve_filters(filters: dict[str, Any], *, start_date: str, end_date: str) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in filters.items():
        if isinstance(value, str):
            resolved[key] = value.format(start_date=start_date, end_date=end_date)
        else:
            resolved[key] = value
    return resolved
