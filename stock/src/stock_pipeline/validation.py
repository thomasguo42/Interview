from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Stage1Config, ValidationConfig
from .io_utils import read_parquet, write_json


def _to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def leakage_audit(
    panel: pd.DataFrame,
    sample_size: int = 1000,
    enforce_lastupdated_cutoff_by_prefix: dict[str, bool] | None = None,
) -> dict[str, object]:
    sample = panel.sample(min(sample_size, len(panel)), random_state=42) if not panel.empty else panel.copy()

    issues: list[dict[str, object]] = []
    prefixes = ["fund", "ins", "inst"]
    policy = enforce_lastupdated_cutoff_by_prefix or {}

    for prefix in prefixes:
        knowledge_col = f"{prefix}_knowledge_date"
        if knowledge_col not in sample.columns:
            continue

        kd = _to_timestamp(sample[knowledge_col])
        asof = _to_timestamp(sample["asof_date"])
        violation_mask = kd.notna() & (kd > asof)

        if violation_mask.any():
            bad = sample.loc[violation_mask, ["permaticker", "asof_date", knowledge_col]].head(10)
            issues.append(
                {
                    "type": "knowledge_date_after_asof",
                    "prefix": prefix,
                    "count": int(violation_mask.sum()),
                    "examples": bad.to_dict(orient="records"),
                }
            )

        lastupdated_col = f"{prefix}_lastupdated"
        if lastupdated_col in sample.columns and policy.get(prefix, True):
            lu = _to_timestamp(sample[lastupdated_col])
            violation_mask = lu.notna() & (lu > asof)
            if violation_mask.any():
                bad = sample.loc[violation_mask, ["permaticker", "asof_date", lastupdated_col]].head(10)
                issues.append(
                    {
                        "type": "lastupdated_after_asof",
                        "prefix": prefix,
                        "count": int(violation_mask.sum()),
                        "examples": bad.to_dict(orient="records"),
                    }
                )

    return {
        "checked_rows": int(len(sample)),
        "issues": issues,
        "enforce_lastupdated_cutoff_by_prefix": {k: bool(v) for k, v in policy.items()},
        "passed": len(issues) == 0,
    }


def survivorship_audit(universe: pd.DataFrame) -> dict[str, object]:
    if universe.empty or "asof_date" not in universe.columns or "permaticker" not in universe.columns:
        return {
            "securities_observed": 0,
            "securities_not_present_at_end": 0,
            "universe_count_std": 0.0,
            "static_membership_warning": True,
            "passed": False,
            "reason": "universe table missing or empty",
        }

    work = universe.copy()
    work["asof_date"] = _to_timestamp(work["asof_date"])
    max_date = work["asof_date"].max()

    lifespan = work.groupby("permaticker", as_index=False).agg(
        first_seen=("asof_date", "min"),
        last_seen=("asof_date", "max"),
        days_present=("asof_date", "nunique"),
    )
    not_present_to_end = int((lifespan["last_seen"] < max_date).sum()) if pd.notna(max_date) else 0

    daily_counts = work.groupby("asof_date")["permaticker"].nunique()
    static_membership = bool(daily_counts.nunique() == 1) if not daily_counts.empty else True

    return {
        "securities_observed": int(lifespan["permaticker"].nunique()),
        "securities_not_present_at_end": not_present_to_end,
        "universe_count_std": float(daily_counts.std(ddof=0)) if not daily_counts.empty else 0.0,
        "static_membership_warning": static_membership,
        "passed": (not_present_to_end > 0) and (not static_membership),
    }


def universe_sanity_audit(
    universe: pd.DataFrame,
    min_price: float,
    min_adv20: float,
    validation_cfg: ValidationConfig,
) -> dict[str, object]:
    required = {"eligible_flag", "close_for_returns", "adv20_dollar", "permaticker"}
    if universe.empty or not required.issubset(universe.columns):
        return {
            "eligible_rows": 0,
            "eligible_unique_securities": 0,
            "eligible_equity_rows": 0,
            "eligible_equity_unique_securities": 0,
            "min_price_observed": np.nan,
            "min_adv20_observed": np.nan,
            "market_cap_present_ratio": 0.0,
            "passed": False,
            "reason": "universe table missing required columns",
        }

    eligible = universe[universe["eligible_flag"]].copy()
    eligible_equity = (
        eligible[~eligible.get("is_etf_allowlist", False).astype(bool)]
        if "is_etf_allowlist" in eligible.columns
        else eligible
    )

    out = {
        "eligible_rows": int(len(eligible)),
        "eligible_unique_securities": int(eligible["permaticker"].nunique()) if not eligible.empty else 0,
        "eligible_equity_rows": int(len(eligible_equity)),
        "eligible_equity_unique_securities": int(eligible_equity["permaticker"].nunique()) if not eligible_equity.empty else 0,
        "min_price_observed": float(eligible["close_for_returns"].min()) if not eligible.empty else np.nan,
        "min_adv20_observed": float(eligible["adv20_dollar"].min()) if not eligible.empty else np.nan,
        "market_cap_present_ratio": float(eligible["market_cap"].notna().mean()) if "market_cap" in eligible.columns and len(eligible) else 0.0,
        "passed": True,
    }

    if not eligible.empty:
        out["passed"] = bool(
            (out["min_price_observed"] >= min_price)
            and (out["min_adv20_observed"] >= min_adv20)
            and (out["eligible_equity_unique_securities"] >= validation_cfg.min_equity_eligible_unique)
            and (out["eligible_equity_rows"] >= validation_cfg.min_equity_eligible_rows)
        )

    return out


def etf_sanity_audit(universe: pd.DataFrame, allowlist: list[str]) -> dict[str, object]:
    if universe.empty or "ticker" not in universe.columns or "asof_date" not in universe.columns:
        return {
            "observed": [],
            "missing": sorted(set(allowlist)),
            "history_days": {},
            "passed": False,
            "reason": "universe table missing or empty",
        }

    etf_rows = universe[universe["ticker"].isin(allowlist)]
    observed = sorted(set(etf_rows["ticker"].dropna().astype(str).str.upper()))
    missing = sorted(set(allowlist) - set(observed))
    counts = etf_rows.groupby("ticker")["asof_date"].nunique().to_dict()
    return {
        "observed": observed,
        "missing": missing,
        "history_days": {str(k): int(v) for k, v in counts.items()},
        "passed": len(missing) == 0,
    }


def panel_coverage_audit(panel: pd.DataFrame, validation_cfg: ValidationConfig) -> dict[str, object]:
    if panel.empty:
        return {
            "rows": 0,
            "fundamental_coverage_ratio": 0.0,
            "passed": False,
            "reason": "daily panel missing or empty",
        }

    fund_col = "fund_knowledge_date" if "fund_knowledge_date" in panel.columns else None
    if not fund_col and "fund_market_cap" in panel.columns:
        fund_col = "fund_market_cap"

    if not fund_col:
        return {
            "rows": int(len(panel)),
            "fundamental_coverage_ratio": 0.0,
            "passed": False,
            "reason": "fundamental columns missing from daily panel",
        }

    coverage = float(panel[fund_col].notna().mean())
    return {
        "rows": int(len(panel)),
        "coverage_column": fund_col,
        "fundamental_coverage_ratio": coverage,
        "min_required_ratio": float(validation_cfg.min_fundamental_coverage_ratio),
        "passed": coverage >= validation_cfg.min_fundamental_coverage_ratio,
    }


def universe_market_cap_alignment_audit(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    validation_cfg: ValidationConfig,
) -> dict[str, object]:
    required_panel = {"permaticker", "asof_date", "fund_market_cap"}
    required_universe = {"permaticker", "asof_date", "market_cap"}
    if panel.empty or universe.empty:
        return {
            "checked_rows": 0,
            "passed": False,
            "reason": "panel or universe missing/empty",
        }
    if not required_panel.issubset(panel.columns) or not required_universe.issubset(universe.columns):
        return {
            "checked_rows": 0,
            "passed": False,
            "reason": "required market-cap columns missing",
        }

    panel_view = panel[["permaticker", "asof_date", "fund_market_cap"]].copy()
    panel_view["asof_date"] = pd.to_datetime(panel_view["asof_date"], errors="coerce").dt.normalize()
    panel_view = panel_view.dropna(subset=["fund_market_cap", "asof_date", "permaticker"]).copy()
    if panel_view.empty:
        return {
            "checked_rows": 0,
            "passed": False,
            "reason": "no rows with fund_market_cap available",
        }

    sample_size = min(len(panel_view), int(validation_cfg.market_cap_alignment_sample_size))
    panel_sample = panel_view.sample(sample_size, random_state=42) if sample_size < len(panel_view) else panel_view

    universe_view = universe[["permaticker", "asof_date", "market_cap"]].copy()
    universe_view["asof_date"] = pd.to_datetime(universe_view["asof_date"], errors="coerce").dt.normalize()
    universe_view = universe_view.dropna(subset=["asof_date", "permaticker"]).copy()
    universe_view = universe_view.drop_duplicates(["permaticker", "asof_date"], keep="last")

    merged = panel_sample.merge(
        universe_view,
        on=["permaticker", "asof_date"],
        how="left",
        validate="one_to_one",
    )

    missing_mask = merged["market_cap"].isna()
    comparable = merged.loc[~missing_mask].copy()
    mismatch_count = 0
    mismatch_examples: list[dict[str, object]] = []
    if not comparable.empty:
        left_vals = pd.to_numeric(comparable["market_cap"], errors="coerce").to_numpy(dtype="float64")
        right_vals = pd.to_numeric(comparable["fund_market_cap"], errors="coerce").to_numpy(dtype="float64")
        equal = np.isclose(
            left_vals,
            right_vals,
            rtol=float(validation_cfg.market_cap_match_rtol),
            atol=float(validation_cfg.market_cap_match_atol),
        )
        mismatch_count = int((~equal).sum())
        if mismatch_count > 0:
            bad = comparable.loc[~equal, ["permaticker", "asof_date", "market_cap", "fund_market_cap"]].head(10)
            mismatch_examples = bad.to_dict(orient="records")

    checked = int(len(merged))
    missing_ratio = float(missing_mask.mean()) if checked else 1.0
    comparable_count = int((~missing_mask).sum())
    mismatch_ratio = float(mismatch_count / comparable_count) if comparable_count else 0.0

    passed = (
        missing_ratio <= float(validation_cfg.max_market_cap_missing_ratio)
        and mismatch_ratio <= float(validation_cfg.max_market_cap_mismatch_ratio)
    )

    return {
        "checked_rows": checked,
        "sample_size": sample_size,
        "missing_market_cap_count": int(missing_mask.sum()),
        "missing_market_cap_ratio": missing_ratio,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_ratio,
        "max_missing_ratio": float(validation_cfg.max_market_cap_missing_ratio),
        "max_mismatch_ratio": float(validation_cfg.max_market_cap_mismatch_ratio),
        "match_rtol": float(validation_cfg.market_cap_match_rtol),
        "match_atol": float(validation_cfg.market_cap_match_atol),
        "examples": mismatch_examples,
        "passed": passed,
    }


def date_coverage_audit(config: Stage1Config) -> dict[str, object]:
    checks = {
        "equity_prices_daily": ("equity_prices_daily.parquet", "date"),
        "etf_prices_daily": ("etf_prices_daily.parquet", "date"),
        "tradable_universe_daily": ("tradable_universe_daily.parquet", "asof_date"),
        "daily_panel": ("daily_panel.parquet", "asof_date"),
    }
    expected_start = pd.Timestamp(config.project.start_date).normalize()
    max_gap_days = int(config.validation.max_start_date_gap_days)

    out: dict[str, object] = {"expected_start_date": expected_start.date().isoformat(), "max_start_gap_days": max_gap_days}
    passed = True

    for alias, (filename, date_col) in checks.items():
        path = config.storage.clean_root / filename
        if not path.exists():
            out[alias] = {"exists": False, "passed": False, "reason": "file missing"}
            passed = False
            continue
        df = read_parquet(path)
        if date_col not in df.columns or df.empty:
            out[alias] = {"exists": True, "passed": False, "reason": "date column missing or empty"}
            passed = False
            continue
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.normalize()
        if dates.empty:
            out[alias] = {"exists": True, "passed": False, "reason": "no valid dates"}
            passed = False
            continue
        observed_start = dates.min()
        start_gap_days = int((observed_start - expected_start).days)
        alias_passed = start_gap_days <= max_gap_days
        out[alias] = {
            "exists": True,
            "observed_start_date": observed_start.date().isoformat(),
            "observed_end_date": dates.max().date().isoformat(),
            "start_gap_days": start_gap_days,
            "passed": alias_passed,
        }
        passed = passed and alias_passed

    out["passed"] = passed
    return out


def _extreme_jump_ratio(df: pd.DataFrame, threshold: float) -> float:
    returns = (
        df.sort_values(["permaticker", "date"])
        .groupby("permaticker", group_keys=False)["close_for_returns"]
        .pct_change()
    )
    valid = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return 0.0
    return float((valid.abs() > threshold).mean())


def _active_missing_ratio(df: pd.DataFrame, calendar: pd.Series) -> float:
    if df.empty or calendar.empty:
        return 0.0

    dates = pd.to_datetime(df["date"]).dt.normalize()
    max_date = dates.max()
    active = df.loc[pd.to_datetime(df["date"]) == max_date, "permaticker"].dropna().unique()
    if len(active) == 0:
        return 0.0

    recent_days = pd.to_datetime(calendar)
    recent_days = recent_days[recent_days <= max_date]
    recent_days = recent_days.sort_values().tail(252)
    if recent_days.empty:
        return 0.0

    recent_set = set(recent_days.tolist())
    active_df = df[df["permaticker"].isin(active)].copy()
    active_df["date"] = pd.to_datetime(active_df["date"]).dt.normalize()
    if active_df.empty:
        return 0.0

    active_df = active_df.sort_values(["permaticker", "date"])
    active_recent = active_df[active_df["date"].isin(recent_set)]
    observed = active_recent.groupby("permaticker")["date"].nunique()
    first_seen = active_df.groupby("permaticker")["date"].min()

    if observed.empty or first_seen.empty:
        return 0.0

    window_start = recent_days.min()
    trading_days_index = pd.DatetimeIndex(recent_days)

    expected_by_ticker: list[int] = []
    for permaticker, first_date in first_seen.items():
        effective_start = max(window_start, first_date)
        expected_days = int((trading_days_index >= effective_start).sum())
        expected_by_ticker.append(max(expected_days, 0))

    expected_series = pd.Series(expected_by_ticker, index=first_seen.index, dtype="int64")
    observed = observed.reindex(expected_series.index, fill_value=0).astype("int64")

    missing_total = int((expected_series - observed).clip(lower=0).sum())
    denom = int(expected_series.sum())
    return float(missing_total / denom) if denom else 0.0


def price_quality_audit(clean_root: Path, validation_cfg: ValidationConfig) -> dict[str, object]:
    results: dict[str, object] = {}
    calendar_path = clean_root / "trading_calendar_nyse.parquet"
    trading_days = pd.Series(dtype="datetime64[ns]")
    if calendar_path.exists():
        calendar_df = read_parquet(calendar_path)
        if "trade_date" in calendar_df.columns:
            trading_days = pd.to_datetime(calendar_df["trade_date"]).dt.normalize()

    for alias in ("equity_prices_daily", "etf_prices_daily"):
        path = clean_root / f"{alias}.parquet"
        if not path.exists():
            continue
        df = read_parquet(path)
        required = {"permaticker", "date", "close_for_returns", "volume_for_liquidity"}
        if not required.issubset(df.columns):
            results[alias] = {
                "rows": int(len(df)),
                "passed": False,
                "reason": "missing required price columns",
            }
            continue
        duplicate_count = int(df.duplicated(["permaticker", "date"]).sum())
        close_numeric = pd.to_numeric(df["close_for_returns"], errors="coerce")
        volume_numeric = pd.to_numeric(df["volume_for_liquidity"], errors="coerce")
        non_positive_close = int((close_numeric <= 0).sum())
        non_positive_volume = int((volume_numeric < 0).sum())
        zero_volume = int((volume_numeric == 0).sum())
        zero_volume_ratio = float(zero_volume / len(df)) if len(df) else 0.0
        jump_ratio = _extreme_jump_ratio(df, threshold=validation_cfg.extreme_jump_threshold)
        missing_active_ratio = _active_missing_ratio(df, trading_days)
        results[alias] = {
            "rows": int(len(df)),
            "duplicates": duplicate_count,
            "non_positive_close": non_positive_close,
            "negative_volume": non_positive_volume,
            "zero_volume": zero_volume,
            "zero_volume_ratio": zero_volume_ratio,
            "extreme_jump_ratio": jump_ratio,
            "missing_active_ratio": missing_active_ratio,
            "passed": (
                duplicate_count == 0
                and non_positive_close == 0
                and non_positive_volume == 0
                and zero_volume_ratio <= validation_cfg.max_zero_volume_ratio
                and jump_ratio <= validation_cfg.max_extreme_jump_ratio
                and missing_active_ratio <= validation_cfg.max_missing_active_ratio
            ),
        }
    return results


def run_stage1_validation(config: Stage1Config) -> dict[str, object]:
    report: dict[str, object] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "checks": {},
    }

    panel_path = config.storage.clean_root / "daily_panel.parquet"
    universe_path = config.storage.clean_root / "tradable_universe_daily.parquet"

    panel = read_parquet(panel_path) if panel_path.exists() else pd.DataFrame()
    universe = read_parquet(universe_path) if universe_path.exists() else pd.DataFrame()

    enforce_lastupdated_policy = {
        "fund": config.dataset("fundamentals").enforce_lastupdated_cutoff,
        "ins": config.dataset("insiders").enforce_lastupdated_cutoff,
        "inst": config.dataset("institutions").enforce_lastupdated_cutoff,
    }

    report["checks"]["leakage"] = leakage_audit(
        panel,
        enforce_lastupdated_cutoff_by_prefix=enforce_lastupdated_policy,
    )
    report["checks"]["survivorship"] = survivorship_audit(universe)
    report["checks"]["universe_sanity"] = universe_sanity_audit(
        universe,
        min_price=config.universe.min_price,
        min_adv20=config.universe.min_adv20_dollar,
        validation_cfg=config.validation,
    )
    report["checks"]["etf_sanity"] = etf_sanity_audit(universe, config.etf_allowlist)
    report["checks"]["panel_coverage"] = panel_coverage_audit(panel, config.validation)
    report["checks"]["universe_market_cap_alignment"] = universe_market_cap_alignment_audit(
        panel,
        universe,
        config.validation,
    )
    report["checks"]["date_coverage"] = date_coverage_audit(config)
    report["checks"]["price_quality"] = price_quality_audit(config.storage.clean_root, config.validation)

    passed = True
    for value in report["checks"].values():
        if isinstance(value, dict) and "passed" in value:
            passed = passed and bool(value["passed"])
        elif isinstance(value, dict):
            for sub in value.values():
                if isinstance(sub, dict) and "passed" in sub:
                    passed = passed and bool(sub["passed"])

    report["passed"] = passed

    out_path = config.storage.manifest_root / "validation" / "stage1_validation_report.json"
    write_json(out_path, report)
    return report
