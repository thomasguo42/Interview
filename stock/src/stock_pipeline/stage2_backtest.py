from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .asof import deterministic_asof_join
from .io_utils import ensure_dir, read_json, read_parquet, write_json, write_parquet
from .stage2_config import Stage2Config, resolve_cost_bps


SUPPORTED_STRATEGIES = {
    "MOM_126X21",
    "MOM_252X21",
    "REV_21",
    "INS_90",
    "INST_CHG",
    "COMBO_RAW",
    "COMBO_OVERLAY",
}


@dataclass(frozen=True)
class PeriodSplit:
    development_start: pd.Timestamp
    development_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    forward_start: pd.Timestamp | None
    forward_end: pd.Timestamp | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "development_start": self.development_start.date().isoformat(),
            "development_end": self.development_end.date().isoformat(),
            "holdout_start": self.holdout_start.date().isoformat(),
            "holdout_end": self.holdout_end.date().isoformat(),
        }
        if self.forward_start is not None and self.forward_end is not None:
            payload["forward_start"] = self.forward_start.date().isoformat()
            payload["forward_end"] = self.forward_end.date().isoformat()
        return payload


@dataclass
class PendingRebalance:
    decision_date: pd.Timestamp
    execution_date: pd.Timestamp
    reference_nav: float
    target_weights: dict[int, float]
    decision_adv: dict[int, float]
    selected_equities: list[int]
    risk_off: bool
    strategy_id: str


class Stage2DataError(ValueError):
    """Raised when stage-2 input data is missing or invalid."""


class PriceAccessor:
    def __init__(
        self,
        equity_adj: pd.DataFrame,
        etf_adj: pd.DataFrame,
        end_date_by_permaticker: pd.Series,
    ) -> None:
        eq = equity_adj.copy()
        etf = etf_adj.copy()
        eq["asset_class"] = "equity"
        etf["asset_class"] = "etf"

        keep_cols = ["permaticker", "ticker", "date", "open_adj", "close_adj", "adv20_dollar", "asset_class"]
        parts: list[pd.DataFrame] = []
        if not eq.empty:
            parts.append(eq[keep_cols])
        if not etf.empty:
            parts.append(etf[keep_cols])
        prices = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep_cols)
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
        prices["ticker"] = prices["ticker"].astype("string").str.upper()
        prices["permaticker"] = pd.to_numeric(prices["permaticker"], errors="coerce").astype("Int64")
        prices = (
            prices.dropna(subset=["permaticker", "date"])
            .sort_values(["permaticker", "date", "asset_class", "ticker"])
            .drop_duplicates(["permaticker", "date"], keep="last")
            .reset_index(drop=True)
        )

        self._index = prices.set_index(["permaticker", "date"])[
            ["open_adj", "close_adj", "adv20_dollar", "ticker", "asset_class"]
        ].sort_index()
        self._meta = (
            prices.sort_values(["ticker", "date", "permaticker"])
            .dropna(subset=["ticker"])
            .drop_duplicates(["ticker"], keep="last")
            .set_index("ticker")
        )
        self._asset_class_by_permaticker = (
            prices.sort_values(["permaticker", "date"])
            .drop_duplicates(["permaticker"], keep="last")
            .set_index("permaticker")["asset_class"]
        )
        self._end_date = pd.to_datetime(end_date_by_permaticker, errors="coerce").dt.normalize()

    def _row(self, permaticker: int, date: pd.Timestamp) -> pd.Series | None:
        key = (int(permaticker), pd.Timestamp(date).normalize())
        try:
            row = self._index.loc[key]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            return row.iloc[-1]
        return row

    def open_adj(self, permaticker: int, date: pd.Timestamp) -> float | None:
        row = self._row(permaticker, date)
        if row is None:
            return None
        value = pd.to_numeric(pd.Series([row["open_adj"]]), errors="coerce").iloc[0]
        return None if pd.isna(value) else float(value)

    def close_adj(self, permaticker: int, date: pd.Timestamp) -> float | None:
        row = self._row(permaticker, date)
        if row is None:
            return None
        value = pd.to_numeric(pd.Series([row["close_adj"]]), errors="coerce").iloc[0]
        return None if pd.isna(value) else float(value)

    def adv20(self, permaticker: int, date: pd.Timestamp) -> float | None:
        row = self._row(permaticker, date)
        if row is None:
            return None
        value = pd.to_numeric(pd.Series([row["adv20_dollar"]]), errors="coerce").iloc[0]
        return None if pd.isna(value) else float(value)

    def ticker(self, permaticker: int, date: pd.Timestamp) -> str:
        row = self._row(permaticker, date)
        if row is not None and pd.notna(row["ticker"]):
            return str(row["ticker"]).upper()
        return str(permaticker)

    def asset_class(self, permaticker: int) -> str:
        value = self._asset_class_by_permaticker.get(int(permaticker))
        return str(value) if pd.notna(value) else "equity"

    def permaticker_for_ticker(self, ticker: str) -> int:
        t = ticker.upper()
        if t not in self._meta.index:
            raise Stage2DataError(f"Ticker {ticker} not found in adjusted price tables")
        value = self._meta.loc[t, "permaticker"]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return int(value)

    def last_known_end_date(self, permaticker: int) -> pd.Timestamp | None:
        value = self._end_date.get(int(permaticker))
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).normalize()


def _normalize_strategy_id(strategy: str, config: Stage2Config) -> str:
    raw = strategy.strip().upper()
    if raw == "COMBO" and config.strategies.combo_alias_to_raw:
        raw = "COMBO_RAW"
    if raw not in SUPPORTED_STRATEGIES:
        choices = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise Stage2DataError(f"Unknown strategy {strategy}. Supported: {choices}")
    return raw


def _load_calendar(config: Stage2Config) -> pd.DatetimeIndex:
    for path in (config.execution.calendar_primary, config.execution.calendar_fallback):
        if not path.exists():
            continue
        df = read_parquet(path)
        if "trade_date" in df.columns:
            col = "trade_date"
        elif "date" in df.columns:
            col = "date"
        else:
            continue
        dates = pd.to_datetime(df[col], errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
        if not dates.empty:
            return pd.DatetimeIndex(dates)
    raise Stage2DataError(
        f"Could not load trading calendar from {config.execution.calendar_primary} or {config.execution.calendar_fallback}"
    )


def _calendar_window(calendar: pd.DatetimeIndex, start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    window = calendar[(calendar >= start) & (calendar <= end)]
    if window.empty:
        raise Stage2DataError(f"No trading days in configured stage-2 window {start.date()}..{end.date()}")
    return window


def _build_decision_execution_schedule(trading_days: pd.DatetimeIndex, execution_delay_days: int = 0) -> pd.DataFrame:
    delay = max(0, int(execution_delay_days))
    dates = pd.Series(pd.to_datetime(trading_days), name="decision_date")
    iso = dates.dt.isocalendar()
    week_max = (
        pd.DataFrame(
            {
                "decision_date": dates,
                "iso_year": iso["year"],
                "iso_week": iso["week"],
            }
        )
        .groupby(["iso_year", "iso_week"], as_index=False)["decision_date"]
        .max()
        .sort_values("decision_date")
        .reset_index(drop=True)
    )

    positions = np.searchsorted(
        trading_days.to_numpy(dtype="datetime64[ns]"),
        week_max["decision_date"].to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    if delay > 0:
        positions = positions + delay
    valid = positions < len(trading_days)
    week_max = week_max.loc[valid].copy()
    week_max["execution_date"] = pd.DatetimeIndex(trading_days[positions[valid]])
    return week_max[["decision_date", "execution_date"]].reset_index(drop=True)


def _compute_period_split(trading_days: pd.DatetimeIndex, holdout_years: int) -> PeriodSplit:
    by_year = pd.DataFrame({"date": trading_days})
    by_year["year"] = by_year["date"].dt.year
    by_year["month"] = by_year["date"].dt.month

    month_span = by_year.groupby("year", as_index=False).agg(min_month=("month", "min"), max_month=("month", "max"))
    full_years = month_span[(month_span["min_month"] == 1) & (month_span["max_month"] == 12)]["year"].tolist()
    if len(full_years) < holdout_years:
        raise Stage2DataError(
            f"Not enough full calendar years for holdout: required={holdout_years}, available={len(full_years)}"
        )

    holdout_year_list = full_years[-holdout_years:]
    holdout_start = pd.Timestamp(f"{holdout_year_list[0]}-01-01")
    holdout_end = pd.Timestamp(f"{holdout_year_list[-1]}-12-31")

    development_dates = trading_days[trading_days < holdout_start]
    if len(development_dates) == 0:
        raise Stage2DataError("Development period is empty after holdout split")

    forward_dates = trading_days[trading_days > holdout_end]
    forward_start = pd.Timestamp(forward_dates.min()) if len(forward_dates) else None
    forward_end = pd.Timestamp(forward_dates.max()) if len(forward_dates) else None

    return PeriodSplit(
        development_start=pd.Timestamp(development_dates.min()),
        development_end=pd.Timestamp(development_dates.max()),
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        forward_start=forward_start,
        forward_end=forward_end,
    )


def _label_period(dates: pd.Series, split: PeriodSplit, include_forward: bool) -> pd.Series:
    ts = pd.to_datetime(dates, errors="coerce").dt.normalize()
    labels = pd.Series("development", index=dates.index, dtype="string")
    labels = labels.mask((ts >= split.holdout_start) & (ts <= split.holdout_end), "holdout")
    if split.forward_start is not None:
        labels = labels.mask(ts >= split.forward_start, "forward")
    if not include_forward:
        labels = labels.mask(labels == "forward", "excluded")
    return labels


def _prepare_adjusted_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("date",):
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    numeric_cols = ["open", "close", "close_adj", "adv20_dollar", "realized_vol20"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    adj_factor = pd.Series(np.nan, index=out.index, dtype="float64")
    valid = out["close"].notna() & (out["close"] > 0) & out["close_adj"].notna()
    adj_factor.loc[valid] = out.loc[valid, "close_adj"] / out.loc[valid, "close"]
    out["open_adj"] = out["open"] * adj_factor
    out["ticker"] = out["ticker"].astype("string").str.upper().str.strip()
    out["permaticker"] = pd.to_numeric(out["permaticker"], errors="coerce").astype("Int64")
    if "realized_vol20" not in out.columns:
        out["realized_vol20"] = np.nan
    keep_cols = [
        "permaticker",
        "ticker",
        "date",
        "open",
        "close",
        "close_adj",
        "adv20_dollar",
        "realized_vol20",
        "open_adj",
    ]
    keep_cols = [col for col in keep_cols if col in out.columns]
    out = out[keep_cols].copy()
    out = (
        out.dropna(subset=["permaticker", "date"])
        .sort_values(["permaticker", "date", "ticker"])
        .drop_duplicates(["permaticker", "date"], keep="last")
        .reset_index(drop=True)
    )
    return out


def prepare_adjusted_price_tables(config: Stage2Config) -> dict[str, str]:
    base_cols = ["permaticker", "ticker", "date", "open", "close", "close_adj", "adv20_dollar", "realized_vol20"]
    try:
        eq = read_parquet(config.storage.clean_root / config.storage.equity_prices, columns=base_cols)
    except Exception:
        eq = read_parquet(
            config.storage.clean_root / config.storage.equity_prices,
            columns=["permaticker", "ticker", "date", "open", "close", "close_adj", "adv20_dollar"],
        )
        eq["realized_vol20"] = np.nan
    try:
        etf = read_parquet(config.storage.clean_root / config.storage.etf_prices, columns=base_cols)
    except Exception:
        etf = read_parquet(
            config.storage.clean_root / config.storage.etf_prices,
            columns=["permaticker", "ticker", "date", "open", "close", "close_adj", "adv20_dollar"],
        )
        etf["realized_vol20"] = np.nan

    eq_adj = _prepare_adjusted_prices(eq)
    etf_adj = _prepare_adjusted_prices(etf)

    eq_path = config.storage.clean_root / config.storage.equity_prices_adj
    etf_path = config.storage.clean_root / config.storage.etf_prices_adj
    write_parquet(eq_adj, eq_path)
    write_parquet(etf_adj, etf_path)
    return {"equity_adj": str(eq_path), "etf_adj": str(etf_path)}


def _load_adjusted_price_tables(config: Stage2Config, *, load_equity: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    eq_path = config.storage.clean_root / config.storage.equity_prices_adj
    etf_path = config.storage.clean_root / config.storage.etf_prices_adj
    if not eq_path.exists() or not etf_path.exists():
        prepare_adjusted_price_tables(config)
    if load_equity:
        eq_adj = read_parquet(
            eq_path,
            columns=["permaticker", "ticker", "date", "open_adj", "close_adj", "adv20_dollar"],
        )
    else:
        eq_adj = pd.DataFrame(
            columns=["permaticker", "ticker", "date", "open_adj", "close_adj", "adv20_dollar"]
        )
    try:
        etf_adj = read_parquet(
            etf_path,
            columns=["permaticker", "ticker", "date", "open_adj", "close_adj", "adv20_dollar", "realized_vol20"],
        )
    except Exception:
        etf_adj = read_parquet(
            etf_path,
            columns=["permaticker", "ticker", "date", "open_adj", "close_adj", "adv20_dollar"],
        )
        etf_adj["realized_vol20"] = np.nan
    return eq_adj, etf_adj


def _compute_price_features(equity_adj: pd.DataFrame) -> pd.DataFrame:
    eq = equity_adj[["permaticker", "date", "ticker", "close_adj", "adv20_dollar"]].copy()
    eq["close_adj"] = pd.to_numeric(eq["close_adj"], errors="coerce")
    eq = eq.sort_values(["permaticker", "date"]).reset_index(drop=True)

    grouped = eq.groupby("permaticker", group_keys=False)["close_adj"]
    eq["mom_126x21"] = grouped.shift(21) / grouped.shift(126) - 1.0
    eq["mom_252x21"] = grouped.shift(21) / grouped.shift(252) - 1.0
    eq["rev_21"] = -(eq["close_adj"] / grouped.shift(21) - 1.0)
    return eq


def _compute_insider_window_scores(insiders_agg: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if insiders_agg.empty:
        return pd.DataFrame(columns=["permaticker", "knowledge_date", "ins_90"])
    work = insiders_agg[["permaticker", "knowledge_date", "insider_net_dollar"]].copy()
    work["knowledge_date"] = pd.to_datetime(work["knowledge_date"], errors="coerce").dt.normalize()
    work["insider_net_dollar"] = pd.to_numeric(work["insider_net_dollar"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["permaticker", "knowledge_date"]).sort_values(["permaticker", "knowledge_date"])

    window = f"{int(window_days)}D"
    rolling = (
        work.groupby("permaticker")
        .rolling(window, on="knowledge_date")["insider_net_dollar"]
        .sum()
        .reset_index()
        .rename(columns={"insider_net_dollar": "ins_90"})
    )
    return rolling[["permaticker", "knowledge_date", "ins_90"]]


def _compute_insider_scores_at_decisions(
    decision_index: pd.DataFrame,
    insiders_agg: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    base = decision_index[["permaticker", "asof_date"]].copy()
    base["permaticker"] = pd.to_numeric(base["permaticker"], errors="coerce").astype("Int64")
    base["asof_date"] = pd.to_datetime(base["asof_date"], errors="coerce").dt.normalize()
    base = (
        base.dropna(subset=["permaticker", "asof_date"])
        .drop_duplicates(["permaticker", "asof_date"], keep="last")
        .reset_index(drop=True)
    )
    if base.empty:
        return pd.DataFrame(columns=["permaticker", "asof_date", "ins_90", "ins_90_knowledge_date"])

    base["_rowid"] = np.arange(len(base), dtype="int64")
    if insiders_agg.empty:
        out = base[["permaticker", "asof_date"]].copy()
        out["ins_90"] = 0.0
        out["ins_90_knowledge_date"] = pd.NaT
        return out

    work = insiders_agg[["permaticker", "knowledge_date", "insider_net_dollar"]].copy()
    work["permaticker"] = pd.to_numeric(work["permaticker"], errors="coerce").astype("Int64")
    work["knowledge_date"] = pd.to_datetime(work["knowledge_date"], errors="coerce").dt.normalize()
    work["insider_net_dollar"] = pd.to_numeric(work["insider_net_dollar"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["permaticker", "knowledge_date"])
    if work.empty:
        out = base[["permaticker", "asof_date"]].copy()
        out["ins_90"] = 0.0
        out["ins_90_knowledge_date"] = pd.NaT
        return out

    # Rolling window as-of decision date:
    # ins_90(d) = cum(d) - cum(d - window_days), where cum(x) is cumulative insider flow up to x.
    events = (
        work.groupby(["permaticker", "knowledge_date"], as_index=False)["insider_net_dollar"]
        .sum()
        .sort_values(["permaticker", "knowledge_date"])
        .reset_index(drop=True)
    )
    events["ins_cum"] = events.groupby("permaticker")["insider_net_dollar"].cumsum()
    event_cum = events[["permaticker", "knowledge_date", "ins_cum"]].copy()

    end_join = deterministic_asof_join(
        base[["permaticker", "asof_date", "_rowid"]],
        event_cum,
        lastupdated_col=None,
        enforce_lastupdated_cutoff=False,
        prefix="insend",
    )

    wd = int(max(1, window_days))
    start_idx = base[["permaticker", "asof_date", "_rowid"]].copy()
    start_idx["asof_date"] = start_idx["asof_date"] - pd.Timedelta(days=wd)
    start_join = deterministic_asof_join(
        start_idx,
        event_cum,
        lastupdated_col=None,
        enforce_lastupdated_cutoff=False,
        prefix="insstart",
    )

    out = (
        base[["permaticker", "asof_date", "_rowid"]]
        .merge(
            end_join[["_rowid", "insend_ins_cum", "insend_knowledge_date"]],
            on="_rowid",
            how="left",
        )
        .merge(
            start_join[["_rowid", "insstart_ins_cum"]],
            on="_rowid",
            how="left",
        )
    )
    end_cum = pd.to_numeric(out["insend_ins_cum"], errors="coerce").fillna(0.0)
    start_cum = pd.to_numeric(out["insstart_ins_cum"], errors="coerce").fillna(0.0)
    out["ins_90"] = end_cum - start_cum
    out["ins_90_knowledge_date"] = pd.to_datetime(out["insend_knowledge_date"], errors="coerce").dt.normalize()
    return out[["permaticker", "asof_date", "ins_90", "ins_90_knowledge_date"]]


def _compute_institution_change_scores(inst_agg: pd.DataFrame) -> pd.DataFrame:
    if inst_agg.empty:
        return pd.DataFrame(columns=["permaticker", "knowledge_date", "inst_chg"])
    work = inst_agg[["permaticker", "knowledge_date", "institution_total_shares"]].copy()
    work["knowledge_date"] = pd.to_datetime(work["knowledge_date"], errors="coerce").dt.normalize()
    work["institution_total_shares"] = pd.to_numeric(work["institution_total_shares"], errors="coerce")
    work = work.dropna(subset=["permaticker", "knowledge_date"]).sort_values(["permaticker", "knowledge_date"])
    work["prev"] = work.groupby("permaticker")["institution_total_shares"].shift(1)
    denom_ok = work["prev"].abs() >= 1.0
    work["inst_chg"] = np.where(
        denom_ok & work["institution_total_shares"].notna(),
        (work["institution_total_shares"] - work["prev"]) / work["prev"].abs(),
        np.nan,
    )
    return work[["permaticker", "knowledge_date", "inst_chg"]]


def _asof_attach(
    decision_index: pd.DataFrame,
    events: pd.DataFrame,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    if decision_index.empty:
        return decision_index.assign(**{out_col: np.nan, f"{out_col}_knowledge_date": pd.NaT})
    if events.empty:
        return decision_index.assign(**{out_col: np.nan, f"{out_col}_knowledge_date": pd.NaT})
    joined = deterministic_asof_join(
        decision_index[["permaticker", "asof_date"]],
        events[["permaticker", "knowledge_date", value_col]],
        lastupdated_col=None,
        enforce_lastupdated_cutoff=False,
        prefix="evt",
    )
    value_name = f"evt_{value_col}"
    lookup = joined[
        ["permaticker", "asof_date", value_name, "evt_knowledge_date"]
    ].drop_duplicates(["permaticker", "asof_date"], keep="last")
    out = decision_index.merge(lookup, on=["permaticker", "asof_date"], how="left")
    out = out.rename(
        columns={
            value_name: out_col,
            "evt_knowledge_date": f"{out_col}_knowledge_date",
        }
    )
    out[f"{out_col}_knowledge_date"] = pd.to_datetime(out[f"{out_col}_knowledge_date"], errors="coerce").dt.normalize()
    return out


def _build_decision_features(
    config: Stage2Config,
    decision_dates: pd.DatetimeIndex,
    universe: pd.DataFrame,
    equity_adj: pd.DataFrame,
    insiders_agg: pd.DataFrame,
    institutions_agg: pd.DataFrame,
) -> pd.DataFrame:
    decision_set = set(pd.to_datetime(decision_dates).normalize())
    uni = universe.copy()
    uni["asof_date"] = pd.to_datetime(uni["asof_date"], errors="coerce").dt.normalize()
    if "is_etf_allowlist" in uni.columns:
        is_etf_allowlist = uni["is_etf_allowlist"].astype(bool)
    else:
        is_etf_allowlist = pd.Series(False, index=uni.index)
    eligible = uni[
        (uni["eligible_flag"].astype(bool))
        & (~is_etf_allowlist)
        & (uni["asof_date"].isin(decision_set))
    ][["asof_date", "permaticker", "ticker", "adv20_dollar"]].copy()
    eligible["permaticker"] = pd.to_numeric(eligible["permaticker"], errors="coerce").astype("Int64")
    eligible = eligible.dropna(subset=["asof_date", "permaticker"]).drop_duplicates(["asof_date", "permaticker"], keep="last")
    if eligible.empty:
        raise Stage2DataError("No eligible decision-date universe rows found for stage 2")

    price_features = _compute_price_features(equity_adj)
    price_features = price_features.rename(columns={"date": "asof_date"})
    decision = eligible.merge(
        price_features[["asof_date", "permaticker", "mom_126x21", "mom_252x21", "rev_21"]],
        on=["asof_date", "permaticker"],
        how="left",
    )

    ins_scores = _compute_insider_scores_at_decisions(
        decision[["permaticker", "asof_date"]],
        insiders_agg,
        window_days=config.strategies.insider_window_calendar_days,
    )
    decision = decision.merge(ins_scores, on=["permaticker", "asof_date"], how="left")

    inst_events = _compute_institution_change_scores(institutions_agg)
    decision = _asof_attach(decision, inst_events, value_col="inst_chg", out_col="inst_chg")

    decision = decision.sort_values(["asof_date", "permaticker"]).reset_index(drop=True)
    return decision


def _winsorize(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    if s.notna().sum() < 5:
        return s
    low = s.quantile(lower_q)
    high = s.quantile(upper_q)
    return s.clip(lower=low, upper=high)


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sigma = s.std(skipna=True, ddof=0)
    if pd.isna(sigma) or sigma <= 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def _score_day(strategy_id: str, day: pd.DataFrame, config: Stage2Config) -> pd.DataFrame:
    out = day[["permaticker", "ticker", "adv20_dollar", "mom_126x21", "mom_252x21", "rev_21", "ins_90", "inst_chg"]].copy()
    out["ticker"] = out["ticker"].astype("string").str.upper()

    if strategy_id == "MOM_126X21":
        out["score"] = out["mom_126x21"]
    elif strategy_id == "MOM_252X21":
        out["score"] = out["mom_252x21"]
    elif strategy_id == "REV_21":
        out["score"] = out["rev_21"]
    elif strategy_id == "INS_90":
        out["score"] = out["ins_90"]
    elif strategy_id == "INST_CHG":
        out["score"] = out["inst_chg"]
    elif strategy_id in {"COMBO_RAW", "COMBO_OVERLAY"}:
        w = config.strategies.combo_weights
        mom = _winsorize(out["mom_126x21"], config.strategies.combo_winsor_lower_q, config.strategies.combo_winsor_upper_q)
        ins = _winsorize(out["ins_90"], config.strategies.combo_winsor_lower_q, config.strategies.combo_winsor_upper_q)
        inst = _winsorize(out["inst_chg"], config.strategies.combo_winsor_lower_q, config.strategies.combo_winsor_upper_q)
        score = (
            _zscore(mom).fillna(0.0) * w["momentum"]
            + _zscore(ins).fillna(0.0) * w["insiders"]
            + _zscore(inst).fillna(0.0) * w["institutions"]
        )
        out["score"] = score
    else:
        raise Stage2DataError(f"Unsupported strategy score for {strategy_id}")

    out = out.dropna(subset=["score", "permaticker"]).copy()
    out = out.sort_values(["score", "ticker", "permaticker"], ascending=[False, True, True], kind="mergesort")
    return out.reset_index(drop=True)


def _apply_turnover_throttle(
    current_equities: set[int],
    ranked: list[int],
    target_n: int,
    max_name_changes: int,
    is_initial_rebalance: bool,
    exempt_initial: bool,
) -> tuple[list[int], int]:
    desired = ranked[:target_n]
    desired_set = set(desired)
    if is_initial_rebalance and exempt_initial:
        return desired, len(desired_set)

    current = set(current_equities)
    final_set = set(current)
    budget = max(0, int(max_name_changes))

    rank = {p: i for i, p in enumerate(ranked)}
    removable = sorted(
        [p for p in current if p not in desired_set],
        key=lambda p: rank.get(p, 10**9),
        reverse=True,
    )
    addable = [p for p in desired if p not in current]

    swaps = min(len(removable), len(addable), budget // 2)
    for i in range(swaps):
        final_set.discard(removable[i])
        final_set.add(addable[i])
    budget -= swaps * 2

    if len(final_set) > target_n and budget > 0:
        excess = len(final_set) - target_n
        sell_list = sorted(final_set, key=lambda p: rank.get(p, 10**9), reverse=True)
        for p in sell_list[: min(excess, budget)]:
            final_set.discard(p)
        budget -= min(excess, budget)

    if len(final_set) < target_n and budget > 0:
        for p in desired:
            if p in final_set:
                continue
            final_set.add(p)
            budget -= 1
            if len(final_set) >= target_n or budget <= 0:
                break

    final_ranked = [p for p in desired if p in final_set]
    if len(final_ranked) < len(final_set):
        extras = sorted([p for p in final_set if p not in set(final_ranked)], key=lambda p: rank.get(p, 10**9))
        final_ranked.extend(extras)

    name_changes = len(current - set(final_ranked)) + len(set(final_ranked) - current)
    return final_ranked[:target_n], name_changes


def _build_risk_off_flags(
    etf_adj: pd.DataFrame,
    spy_permaticker: int,
    decision_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, bool]:
    spy = etf_adj[pd.to_numeric(etf_adj["permaticker"], errors="coerce") == int(spy_permaticker)].copy()
    if spy.empty:
        return {pd.Timestamp(d).normalize(): False for d in decision_dates}
    spy["date"] = pd.to_datetime(spy["date"], errors="coerce").dt.normalize()
    spy = spy.sort_values("date")
    spy["close_adj"] = pd.to_numeric(spy["close_adj"], errors="coerce")
    if "realized_vol20" in spy.columns:
        spy["realized_vol20"] = pd.to_numeric(spy["realized_vol20"], errors="coerce")
    else:
        spy["realized_vol20"] = spy["close_adj"].pct_change().rolling(20, min_periods=20).std()
    spy["ma200"] = spy["close_adj"].rolling(200, min_periods=200).mean()
    spy["vol_median_1y"] = spy["realized_vol20"].rolling(252, min_periods=126).median()
    spy["risk_off"] = (spy["close_adj"] < spy["ma200"]) & (spy["realized_vol20"] > spy["vol_median_1y"])

    lookup = spy.set_index("date")["risk_off"].sort_index()
    flags: dict[pd.Timestamp, bool] = {}
    for d in decision_dates:
        date = pd.Timestamp(d).normalize()
        sub = lookup.loc[:date]
        flags[date] = bool(sub.iloc[-1]) if len(sub) else False
    return flags


def _target_weights_for_strategy(
    *,
    strategy_id: str,
    day_scores: pd.DataFrame,
    current_equities: set[int],
    is_initial_rebalance: bool,
    risk_off: bool,
    spy_permaticker: int,
    defensive_permaticker: int,
    config: Stage2Config,
) -> tuple[dict[int, float], list[int], int]:
    target_n = min(config.portfolio.target_positions, config.portfolio.max_positions)
    ranked = [int(x) for x in day_scores["permaticker"].tolist()]
    selected, name_changes = _apply_turnover_throttle(
        current_equities=current_equities,
        ranked=ranked,
        target_n=target_n,
        max_name_changes=config.portfolio.turnover_max_name_changes,
        is_initial_rebalance=is_initial_rebalance,
        exempt_initial=config.portfolio.turnover_initial_rebalance_exempt,
    )

    if strategy_id == "COMBO_OVERLAY" and config.risk_overlay.enabled_for_combo_overlay and risk_off:
        spy_floor = float(max(0.0, min(1.0, config.risk_overlay.spy_floor_weight)))
        rank = {p: i for i, p in enumerate(ranked)}
        current_ranked = sorted(current_equities, key=lambda p: rank.get(p, 10**9))
        if is_initial_rebalance and config.portfolio.turnover_initial_rebalance_exempt:
            max_sells = len(current_ranked)
        else:
            max_sells = max(0, int(config.portfolio.turnover_max_name_changes))
        keep_count = max(0, len(current_ranked) - max_sells)
        kept_equities = current_ranked[:keep_count]
        risk_name_changes = len(current_ranked) - len(kept_equities)

        weights: dict[int, float] = {}
        if kept_equities and len(current_ranked) > 0:
            equity_sleeve_weight = float(len(kept_equities) / len(current_ranked))
            per_name = min(
                equity_sleeve_weight / len(kept_equities),
                config.portfolio.max_weight_per_name,
            )
            for permaticker in kept_equities:
                weights[int(permaticker)] = per_name

        etf_budget = max(0.0, 1.0 - sum(weights.values()))
        if spy_permaticker == defensive_permaticker:
            weights[spy_permaticker] = weights.get(spy_permaticker, 0.0) + etf_budget
        else:
            spy_weight = min(etf_budget, spy_floor)
            defensive_weight = max(0.0, etf_budget - spy_weight)
            weights[spy_permaticker] = weights.get(spy_permaticker, 0.0) + spy_weight
            if defensive_weight > 0:
                weights[defensive_permaticker] = weights.get(defensive_permaticker, 0.0) + defensive_weight
        return weights, kept_equities, risk_name_changes

    weights: dict[int, float] = {}
    if selected:
        per_name = min(1.0 / len(selected), config.portfolio.max_weight_per_name)
        for permaticker in selected:
            weights[int(permaticker)] = per_name

    allocated = sum(weights.values())
    if allocated < 1.0:
        weights[spy_permaticker] = weights.get(spy_permaticker, 0.0) + (1.0 - allocated)

    return weights, selected, name_changes


def _compute_metrics(nav: pd.DataFrame, trades: pd.DataFrame, holdings: pd.DataFrame) -> dict[str, object]:
    if nav.empty:
        return {
            "rows": 0,
            "start_nav": np.nan,
            "end_nav": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "annual_turnover": np.nan,
            "avg_weekly_turnover": np.nan,
            "avg_positions": np.nan,
            "avg_equity_weight": np.nan,
            "avg_etf_weight": np.nan,
            "avg_cash_weight": np.nan,
            "total_cost": 0.0,
            "cost_bps_per_year": np.nan,
        }

    frame = nav.sort_values("date").reset_index(drop=True).copy()
    frame["ret"] = frame["nav"].pct_change().fillna(0.0)
    n = len(frame)
    years = max(n / 252.0, 1.0 / 252.0)
    start_nav = float(frame["nav"].iloc[0])
    end_nav = float(frame["nav"].iloc[-1])
    cagr = (end_nav / start_nav) ** (1.0 / years) - 1.0 if start_nav > 0 else np.nan
    ann_vol = float(frame["ret"].std(ddof=0) * math.sqrt(252.0))
    sharpe = float(frame["ret"].mean() * 252.0 / ann_vol) if ann_vol > 0 else np.nan
    downside = frame.loc[frame["ret"] < 0, "ret"]
    downside_vol = float(downside.std(ddof=0) * math.sqrt(252.0)) if len(downside) else 0.0
    sortino = float(frame["ret"].mean() * 252.0 / downside_vol) if downside_vol > 0 else np.nan
    drawdown = frame["nav"] / frame["nav"].cummax() - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan

    turnover_year = np.nan
    avg_weekly_turnover_all = 0.0 if n > 0 else np.nan
    avg_weekly_turnover_trade_weeks = np.nan
    total_cost = float(pd.to_numeric(trades.get("cost", 0.0), errors="coerce").fillna(0.0).sum()) if not trades.empty else 0.0
    if not trades.empty:
        t = trades.copy()
        t["execution_date"] = pd.to_datetime(t["execution_date"], errors="coerce").dt.normalize()
        t["executed_notional_abs"] = pd.to_numeric(t["executed_notional_abs"], errors="coerce").fillna(0.0)
        total_notional = float(t["executed_notional_abs"].sum())
        avg_nav = float(frame["nav"].mean()) if len(frame) else np.nan
        if avg_nav and avg_nav > 0:
            turnover_year = total_notional / avg_nav / years
        weekly = t.groupby("execution_date", as_index=False)["executed_notional_abs"].sum()
        nav_lookup = frame.set_index("date")["nav"]
        weekly["nav"] = weekly["execution_date"].map(nav_lookup)
        weekly = weekly[weekly["nav"] > 0].copy()
        if not weekly.empty:
            avg_weekly_turnover_trade_weeks = float((weekly["executed_notional_abs"] / weekly["nav"]).mean())

        week_ref = frame[["date", "nav"]].copy()
        iso_ref = week_ref["date"].dt.isocalendar()
        week_ref["iso_year"] = iso_ref["year"].astype("int64")
        week_ref["iso_week"] = iso_ref["week"].astype("int64")
        week_panel = (
            week_ref.groupby(["iso_year", "iso_week"], as_index=False)
            .agg(week_nav=("nav", "last"))
        )

        weekly_iso = t.copy()
        iso_trade = weekly_iso["execution_date"].dt.isocalendar()
        weekly_iso["iso_year"] = iso_trade["year"].astype("int64")
        weekly_iso["iso_week"] = iso_trade["week"].astype("int64")
        weekly_iso = (
            weekly_iso.groupby(["iso_year", "iso_week"], as_index=False)["executed_notional_abs"]
            .sum()
            .rename(columns={"executed_notional_abs": "week_executed_notional_abs"})
        )

        week_panel = week_panel.merge(weekly_iso, on=["iso_year", "iso_week"], how="left")
        week_panel["week_executed_notional_abs"] = pd.to_numeric(
            week_panel["week_executed_notional_abs"], errors="coerce"
        ).fillna(0.0)
        week_panel = week_panel[week_panel["week_nav"] > 0].copy()
        if not week_panel.empty:
            avg_weekly_turnover_all = float((week_panel["week_executed_notional_abs"] / week_panel["week_nav"]).mean())

    avg_positions = np.nan
    if not holdings.empty:
        h = holdings.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
        pos = h.groupby("date")["permaticker"].nunique()
        if len(pos):
            avg_positions = float(pos.mean())

    nav_nonzero = frame[frame["nav"] > 0].copy()
    if nav_nonzero.empty:
        avg_eq_w = np.nan
        avg_etf_w = np.nan
        avg_cash_w = np.nan
    else:
        avg_eq_w = float((nav_nonzero["equity_value"] / nav_nonzero["nav"]).mean())
        avg_etf_w = float((nav_nonzero["etf_value"] / nav_nonzero["nav"]).mean())
        avg_cash_w = float((nav_nonzero["cash"] / nav_nonzero["nav"]).mean())

    avg_nav = float(frame["nav"].mean()) if len(frame) else np.nan
    cost_bps_per_year = float((total_cost / avg_nav) * 10000.0 / years) if avg_nav and avg_nav > 0 else np.nan

    return {
        "rows": int(n),
        "start_nav": start_nav,
        "end_nav": end_nav,
        "cagr": float(cagr) if pd.notna(cagr) else np.nan,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "annual_turnover": float(turnover_year) if pd.notna(turnover_year) else np.nan,
        "avg_weekly_turnover": float(avg_weekly_turnover_all) if pd.notna(avg_weekly_turnover_all) else np.nan,
        "avg_weekly_turnover_on_trade_weeks": (
            float(avg_weekly_turnover_trade_weeks) if pd.notna(avg_weekly_turnover_trade_weeks) else np.nan
        ),
        "avg_positions": float(avg_positions) if pd.notna(avg_positions) else np.nan,
        "avg_equity_weight": avg_eq_w,
        "avg_etf_weight": avg_etf_w,
        "avg_cash_weight": avg_cash_w,
        "total_cost": total_cost,
        "cost_bps_per_year": float(cost_bps_per_year) if pd.notna(cost_bps_per_year) else np.nan,
    }


def _benchmark_paths(config: Stage2Config, symbol: str, cost_scenario: str) -> tuple[Path, Path]:
    bench_id = f"BENCHMARK_{symbol.upper()}"
    base = config.storage.output_root / bench_id / cost_scenario.lower()
    return base / "summary.json", base / "nav_daily.parquet"


def _compute_relative_performance(strategy_nav: pd.DataFrame, benchmark_nav: pd.DataFrame) -> dict[str, object]:
    s = strategy_nav[["date", "nav"]].copy()
    b = benchmark_nav[["date", "nav"]].copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
    s["nav"] = pd.to_numeric(s["nav"], errors="coerce")
    b["nav"] = pd.to_numeric(b["nav"], errors="coerce")
    joined = s.merge(b, on="date", how="inner", suffixes=("_strategy", "_benchmark")).dropna()
    if len(joined) < 2:
        return {
            "aligned_rows": int(len(joined)),
            "ann_excess_return": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "beta": np.nan,
            "correlation": np.nan,
        }

    joined = joined.sort_values("date").reset_index(drop=True)
    ret_s = joined["nav_strategy"].pct_change().fillna(0.0)
    ret_b = joined["nav_benchmark"].pct_change().fillna(0.0)
    excess = ret_s - ret_b
    ann_excess_return = float(excess.mean() * 252.0)
    tracking_error = float(excess.std(ddof=0) * math.sqrt(252.0))
    information_ratio = float(ann_excess_return / tracking_error) if tracking_error > 0 else np.nan
    var_b = float(ret_b.var(ddof=0))
    cov_sb = float(np.cov(ret_s, ret_b, ddof=0)[0, 1]) if len(ret_s) > 1 else np.nan
    beta = float(cov_sb / var_b) if var_b > 0 and pd.notna(cov_sb) else np.nan
    corr = float(ret_s.corr(ret_b)) if len(ret_s) > 1 else np.nan
    return {
        "aligned_rows": int(len(joined)),
        "ann_excess_return": ann_excess_return,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "correlation": corr,
    }


def _attach_benchmark_comparisons(
    config: Stage2Config,
    *,
    strategy_nav: pd.DataFrame,
    strategy_summary: dict[str, object],
    cost_scenario: str,
) -> dict[str, object]:
    out: dict[str, object] = {}
    periods = list(strategy_summary.get("metrics", {}).get("periods", {}).keys())

    for symbol in config.benchmarks.symbols:
        summary_path, nav_path = _benchmark_paths(config, symbol, cost_scenario)
        if not summary_path.exists() or not nav_path.exists():
            if config.validation.autorun_missing_benchmarks:
                run_benchmark_backtest(config, benchmark_ticker=symbol, cost_scenario=cost_scenario)
            else:
                out[symbol] = {
                    "available": False,
                    "reason": "benchmark_artifacts_missing",
                    "summary_path": str(summary_path),
                    "nav_path": str(nav_path),
                }
                continue

        if not summary_path.exists() or not nav_path.exists():
            out[symbol] = {
                "available": False,
                "reason": "benchmark_artifacts_missing_after_autorun",
                "summary_path": str(summary_path),
                "nav_path": str(nav_path),
            }
            continue

        bench_summary = read_json(summary_path)
        try:
            bench_nav = read_parquet(nav_path, columns=["date", "nav", "period"])
        except Exception:
            bench_nav = read_parquet(nav_path, columns=["date", "nav"])
            split_raw = strategy_summary.get("period_split", {})
            try:
                split = PeriodSplit(
                    development_start=pd.Timestamp(split_raw["development_start"]),
                    development_end=pd.Timestamp(split_raw["development_end"]),
                    holdout_start=pd.Timestamp(split_raw["holdout_start"]),
                    holdout_end=pd.Timestamp(split_raw["holdout_end"]),
                    forward_start=pd.Timestamp(split_raw["forward_start"]) if split_raw.get("forward_start") else None,
                    forward_end=pd.Timestamp(split_raw["forward_end"]) if split_raw.get("forward_end") else None,
                )
                bench_nav["period"] = _label_period(
                    bench_nav["date"],
                    split,
                    include_forward=bool(split_raw.get("forward_start")),
                )
            except Exception:
                bench_nav["period"] = "development"
        rel_overall = _compute_relative_performance(
            strategy_nav[strategy_nav["period"] != "excluded"].copy(),
            bench_nav[bench_nav["period"] != "excluded"].copy(),
        )
        period_rel: dict[str, object] = {}
        for period in periods:
            period_rel[period] = _compute_relative_performance(
                strategy_nav[strategy_nav["period"] == period].copy(),
                bench_nav[bench_nav["period"] == period].copy(),
            )

        strategy_overall = strategy_summary["metrics"]["overall"]
        bench_overall = bench_summary.get("metrics", {}).get("overall", {})
        strategy_periods = strategy_summary.get("metrics", {}).get("periods", {})
        bench_periods = bench_summary.get("metrics", {}).get("periods", {})
        period_spread: dict[str, object] = {}
        for period in periods:
            s_m = strategy_periods.get(period, {})
            b_m = bench_periods.get(period, {})
            period_spread[period] = {
                "cagr_spread": (
                    float(s_m["cagr"] - b_m["cagr"])
                    if ("cagr" in s_m and "cagr" in b_m and pd.notna(s_m["cagr"]) and pd.notna(b_m["cagr"]))
                    else np.nan
                ),
                "sharpe_spread": (
                    float(s_m["sharpe"] - b_m["sharpe"])
                    if ("sharpe" in s_m and "sharpe" in b_m and pd.notna(s_m["sharpe"]) and pd.notna(b_m["sharpe"]))
                    else np.nan
                ),
                "max_drawdown_spread": (
                    float(s_m["max_drawdown"] - b_m["max_drawdown"])
                    if (
                        "max_drawdown" in s_m
                        and "max_drawdown" in b_m
                        and pd.notna(s_m["max_drawdown"])
                        and pd.notna(b_m["max_drawdown"])
                    )
                    else np.nan
                ),
            }

        out[symbol] = {
            "available": True,
            "benchmark_strategy": bench_summary.get("strategy"),
            "summary_path": str(summary_path),
            "nav_path": str(nav_path),
            "overall_spread": {
                "cagr_spread": (
                    float(strategy_overall["cagr"] - bench_overall["cagr"])
                    if ("cagr" in strategy_overall and "cagr" in bench_overall and pd.notna(strategy_overall["cagr"]) and pd.notna(bench_overall["cagr"]))
                    else np.nan
                ),
                "sharpe_spread": (
                    float(strategy_overall["sharpe"] - bench_overall["sharpe"])
                    if (
                        "sharpe" in strategy_overall
                        and "sharpe" in bench_overall
                        and pd.notna(strategy_overall["sharpe"])
                        and pd.notna(bench_overall["sharpe"])
                    )
                    else np.nan
                ),
                "max_drawdown_spread": (
                    float(strategy_overall["max_drawdown"] - bench_overall["max_drawdown"])
                    if (
                        "max_drawdown" in strategy_overall
                        and "max_drawdown" in bench_overall
                        and pd.notna(strategy_overall["max_drawdown"])
                        and pd.notna(bench_overall["max_drawdown"])
                    )
                    else np.nan
                ),
            },
            "relative_performance": rel_overall,
            "period_spread": period_spread,
            "period_relative_performance": period_rel,
        }
    return out


def _attach_cost_scenario_comparison(
    config: Stage2Config,
    *,
    strategy_id: str,
    current_summary: dict[str, object],
    current_cost_scenario: str,
) -> dict[str, object]:
    other = "conservative" if current_cost_scenario.lower() == "base" else "base"
    other_summary_path = config.storage.output_root / strategy_id / other / "summary.json"
    if not other_summary_path.exists():
        return {
            "other_cost_scenario": other,
            "available": False,
            "summary_path": str(other_summary_path),
        }
    other_summary = read_json(other_summary_path)
    cur = current_summary["metrics"]["overall"]
    oth = other_summary.get("metrics", {}).get("overall", {})
    return {
        "other_cost_scenario": other,
        "available": True,
        "summary_path": str(other_summary_path),
        "spread_vs_other": {
            "cagr_spread": (
                float(cur["cagr"] - oth["cagr"])
                if ("cagr" in cur and "cagr" in oth and pd.notna(cur["cagr"]) and pd.notna(oth["cagr"]))
                else np.nan
            ),
            "sharpe_spread": (
                float(cur["sharpe"] - oth["sharpe"])
                if ("sharpe" in cur and "sharpe" in oth and pd.notna(cur["sharpe"]) and pd.notna(oth["sharpe"]))
                else np.nan
            ),
            "total_cost_spread": (
                float(cur["total_cost"] - oth["total_cost"])
                if ("total_cost" in cur and "total_cost" in oth and pd.notna(cur["total_cost"]) and pd.notna(oth["total_cost"]))
                else np.nan
            ),
        },
    }


def _run_rebalance_spot_audit(
    *,
    decision_features: pd.DataFrame,
    rebalances_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    accessor: PriceAccessor,
    sample_size: int,
    seed: int,
) -> dict[str, object]:
    if rebalances_df.empty:
        return {"available": False, "reason": "no_rebalances"}

    work = rebalances_df.copy()
    work["decision_date"] = pd.to_datetime(work["decision_date"], errors="coerce").dt.normalize()
    work["execution_date"] = pd.to_datetime(work["execution_date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["decision_date", "execution_date"]).reset_index(drop=True)
    if work.empty:
        return {"available": False, "reason": "no_valid_rebalances"}

    n = min(max(1, int(sample_size)), len(work))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(work), size=n, replace=False)
    sample = work.iloc[np.sort(idx)].copy().reset_index(drop=True)

    failures: list[dict[str, object]] = []
    checks_total = 0
    checks_pass = 0

    ins_col = "ins_90_knowledge_date"
    inst_col = "inst_chg_knowledge_date"

    for row in sample.itertuples(index=False):
        d = pd.Timestamp(row.decision_date).normalize()
        x = pd.Timestamp(row.execution_date).normalize()

        day_features = decision_features[decision_features["asof_date"] == d].copy()
        ins_ok = True
        inst_ok = True
        if ins_col in day_features.columns:
            ins_dates = pd.to_datetime(day_features[ins_col], errors="coerce").dropna()
            ins_ok = bool((ins_dates <= d).all())
        if inst_col in day_features.columns:
            inst_dates = pd.to_datetime(day_features[inst_col], errors="coerce").dropna()
            inst_ok = bool((inst_dates <= d).all())
        feature_ts_ok = bool(ins_ok and inst_ok)

        if trades_df.empty or "decision_date" not in trades_df.columns:
            trades = pd.DataFrame(columns=["execution_date", "reason", "open_adj", "permaticker"])
        else:
            trades = trades_df[pd.to_datetime(trades_df["decision_date"], errors="coerce").dt.normalize() == d].copy()
        if trades.empty:
            timing_ok = True
            open_ok = True
            close_ok = True
        else:
            t_exec = pd.to_datetime(trades["execution_date"], errors="coerce").dt.normalize()
            timing_ok = bool(((t_exec == x) & (t_exec > d)).all())
            executed = trades[trades["reason"] == "executed"].copy()
            open_ok = bool(executed["open_adj"].notna().all()) if len(executed) else True

            close_ok = True
            for tr in executed.itertuples(index=False):
                p = int(tr.permaticker)
                ex = pd.Timestamp(tr.execution_date).normalize()
                close_px = accessor.close_adj(p, ex)
                if close_px is None or close_px <= 0:
                    close_ok = False
                    break

        checks_total += 4
        checks_pass += int(feature_ts_ok) + int(timing_ok) + int(open_ok) + int(close_ok)
        if not (feature_ts_ok and timing_ok and open_ok and close_ok):
            failures.append(
                {
                    "decision_date": d.date().isoformat(),
                    "execution_date": x.date().isoformat(),
                    "feature_timestamp_ok": feature_ts_ok,
                    "trade_timing_ok": timing_ok,
                    "open_adj_ok": open_ok,
                    "close_adj_ok": close_ok,
                }
            )

    return {
        "available": True,
        "sample_size": int(n),
        "total_rebalances": int(len(work)),
        "all_checks_passed": bool(len(failures) == 0),
        "checks_pass_rate": float(checks_pass / checks_total) if checks_total > 0 else np.nan,
        "failure_count": int(len(failures)),
        "failures": failures[:10],
    }


def _render_report(summary: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append(f"# Backtest Report: {summary['strategy']} ({summary['cost_scenario']})")
    lines.append("")
    lines.append(f"- Generated at UTC: {summary['generated_at_utc']}")
    lines.append(f"- Date range: {summary['date_range']['start']} -> {summary['date_range']['end']}")
    lines.append("")
    lines.append("## Overall")
    overall = summary["metrics"]["overall"]
    lines.extend(
        [
            f"- CAGR: {overall['cagr']:.4f}" if pd.notna(overall["cagr"]) else "- CAGR: nan",
            f"- Ann Vol: {overall['ann_vol']:.4f}" if pd.notna(overall["ann_vol"]) else "- Ann Vol: nan",
            f"- Sharpe: {overall['sharpe']:.4f}" if pd.notna(overall["sharpe"]) else "- Sharpe: nan",
            f"- Max Drawdown: {overall['max_drawdown']:.4f}" if pd.notna(overall["max_drawdown"]) else "- Max Drawdown: nan",
            f"- Annual Turnover: {overall['annual_turnover']:.4f}" if pd.notna(overall["annual_turnover"]) else "- Annual Turnover: nan",
            f"- Total Costs: {overall['total_cost']:.2f}",
        ]
    )
    lines.append("")
    lines.append("## By Period")
    for period_name, payload in summary["metrics"]["periods"].items():
        lines.append(f"### {period_name.title()}")
        lines.append(
            f"- Rows: {payload['rows']}, Start NAV: {payload['start_nav']:.2f}, End NAV: {payload['end_nav']:.2f}"
            if payload["rows"] > 0
            else "- Rows: 0"
        )
        lines.append(f"- CAGR: {payload['cagr']:.4f}" if pd.notna(payload["cagr"]) else "- CAGR: nan")
        lines.append(f"- Sharpe: {payload['sharpe']:.4f}" if pd.notna(payload["sharpe"]) else "- Sharpe: nan")
    lines.append("")

    comparisons = summary.get("comparisons", {})
    bench = comparisons.get("benchmarks", {})
    if bench:
        lines.append("## Benchmark Comparison")
        for symbol, payload in bench.items():
            lines.append(f"### vs {symbol}")
            if not payload.get("available", False):
                lines.append(f"- Available: False ({payload.get('reason', 'unknown')})")
                lines.append("")
                continue
            spread = payload.get("overall_spread", {})
            rel = payload.get("relative_performance", {})
            if "cagr_spread" in spread and pd.notna(spread["cagr_spread"]):
                lines.append(f"- CAGR Spread: {spread['cagr_spread']:.4f}")
            if "sharpe_spread" in spread and pd.notna(spread["sharpe_spread"]):
                lines.append(f"- Sharpe Spread: {spread['sharpe_spread']:.4f}")
            if "information_ratio" in rel and pd.notna(rel["information_ratio"]):
                lines.append(f"- Information Ratio: {rel['information_ratio']:.4f}")
            if "beta" in rel and pd.notna(rel["beta"]):
                lines.append(f"- Beta: {rel['beta']:.4f}")
            lines.append("")

    cost_cmp = comparisons.get("cost_scenario")
    if isinstance(cost_cmp, dict):
        lines.append("## Cost Scenario Comparison")
        lines.append(f"- Other Scenario: {cost_cmp.get('other_cost_scenario')}")
        lines.append(f"- Available: {cost_cmp.get('available')}")
        spread = cost_cmp.get("spread_vs_other", {})
        if isinstance(spread, dict):
            if "cagr_spread" in spread and pd.notna(spread["cagr_spread"]):
                lines.append(f"- CAGR Spread: {spread['cagr_spread']:.4f}")
            if "sharpe_spread" in spread and pd.notna(spread["sharpe_spread"]):
                lines.append(f"- Sharpe Spread: {spread['sharpe_spread']:.4f}")
            if "total_cost_spread" in spread and pd.notna(spread["total_cost_spread"]):
                lines.append(f"- Total Cost Spread: {spread['total_cost_spread']:.2f}")
        lines.append("")

    sensitivity = summary.get("sensitivity_checks", {})
    if sensitivity:
        lines.append("## Sensitivity Checks")
        for name, payload in sensitivity.items():
            lines.append(f"### {name}")
            for k, v in payload.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

    audits = summary.get("audits", {})
    if audits:
        lines.append("## Spot Audits")
        for name, payload in audits.items():
            lines.append(f"### {name}")
            for k, v in payload.items():
                if k == "failures":
                    lines.append(f"- failures: {len(v)} shown={min(len(v), 10)}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")

    sanity = summary.get("sanity_flags", {})
    lines.append("## Sanity Flags")
    for k, v in sanity.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def _write_artifacts(
    config: Stage2Config,
    strategy_id: str,
    cost_scenario: str,
    nav_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    rebalances_df: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, str]:
    out_dir = ensure_dir(config.storage.output_root / strategy_id / cost_scenario.lower())

    nav_path = out_dir / "nav_daily.parquet"
    trades_path = out_dir / "trades.parquet"
    holdings_path = out_dir / "holdings.parquet"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    rebalances_path = out_dir / "rebalances.parquet"

    write_parquet(nav_df, nav_path)
    write_parquet(trades_df, trades_path)
    write_parquet(holdings_df, holdings_path)
    write_parquet(rebalances_df, rebalances_path)
    write_json(summary_path, summary)
    report_path.write_text(_render_report(summary) + "\n")

    return {
        "output_dir": str(out_dir),
        "nav_daily": str(nav_path),
        "trades": str(trades_path),
        "holdings": str(holdings_path),
        "rebalances": str(rebalances_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }


def _refresh_counter_cost_comparison(
    config: Stage2Config,
    *,
    strategy_id: str,
    just_run_cost_scenario: str,
) -> None:
    current = just_run_cost_scenario.lower()
    other = "conservative" if current == "base" else "base"
    other_dir = config.storage.output_root / strategy_id / other
    other_summary_path = other_dir / "summary.json"
    if not other_summary_path.exists():
        return

    other_summary = read_json(other_summary_path)
    comparisons = other_summary.get("comparisons")
    if not isinstance(comparisons, dict):
        comparisons = {}
        other_summary["comparisons"] = comparisons

    comparisons["cost_scenario"] = _attach_cost_scenario_comparison(
        config,
        strategy_id=strategy_id,
        current_summary=other_summary,
        current_cost_scenario=other,
    )
    write_json(other_summary_path, other_summary)
    (other_dir / "report.md").write_text(_render_report(other_summary) + "\n")


def _simulate(
    *,
    config: Stage2Config,
    strategy_id: str,
    cost_scenario: str,
    benchmark_ticker: str | None = None,
    execution_delay_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    calendar_full = _load_calendar(config)
    trading_days = _calendar_window(calendar_full, config.period.start_date, config.period.end_date)
    schedule = _build_decision_execution_schedule(trading_days, execution_delay_days=execution_delay_days)
    if schedule.empty:
        raise Stage2DataError("No valid decision/execution schedule generated")

    split = _compute_period_split(trading_days, config.period.holdout_years)

    load_equity_inputs = benchmark_ticker is None
    if load_equity_inputs:
        universe_cols = ["asof_date", "permaticker", "ticker", "eligible_flag", "is_etf_allowlist", "adv20_dollar"]
        try:
            universe = read_parquet(config.storage.clean_root / config.storage.universe, columns=universe_cols)
        except Exception:
            # Backward-compatible fallback for older Stage-1 outputs without explicit ETF allowlist flag.
            universe = read_parquet(
                config.storage.clean_root / config.storage.universe,
                columns=["asof_date", "permaticker", "ticker", "eligible_flag", "adv20_dollar"],
            )
            universe["is_etf_allowlist"] = False
        insiders = read_parquet(
            config.storage.clean_root / config.storage.insiders_agg,
            columns=["permaticker", "knowledge_date", "insider_net_dollar"],
        )
        institutions = read_parquet(
            config.storage.clean_root / config.storage.institutions_agg,
            columns=["permaticker", "knowledge_date", "institution_total_shares"],
        )
    else:
        universe = pd.DataFrame(columns=["asof_date", "permaticker", "ticker", "eligible_flag", "is_etf_allowlist", "adv20_dollar"])
        insiders = pd.DataFrame(columns=["permaticker", "knowledge_date", "insider_net_dollar"])
        institutions = pd.DataFrame(columns=["permaticker", "knowledge_date", "institution_total_shares"])
    master = read_parquet(config.storage.clean_root / config.storage.master_security, columns=["permaticker", "end_date"])
    eq_adj, etf_adj = _load_adjusted_price_tables(config, load_equity=load_equity_inputs)

    for df, date_col in (
        (universe, "asof_date"),
        (insiders, "knowledge_date"),
        (institutions, "knowledge_date"),
        (master, "end_date"),
        (eq_adj, "date"),
        (etf_adj, "date"),
    ):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    end_date_lookup = (
        master[["permaticker", "end_date"]]
        .dropna(subset=["permaticker"])
        .assign(permaticker=lambda x: pd.to_numeric(x["permaticker"], errors="coerce").astype("Int64"))
        .dropna(subset=["permaticker"])
        .sort_values(["permaticker", "end_date"])
        .drop_duplicates(["permaticker"], keep="last")
        .set_index("permaticker")["end_date"]
    )
    accessor = PriceAccessor(eq_adj, etf_adj, end_date_lookup)

    spy_permaticker = accessor.permaticker_for_ticker("SPY")
    defensive_permaticker = accessor.permaticker_for_ticker(config.risk_overlay.defensive_ticker)

    decision_dates = pd.DatetimeIndex(schedule["decision_date"])
    decision_date_set = set(pd.DatetimeIndex(schedule["decision_date"]).normalize())
    execution_by_decision = {
        pd.Timestamp(row.decision_date).normalize(): pd.Timestamp(row.execution_date).normalize()
        for row in schedule.itertuples(index=False)
    }

    decision_features = pd.DataFrame()
    risk_flags = {d: False for d in decision_dates}
    if benchmark_ticker is None:
        decision_features = _build_decision_features(
            config=config,
            decision_dates=decision_dates,
            universe=universe,
            equity_adj=eq_adj,
            insiders_agg=insiders,
            institutions_agg=institutions,
        )
        risk_flags = _build_risk_off_flags(etf_adj, spy_permaticker, decision_dates)

    universe_adv = universe[["asof_date", "permaticker", "adv20_dollar"]].copy()
    universe_adv["asof_date"] = pd.to_datetime(universe_adv["asof_date"], errors="coerce").dt.normalize()
    universe_adv["permaticker"] = pd.to_numeric(universe_adv["permaticker"], errors="coerce").astype("Int64")
    universe_adv["adv20_dollar"] = pd.to_numeric(universe_adv["adv20_dollar"], errors="coerce")
    universe_adv = universe_adv.dropna(subset=["asof_date", "permaticker"]).drop_duplicates(["asof_date", "permaticker"], keep="last")
    adv_lookup = universe_adv.set_index(["asof_date", "permaticker"])["adv20_dollar"]

    holdings: dict[int, float] = {}
    last_close: dict[int, float] = {}
    stale_days: dict[int, int] = {}
    cash = float(config.portfolio.initial_capital)
    cumulative_cost = 0.0
    pending: dict[pd.Timestamp, PendingRebalance] = {}
    is_first_rebalance = True
    benchmark_entered = False
    benchmark_permaticker: int | None = accessor.permaticker_for_ticker(benchmark_ticker) if benchmark_ticker else None

    nav_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []

    for day in trading_days:
        day = pd.Timestamp(day).normalize()

        if day in pending:
            reb = pending.pop(day)
            assets = sorted(set(holdings.keys()) | set(reb.target_weights.keys()))
            requested: dict[int, float] = {}
            for asset in assets:
                open_px = accessor.open_adj(asset, day)
                target_weight = float(reb.target_weights.get(asset, 0.0))
                current_shares = float(holdings.get(asset, 0.0))
                if open_px is None or open_px <= 0:
                    requested[asset] = 0.0
                    trade_rows.append(
                        {
                            "decision_date": reb.decision_date,
                            "execution_date": day,
                            "permaticker": int(asset),
                            "ticker": accessor.ticker(asset, day),
                            "asset_class": accessor.asset_class(asset),
                            "requested_shares": float(np.nan),
                            "executed_shares": 0.0,
                            "open_adj": np.nan,
                            "executed_notional": 0.0,
                            "executed_notional_abs": 0.0,
                            "cost": 0.0,
                            "reason": "missing_open_adj",
                        }
                    )
                    continue
                target_notional = target_weight * reb.reference_nav
                desired_shares = target_notional / open_px
                if config.portfolio.integer_shares:
                    desired_shares = math.floor(max(desired_shares, 0.0))
                delta = desired_shares - current_shares
                requested[asset] = delta

            execution_order = sorted(assets, key=lambda a: (requested.get(a, 0.0) > 0, accessor.ticker(a, day), int(a)))
            for asset in execution_order:
                delta = requested.get(asset, 0.0)
                if abs(delta) <= 0:
                    continue
                open_px = accessor.open_adj(asset, day)
                if open_px is None or open_px <= 0:
                    continue

                requested_notional = abs(delta * open_px)
                if requested_notional < config.portfolio.min_trade_notional:
                    continue

                is_etf = accessor.asset_class(asset) == "etf"
                decision_adv = reb.decision_adv.get(asset)
                if decision_adv is None or decision_adv <= 0:
                    if reb.strategy_id.startswith("BENCHMARK_") or is_etf:
                        # Benchmarks are specified as enter-and-hold on first execution open.
                        # Do not block benchmark/ETF sleeve entry on missing ADV20.
                        decision_adv = float("inf")
                    else:
                        trade_rows.append(
                            {
                                "decision_date": reb.decision_date,
                                "execution_date": day,
                                "permaticker": int(asset),
                                "ticker": accessor.ticker(asset, day),
                                "asset_class": accessor.asset_class(asset),
                                "requested_shares": float(delta),
                                "executed_shares": 0.0,
                                "open_adj": float(open_px),
                                "executed_notional": 0.0,
                                "executed_notional_abs": 0.0,
                                "cost": 0.0,
                                "reason": "missing_decision_adv20",
                            }
                        )
                        continue

                decision_adv_value = float(decision_adv)
                if math.isinf(decision_adv_value):
                    max_shares = math.ceil(abs(delta))
                else:
                    max_notional = decision_adv_value * config.execution.liquidity_adv_fraction
                    max_shares = math.floor(max_notional / open_px) if open_px > 0 else 0
                if max_shares <= 0:
                    continue
                exec_delta = math.copysign(min(abs(delta), max_shares), delta)
                if config.portfolio.integer_shares:
                    exec_delta = float(int(exec_delta))
                if abs(exec_delta) <= 0:
                    continue

                bps = resolve_cost_bps(config, cost_scenario, is_etf=is_etf)
                rate = bps / 10000.0
                if exec_delta > 0:
                    affordable = math.floor(max(cash, 0.0) / (open_px * (1.0 + rate)))
                    exec_delta = float(min(exec_delta, affordable))
                    if abs(exec_delta) <= 0:
                        continue

                trade_notional = exec_delta * open_px
                cost = abs(trade_notional) * rate
                cash -= trade_notional + cost
                cumulative_cost += cost
                holdings[asset] = float(holdings.get(asset, 0.0) + exec_delta)
                if abs(holdings[asset]) < 1e-9:
                    holdings.pop(asset, None)
                    last_close.pop(asset, None)
                    stale_days.pop(asset, None)

                trade_rows.append(
                    {
                        "decision_date": reb.decision_date,
                        "execution_date": day,
                        "permaticker": int(asset),
                        "ticker": accessor.ticker(asset, day),
                        "asset_class": accessor.asset_class(asset),
                        "requested_shares": float(delta),
                        "executed_shares": float(exec_delta),
                        "open_adj": float(open_px),
                        "executed_notional": float(trade_notional),
                        "executed_notional_abs": float(abs(trade_notional)),
                        "cost": float(cost),
                        "reason": "executed",
                    }
                )

            if benchmark_permaticker is not None and holdings.get(int(benchmark_permaticker), 0.0) > 0:
                benchmark_entered = True

        equity_value = 0.0
        etf_value = 0.0
        stale_count = 0
        forced_assets: list[int] = []

        for asset, shares in list(holdings.items()):
            if shares == 0:
                forced_assets.append(asset)
                continue
            close_px = accessor.close_adj(asset, day)
            mark_px: float | None = None
            stale = False
            if close_px is None or close_px <= 0:
                prev_px = last_close.get(asset)
                if prev_px is not None and prev_px > 0:
                    mark_px = prev_px
                    stale = True
                    stale_days[asset] = int(stale_days.get(asset, 0) + 1)
                    stale_count += 1
                else:
                    mark_px = None
                    stale_days[asset] = int(stale_days.get(asset, 0) + 1)
            else:
                mark_px = float(close_px)
                last_close[asset] = mark_px
                stale_days[asset] = 0

            end_proxy = accessor.last_known_end_date(asset)
            should_force = (
                shares > 0
                and mark_px is not None
                and (
                    stale_days.get(asset, 0) > config.execution.stale_price_max_days
                    or (end_proxy is not None and day > end_proxy)
                )
            )
            if should_force:
                is_etf = accessor.asset_class(asset) == "etf"
                base_bps = resolve_cost_bps(config, cost_scenario, is_etf=is_etf)
                rate = (base_bps + config.execution.stale_liquidation_penalty_bps) / 10000.0
                liquidation_notional = shares * mark_px
                cost = abs(liquidation_notional) * rate
                cash += liquidation_notional - cost
                cumulative_cost += cost
                trade_rows.append(
                    {
                        "decision_date": pd.NaT,
                        "execution_date": day,
                        "permaticker": int(asset),
                        "ticker": accessor.ticker(asset, day),
                        "asset_class": accessor.asset_class(asset),
                        "requested_shares": float(-shares),
                        "executed_shares": float(-shares),
                        "open_adj": np.nan,
                        "executed_notional": float(-liquidation_notional),
                        "executed_notional_abs": float(abs(liquidation_notional)),
                        "cost": float(cost),
                        "reason": "forced_stale_liquidation",
                    }
                )
                forced_assets.append(asset)
                continue

            if mark_px is None:
                continue
            market_value = shares * mark_px
            if accessor.asset_class(asset) == "etf":
                etf_value += market_value
            else:
                equity_value += market_value

        for asset in forced_assets:
            holdings.pop(asset, None)
            last_close.pop(asset, None)
            stale_days.pop(asset, None)

        nav = cash + equity_value + etf_value
        nav_rows.append(
            {
                "date": day,
                "nav": float(nav),
                "cash": float(cash),
                "equity_value": float(equity_value),
                "etf_value": float(etf_value),
                "cumulative_cost": float(cumulative_cost),
                "stale_holding_count": int(stale_count),
            }
        )

        if nav > 0 and holdings:
            for asset, shares in holdings.items():
                if shares == 0:
                    continue
                close_px = accessor.close_adj(asset, day)
                if close_px is None or close_px <= 0:
                    close_px = last_close.get(asset)
                if close_px is None:
                    continue
                mv = float(shares * close_px)
                holdings_rows.append(
                    {
                        "date": day,
                        "permaticker": int(asset),
                        "ticker": accessor.ticker(asset, day),
                        "asset_class": accessor.asset_class(asset),
                        "shares": float(shares),
                        "close_adj": float(close_px),
                        "market_value": mv,
                        "weight": float(mv / nav) if nav > 0 else np.nan,
                    }
                )

        if day in execution_by_decision:
            # This is an execution date for some other decision; decisions are applied later in the day.
            pass

        if day in decision_date_set:
            exec_day = execution_by_decision.get(day)
            if exec_day is None:
                continue
            if exec_day > trading_days.max():
                continue

            decision_adv: dict[int, float] = {}

            if benchmark_ticker is not None:
                if benchmark_entered:
                    continue
                if benchmark_permaticker is None:
                    raise Stage2DataError(f"Benchmark ticker {benchmark_ticker} could not be resolved to permaticker")
                target_weights = {int(benchmark_permaticker): 1.0}
                selected = []
                name_changes = 0
                risk_off = False
            else:
                day_df = decision_features[decision_features["asof_date"] == day].copy()
                if day_df.empty:
                    day_df = pd.DataFrame(columns=decision_features.columns)
                scores = _score_day(strategy_id, day_df, config)
                current_equities = {
                    int(asset)
                    for asset, shares in holdings.items()
                    if shares > 0 and accessor.asset_class(asset) == "equity"
                }
                risk_off = bool(risk_flags.get(day, False))
                target_weights, selected, name_changes = _target_weights_for_strategy(
                    strategy_id=strategy_id,
                    day_scores=scores,
                    current_equities=current_equities,
                    is_initial_rebalance=is_first_rebalance,
                    risk_off=risk_off,
                    spy_permaticker=spy_permaticker,
                    defensive_permaticker=defensive_permaticker,
                    config=config,
                )
                is_first_rebalance = False

            # Build decision-date ADV lookup for all target and held names.
            candidates = set(target_weights.keys()) | set(holdings.keys())
            for asset in candidates:
                key = (day, int(asset))
                adv_val = adv_lookup.get(key)
                if pd.isna(adv_val):
                    adv_val = accessor.adv20(asset, day)
                if adv_val is not None and not pd.isna(adv_val):
                    decision_adv[int(asset)] = float(adv_val)

            if exec_day in pending:
                raise Stage2DataError(
                    f"Execution-date collision for {exec_day.date()}: multiple decisions map to the same execution day"
                )

            pending[exec_day] = PendingRebalance(
                decision_date=day,
                execution_date=exec_day,
                reference_nav=float(nav),
                target_weights={int(k): float(v) for k, v in target_weights.items() if v > 0},
                decision_adv=decision_adv,
                selected_equities=[int(x) for x in selected] if benchmark_ticker is None else [],
                risk_off=bool(risk_off) if benchmark_ticker is None else False,
                strategy_id=strategy_id,
            )
            rebalance_rows.append(
                {
                    "decision_date": day,
                    "execution_date": exec_day,
                    "strategy": strategy_id,
                    "risk_off": bool(risk_off) if benchmark_ticker is None else False,
                    "selected_equity_count": len(selected) if benchmark_ticker is None else 0,
                    "equity_name_changes": int(name_changes) if benchmark_ticker is None else 0,
                    "target_weight_sum": float(sum(target_weights.values())),
                    "target_weights": ";".join(
                        f"{int(k)}:{float(v):.6f}" for k, v in sorted(target_weights.items(), key=lambda kv: kv[0])
                    ),
                }
            )

    nav_df = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame(trade_rows)
    holdings_df = pd.DataFrame(holdings_rows)
    rebalances_df = pd.DataFrame(rebalance_rows)

    nav_df["period"] = _label_period(nav_df["date"], split, config.period.include_forward_period)
    if not trades_df.empty:
        trades_df["period"] = _label_period(trades_df["execution_date"], split, config.period.include_forward_period)
    else:
        trades_df["period"] = pd.Series(dtype="string")
    if not holdings_df.empty:
        holdings_df["period"] = _label_period(holdings_df["date"], split, config.period.include_forward_period)
    else:
        holdings_df["period"] = pd.Series(dtype="string")

    periods = ["development", "holdout"]
    if config.period.include_forward_period:
        periods.append("forward")

    period_metrics: dict[str, dict[str, object]] = {}
    for period in periods:
        nav_sub = nav_df[nav_df["period"] == period].copy()
        trades_sub = trades_df[trades_df["period"] == period].copy() if not trades_df.empty else pd.DataFrame()
        holdings_sub = holdings_df[holdings_df["period"] == period].copy() if not holdings_df.empty else pd.DataFrame()
        period_metrics[period] = _compute_metrics(nav_sub, trades_sub, holdings_sub)

    overall_metrics = _compute_metrics(
        nav_df[nav_df["period"] != "excluded"].copy(),
        trades_df[trades_df["period"] != "excluded"].copy() if not trades_df.empty else pd.DataFrame(),
        holdings_df[holdings_df["period"] != "excluded"].copy() if not holdings_df.empty else pd.DataFrame(),
    )
    if trades_df.empty:
        executed_missing_open = 0
        executed_bad_timing = 0
        forced_liquidations = 0
        blocked_missing_open = 0
        blocked_missing_adv = 0
    else:
        executed = trades_df[trades_df["reason"] == "executed"].copy()
        executed_missing_open = int(executed["open_adj"].isna().sum()) if "open_adj" in executed.columns else 0

        if len(executed):
            executed_dates = pd.to_datetime(executed["execution_date"], errors="coerce").dt.normalize()
            decision_dates_executed = pd.to_datetime(executed["decision_date"], errors="coerce").dt.normalize()
            bad_mask = decision_dates_executed.notna() & (executed_dates <= decision_dates_executed)
            executed_bad_timing = int(bad_mask.sum())
        else:
            executed_bad_timing = 0

        forced_liquidations = int((trades_df["reason"] == "forced_stale_liquidation").sum())
        blocked_missing_open = int((trades_df["reason"] == "missing_open_adj").sum())
        blocked_missing_adv = int((trades_df["reason"] == "missing_decision_adv20").sum())

    sanity_flags = {
        "extreme_sharpe_over_3": bool(
            benchmark_ticker is None and pd.notna(overall_metrics["sharpe"]) and overall_metrics["sharpe"] > 3.0
        ),
        "executed_trades_with_missing_open_adj": executed_missing_open,
        "executed_trades_with_execution_not_after_decision": executed_bad_timing,
        "forced_stale_liquidations": forced_liquidations,
        "blocked_trades_missing_open_adj": blocked_missing_open,
        "blocked_trades_missing_decision_adv20": blocked_missing_adv,
    }

    audits: dict[str, object] = {}
    if benchmark_ticker is None:
        audits["rebalance_spot_audit"] = _run_rebalance_spot_audit(
            decision_features=decision_features,
            rebalances_df=rebalances_df,
            trades_df=trades_df,
            accessor=accessor,
            sample_size=config.validation.rebalance_audit_sample_size,
            seed=config.validation.rebalance_audit_seed,
        )

    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "strategy": strategy_id,
        "cost_scenario": cost_scenario.lower(),
        "benchmark_ticker": benchmark_ticker,
        "execution_delay_days": int(max(0, execution_delay_days)),
        "date_range": {
            "start": pd.Timestamp(trading_days.min()).date().isoformat(),
            "end": pd.Timestamp(trading_days.max()).date().isoformat(),
        },
        "period_split": split.to_dict(),
        "metrics": {
            "overall": overall_metrics,
            "periods": period_metrics,
        },
        "counts": {
            "nav_rows": int(len(nav_df)),
            "trade_rows": int(len(trades_df)),
            "holding_rows": int(len(holdings_df)),
            "rebalance_rows": int(len(rebalances_df)),
        },
        "sanity_flags": sanity_flags,
        "audits": audits,
    }
    return nav_df, trades_df, holdings_df, rebalances_df, summary


def run_strategy_backtest(config: Stage2Config, strategy: str, cost_scenario: str) -> dict[str, object]:
    strategy_id = _normalize_strategy_id(strategy, config)
    if cost_scenario.lower() not in {"base", "conservative"}:
        raise Stage2DataError("cost_scenario must be base or conservative")
    resolved_cost = cost_scenario.lower()

    nav_df, trades_df, holdings_df, rebalances_df, summary = _simulate(
        config=config,
        strategy_id=strategy_id,
        cost_scenario=resolved_cost,
        benchmark_ticker=None,
    )

    summary["comparisons"] = {
        "benchmarks": _attach_benchmark_comparisons(
            config,
            strategy_nav=nav_df,
            strategy_summary=summary,
            cost_scenario=resolved_cost,
        ),
        "cost_scenario": _attach_cost_scenario_comparison(
            config,
            strategy_id=strategy_id,
            current_summary=summary,
            current_cost_scenario=resolved_cost,
        ),
    }

    sensitivity_checks: dict[str, object] = {}
    if config.validation.run_delay_sanity_check and config.validation.delay_sanity_days > 0:
        delayed_nav_df, delayed_trades_df, _, _, delayed_summary = _simulate(
            config=config,
            strategy_id=strategy_id,
            cost_scenario=resolved_cost,
            benchmark_ticker=None,
            execution_delay_days=config.validation.delay_sanity_days,
        )
        base_overall = summary["metrics"]["overall"]
        delayed_overall = delayed_summary["metrics"]["overall"]
        cagr_base = base_overall.get("cagr")
        cagr_delayed = delayed_overall.get("cagr")
        sharpe_base = base_overall.get("sharpe")
        sharpe_delayed = delayed_overall.get("sharpe")
        cagr_weaker = (
            bool(cagr_delayed <= cagr_base)
            if (pd.notna(cagr_delayed) and pd.notna(cagr_base))
            else None
        )
        sharpe_weaker = (
            bool(sharpe_delayed <= sharpe_base)
            if (pd.notna(sharpe_delayed) and pd.notna(sharpe_base))
            else None
        )

        base_exec = pd.to_datetime(
            trades_df.loc[trades_df.get("reason", pd.Series(dtype="string")) == "executed", "execution_date"],
            errors="coerce",
        ).dt.normalize()
        delayed_exec = pd.to_datetime(
            delayed_trades_df.loc[delayed_trades_df.get("reason", pd.Series(dtype="string")) == "executed", "execution_date"],
            errors="coerce",
        ).dt.normalize()
        base_last_exec = base_exec.max() if len(base_exec) else pd.NaT
        delayed_last_exec = delayed_exec.max() if len(delayed_exec) else pd.NaT
        aligned_end = pd.NaT
        if pd.notna(base_last_exec) and pd.notna(delayed_last_exec):
            aligned_end = min(base_last_exec, delayed_last_exec)

        nav_base_work = nav_df.copy()
        nav_delay_work = delayed_nav_df.copy()
        nav_base_work["date"] = pd.to_datetime(nav_base_work["date"], errors="coerce").dt.normalize()
        nav_delay_work["date"] = pd.to_datetime(nav_delay_work["date"], errors="coerce").dt.normalize()
        if pd.notna(aligned_end):
            nav_base_work = nav_base_work[nav_base_work["date"] <= aligned_end].copy()
            nav_delay_work = nav_delay_work[nav_delay_work["date"] <= aligned_end].copy()

        aligned_base_overall = _compute_metrics(nav_base_work, pd.DataFrame(), pd.DataFrame())
        aligned_delay_overall = _compute_metrics(nav_delay_work, pd.DataFrame(), pd.DataFrame())

        period_deltas: dict[str, object] = {}
        warning_periods: list[str] = []
        for period_name in ("development", "holdout"):
            base_period = summary["metrics"]["periods"].get(period_name, {})
            delayed_period = delayed_summary["metrics"]["periods"].get(period_name, {})
            cagr_delta_period = (
                float(delayed_period["cagr"] - base_period["cagr"])
                if ("cagr" in delayed_period and "cagr" in base_period and pd.notna(delayed_period["cagr"]) and pd.notna(base_period["cagr"]))
                else np.nan
            )
            sharpe_delta_period = (
                float(delayed_period["sharpe"] - base_period["sharpe"])
                if (
                    "sharpe" in delayed_period
                    and "sharpe" in base_period
                    and pd.notna(delayed_period["sharpe"])
                    and pd.notna(base_period["sharpe"])
                )
                else np.nan
            )

            base_period_aligned = _compute_metrics(
                nav_base_work[nav_base_work["period"] == period_name].copy(),
                pd.DataFrame(),
                pd.DataFrame(),
            )
            delayed_period_aligned = _compute_metrics(
                nav_delay_work[nav_delay_work["period"] == period_name].copy(),
                pd.DataFrame(),
                pd.DataFrame(),
            )
            aligned_cagr_delta_period = (
                float(delayed_period_aligned["cagr"] - base_period_aligned["cagr"])
                if (
                    pd.notna(delayed_period_aligned.get("cagr"))
                    and pd.notna(base_period_aligned.get("cagr"))
                )
                else np.nan
            )
            aligned_sharpe_delta_period = (
                float(delayed_period_aligned["sharpe"] - base_period_aligned["sharpe"])
                if (
                    pd.notna(delayed_period_aligned.get("sharpe"))
                    and pd.notna(base_period_aligned.get("sharpe"))
                )
                else np.nan
            )

            if (
                pd.notna(aligned_cagr_delta_period)
                and pd.notna(aligned_sharpe_delta_period)
                and aligned_cagr_delta_period > 0
                and aligned_sharpe_delta_period > 0
            ):
                warning_periods.append(period_name)

            period_deltas[period_name] = {
                "cagr_delta": cagr_delta_period,
                "sharpe_delta": sharpe_delta_period,
                "aligned_cagr_delta": aligned_cagr_delta_period,
                "aligned_sharpe_delta": aligned_sharpe_delta_period,
                "base_rows_aligned": int(base_period_aligned.get("rows", 0)),
                "delayed_rows_aligned": int(delayed_period_aligned.get("rows", 0)),
            }

        sensitivity_checks["execution_delay"] = {
            "delay_days": int(config.validation.delay_sanity_days),
            "warning_only": True,
            "notes": (
                "Delay-check improvements are warning signals only; investigate by period and overlap-aligned comparisons."
            ),
            "base_cagr": cagr_base,
            "delayed_cagr": cagr_delayed,
            "cagr_delta": (
                float(cagr_delayed - cagr_base)
                if (pd.notna(cagr_delayed) and pd.notna(cagr_base))
                else np.nan
            ),
            "base_sharpe": sharpe_base,
            "delayed_sharpe": sharpe_delayed,
            "sharpe_delta": (
                float(sharpe_delayed - sharpe_base)
                if (pd.notna(sharpe_delayed) and pd.notna(sharpe_base))
                else np.nan
            ),
            "cagr_weaker_or_equal": cagr_weaker,
            "sharpe_weaker_or_equal": sharpe_weaker,
            "aligned_end_date": aligned_end.date().isoformat() if pd.notna(aligned_end) else None,
            "aligned_cagr_delta": (
                float(aligned_delay_overall["cagr"] - aligned_base_overall["cagr"])
                if (
                    pd.notna(aligned_delay_overall.get("cagr"))
                    and pd.notna(aligned_base_overall.get("cagr"))
                )
                else np.nan
            ),
            "aligned_sharpe_delta": (
                float(aligned_delay_overall["sharpe"] - aligned_base_overall["sharpe"])
                if (
                    pd.notna(aligned_delay_overall.get("sharpe"))
                    and pd.notna(aligned_base_overall.get("sharpe"))
                )
                else np.nan
            ),
            "period_deltas": period_deltas,
            "warning_triggered_periods": warning_periods,
        }
        summary["sanity_flags"]["delay_plus1day_cagr_weaker_or_equal"] = cagr_weaker
        summary["sanity_flags"]["delay_plus1day_warning"] = bool(warning_periods)
    summary["sensitivity_checks"] = sensitivity_checks

    artifacts = _write_artifacts(
        config=config,
        strategy_id=strategy_id,
        cost_scenario=resolved_cost,
        nav_df=nav_df,
        trades_df=trades_df,
        holdings_df=holdings_df,
        rebalances_df=rebalances_df,
        summary=summary,
    )
    _refresh_counter_cost_comparison(
        config,
        strategy_id=strategy_id,
        just_run_cost_scenario=resolved_cost,
    )
    return {"summary": summary, "artifacts": artifacts}


def run_benchmark_backtest(config: Stage2Config, benchmark_ticker: str, cost_scenario: str) -> dict[str, object]:
    symbol = benchmark_ticker.upper()
    if symbol not in set(config.benchmarks.symbols):
        raise Stage2DataError(f"Benchmark {symbol} not enabled in config.benchmarks.symbols={config.benchmarks.symbols}")
    if cost_scenario.lower() not in {"base", "conservative"}:
        raise Stage2DataError("cost_scenario must be base or conservative")

    strategy_id = f"BENCHMARK_{symbol}"
    nav_df, trades_df, holdings_df, rebalances_df, summary = _simulate(
        config=config,
        strategy_id=strategy_id,
        cost_scenario=cost_scenario.lower(),
        benchmark_ticker=symbol,
    )
    artifacts = _write_artifacts(
        config=config,
        strategy_id=strategy_id,
        cost_scenario=cost_scenario.lower(),
        nav_df=nav_df,
        trades_df=trades_df,
        holdings_df=holdings_df,
        rebalances_df=rebalances_df,
        summary=summary,
    )
    return {"summary": summary, "artifacts": artifacts}
