from __future__ import annotations

import numpy as np
import pandas as pd

from .asof import deterministic_asof_join
from .config import Stage1Config
from .io_utils import read_parquet, write_parquet
from .manifest import write_clean_manifest


def _latest_classification(classification: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "permaticker",
        "ticker",
        "security_type_raw",
        "exchange",
        "country",
        "is_otc",
        "is_adr",
        "is_etf",
        "is_common_stock",
    ]
    for col in cols:
        if col not in classification.columns:
            classification[col] = pd.NA
    latest = classification[cols].sort_values(["permaticker", "ticker"]).drop_duplicates("permaticker", keep="last")
    return latest


def _classification_asof(master_security: pd.DataFrame, panel_index: pd.DataFrame) -> pd.DataFrame:
    class_cols = [
        "security_type_raw",
        "exchange",
        "country",
        "is_otc",
        "is_adr",
        "is_etf",
        "is_common_stock",
    ]
    required_cols = ["permaticker", "start_date", "end_date", *class_cols]
    for col in required_cols:
        if col not in master_security.columns:
            master_security[col] = pd.NA

    ref = master_security[required_cols].copy()
    ref["start_date"] = pd.to_datetime(ref["start_date"], errors="coerce").dt.normalize()
    ref["end_date"] = pd.to_datetime(ref["end_date"], errors="coerce").dt.normalize()
    ref = ref.dropna(subset=["permaticker", "start_date"]).copy()

    # Resolve duplicate ranges deterministically.
    ref = (
        ref.sort_values(["permaticker", "start_date", "end_date"])
        .drop_duplicates(["permaticker", "start_date", "end_date"], keep="last")
        .reset_index(drop=True)
    )

    left = panel_index[["permaticker", "asof_date"]].copy()
    left = left.reset_index().rename(columns={"index": "_rowid"})
    left["asof_date"] = pd.to_datetime(left["asof_date"], errors="coerce").dt.normalize()
    left = left.dropna(subset=["permaticker", "asof_date"]).copy()

    left_sorted = left.sort_values(["asof_date", "permaticker"]).reset_index(drop=True)
    right_sorted = ref.sort_values(["start_date", "permaticker"]).reset_index(drop=True)
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="asof_date",
        right_on="start_date",
        by="permaticker",
        direction="backward",
        allow_exact_matches=True,
    )

    valid_range = merged["end_date"].isna() | (merged["asof_date"] <= merged["end_date"])
    bool_cols = ["is_otc", "is_adr", "is_etf", "is_common_stock"]
    for col in bool_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype("boolean")
    for col in class_cols:
        merged.loc[~valid_range, col] = pd.NA

    merged = merged.sort_values("_rowid").reset_index(drop=True)
    return merged[class_cols]


def _build_reason_column(df: pd.DataFrame) -> pd.Series:
    reasons: list[list[str]] = []
    for row in df.itertuples(index=False):
        row_reasons: list[str] = []
        if not bool(getattr(row, "screen_security_type_pass", False)):
            row_reasons.append("security_type")
        if not bool(getattr(row, "screen_price_pass", False)):
            row_reasons.append("price")
        if not bool(getattr(row, "screen_adv20_pass", False)):
            row_reasons.append("adv20")
        if not bool(getattr(row, "screen_market_cap_pass", False)):
            row_reasons.append("market_cap")
        if bool(getattr(row, "is_etf_allowlist", False)):
            row_reasons = ["etf_allowlist"]
        reasons.append(row_reasons)
    return pd.Series([";".join(items) if items else "eligible" for items in reasons], index=df.index)


def build_tradable_universe(config: Stage1Config) -> pd.DataFrame:
    master_path = config.storage.clean_root / "master_security.parquet"
    classification_path = config.storage.clean_root / "security_classification.parquet"
    equity_prices = read_parquet(config.storage.clean_root / "equity_prices_daily.parquet")
    etf_prices = read_parquet(config.storage.clean_root / "etf_prices_daily.parquet")

    equity = equity_prices.rename(columns={"date": "asof_date"}).copy()
    equity["asof_date"] = pd.to_datetime(equity["asof_date"]).dt.normalize()
    if master_path.exists():
        master_security = read_parquet(master_path)
        class_asof = _classification_asof(master_security, equity[["permaticker", "asof_date"]])
        for col in class_asof.columns:
            equity[col] = class_asof[col].values
    elif classification_path.exists():
        classification = read_parquet(classification_path)
        latest = _latest_classification(classification)
        equity = equity.merge(latest, on="permaticker", how="left", suffixes=("", "_class"))
    else:
        raise FileNotFoundError("Missing both master_security.parquet and security_classification.parquet")

    for col in ("is_common_stock", "is_otc", "is_adr"):
        if col in equity.columns:
            equity[col] = equity[col].astype("boolean")

    equity["screen_security_type_pass"] = (
        equity["is_common_stock"].fillna(False)
        & ~equity["is_otc"].fillna(False)
        & ~equity["is_adr"].fillna(False)
    )
    equity["screen_price_pass"] = equity["close_for_returns"] > config.universe.min_price
    equity["screen_adv20_pass"] = equity["adv20_dollar"] > config.universe.min_adv20_dollar

    market_cap_available = False
    equity["market_cap"] = np.nan
    fundamentals_path = config.storage.clean_root / "fundamentals_pit.parquet"
    if fundamentals_path.exists():
        fundamentals = read_parquet(fundamentals_path)
        if "market_cap" in fundamentals.columns and "knowledge_date" in fundamentals.columns:
            market_cap_available = True
            panel_idx = equity[["permaticker", "asof_date"]].copy()
            market_cap_join = deterministic_asof_join(
                panel_idx,
                fundamentals[["permaticker", "knowledge_date", "market_cap", "lastupdated"]],
                enforce_lastupdated_cutoff=config.dataset("fundamentals").enforce_lastupdated_cutoff,
                prefix="fund",
            )
            market_cap_view = (
                market_cap_join[["permaticker", "asof_date", "fund_market_cap"]]
                .drop_duplicates(["permaticker", "asof_date"], keep="last")
            )
            equity = equity.merge(
                market_cap_view,
                on=["permaticker", "asof_date"],
                how="left",
                sort=False,
            )
            equity["market_cap"] = equity["fund_market_cap"]
            equity = equity.drop(columns=["fund_market_cap"])

    if market_cap_available:
        # If point-in-time market cap is missing for a row, do not hard-fail eligibility.
        # Stage 1 requires applying the screen only where market cap is available.
        equity["screen_market_cap_pass"] = equity["market_cap"].isna() | (
            equity["market_cap"] > config.universe.min_market_cap
        )
    else:
        equity["screen_market_cap_pass"] = True

    equity["is_etf_allowlist"] = False

    etf = etf_prices.rename(columns={"date": "asof_date"}).copy()
    etf["asof_date"] = pd.to_datetime(etf["asof_date"]).dt.normalize()
    etf["ticker"] = etf["ticker"].astype("string").str.upper()
    etf = etf[etf["ticker"].isin(config.etf_allowlist)].copy()

    etf = etf.assign(
        security_type_raw="ETF",
        exchange=pd.NA,
        country="USA",
        is_otc=False,
        is_adr=False,
        is_etf=True,
        is_common_stock=False,
        screen_security_type_pass=True,
        screen_price_pass=True,
        screen_adv20_pass=True,
        market_cap=np.nan,
        screen_market_cap_pass=True,
        is_etf_allowlist=True,
    )

    keep_cols = [
        "asof_date",
        "permaticker",
        "ticker",
        "close_for_returns",
        "adv20_dollar",
        "market_cap",
        "security_type_raw",
        "exchange",
        "country",
        "is_otc",
        "is_adr",
        "is_etf",
        "is_common_stock",
        "screen_security_type_pass",
        "screen_price_pass",
        "screen_adv20_pass",
        "screen_market_cap_pass",
        "is_etf_allowlist",
    ]

    universe = pd.concat([equity[keep_cols], etf[keep_cols]], ignore_index=True)
    universe["eligible_flag"] = (
        universe["screen_security_type_pass"]
        & universe["screen_price_pass"]
        & universe["screen_adv20_pass"]
        & universe["screen_market_cap_pass"]
    ) | universe["is_etf_allowlist"]

    universe["eligibility_reason"] = _build_reason_column(universe)

    universe = universe.sort_values(["asof_date", "permaticker", "ticker"]).drop_duplicates(
        ["asof_date", "permaticker"], keep="last"
    )

    out_path = config.storage.clean_root / "tradable_universe_daily.parquet"
    write_parquet(universe.reset_index(drop=True), out_path)
    write_clean_manifest(
        config.storage.manifest_root,
        alias="tradable_universe_daily",
        clean_path=out_path,
        row_count=len(universe),
        columns=list(universe.columns),
    )

    return universe
