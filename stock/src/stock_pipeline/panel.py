from __future__ import annotations

from pathlib import Path

import pandas as pd

from .asof import deterministic_asof_join
from .config import Stage1Config
from .io_utils import read_parquet, write_parquet
from .manifest import write_clean_manifest


def _join_fact(
    panel: pd.DataFrame,
    fact: pd.DataFrame,
    prefix: str,
    *,
    enforce_lastupdated_cutoff: bool = True,
) -> pd.DataFrame:
    joined = deterministic_asof_join(
        panel[["permaticker", "asof_date"]],
        fact,
        enforce_lastupdated_cutoff=enforce_lastupdated_cutoff,
        prefix=prefix,
    )
    fact_cols = [col for col in joined.columns if col.startswith(f"{prefix}_")]
    return panel.merge(joined[["permaticker", "asof_date", *fact_cols]], on=["permaticker", "asof_date"], how="left")


def build_daily_panel(config: Stage1Config) -> Path:
    universe = read_parquet(config.storage.clean_root / "tradable_universe_daily.parquet")
    panel = universe[universe["eligible_flag"]].copy()
    panel["asof_date"] = pd.to_datetime(panel["asof_date"]).dt.normalize()

    fundamentals_path = config.storage.clean_root / "fundamentals_pit.parquet"
    if fundamentals_path.exists():
        fundamentals = read_parquet(fundamentals_path)
        fundamentals["knowledge_date"] = pd.to_datetime(fundamentals["knowledge_date"]).dt.normalize()
        if "lastupdated" in fundamentals.columns:
            fundamentals["lastupdated"] = pd.to_datetime(fundamentals["lastupdated"], errors="coerce").dt.normalize()
        panel = _join_fact(
            panel,
            fundamentals,
            prefix="fund",
            enforce_lastupdated_cutoff=config.dataset("fundamentals").enforce_lastupdated_cutoff,
        )

    insiders_agg_path = config.storage.clean_root / "insiders_daily_agg.parquet"
    if insiders_agg_path.exists():
        insiders = read_parquet(insiders_agg_path)
        insiders["knowledge_date"] = pd.to_datetime(insiders["knowledge_date"]).dt.normalize()
        panel = _join_fact(
            panel,
            insiders,
            prefix="ins",
            enforce_lastupdated_cutoff=config.dataset("insiders").enforce_lastupdated_cutoff,
        )

    institutions_agg_path = config.storage.clean_root / "institutions_daily_agg.parquet"
    if institutions_agg_path.exists():
        institutions = read_parquet(institutions_agg_path)
        institutions["knowledge_date"] = pd.to_datetime(institutions["knowledge_date"]).dt.normalize()
        panel = _join_fact(
            panel,
            institutions,
            prefix="inst",
            enforce_lastupdated_cutoff=config.dataset("institutions").enforce_lastupdated_cutoff,
        )

    panel = panel.sort_values(["asof_date", "permaticker"]).reset_index(drop=True)

    out_path = config.storage.clean_root / "daily_panel.parquet"
    write_parquet(panel, out_path)
    write_clean_manifest(
        config.storage.manifest_root,
        alias="daily_panel",
        clean_path=out_path,
        row_count=len(panel),
        columns=list(panel.columns),
    )
    return out_path
