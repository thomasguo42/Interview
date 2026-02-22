import pandas as pd

from stock_pipeline.config import ValidationConfig
from stock_pipeline.validation import (
    _active_missing_ratio,
    leakage_audit,
    universe_market_cap_alignment_audit,
)


def test_active_missing_ratio_accounts_for_recent_inceptions():
    calendar = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        )
    )

    # Both securities are active on max date (2024-01-05).
    # Security 2 only starts on 2024-01-04 and should not be penalized for earlier days.
    df = pd.DataFrame(
        {
            "permaticker": [1, 1, 1, 1, 1, 2, 2],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
        }
    )

    ratio = _active_missing_ratio(df, calendar)
    assert ratio == 0.0


def test_leakage_audit_honors_lastupdated_policy_toggle():
    panel = pd.DataFrame(
        {
            "permaticker": [1],
            "asof_date": [pd.Timestamp("2024-01-03")],
            "fund_knowledge_date": [pd.Timestamp("2024-01-02")],
            "fund_lastupdated": [pd.Timestamp("2024-01-10")],
        }
    )

    strict = leakage_audit(panel, enforce_lastupdated_cutoff_by_prefix={"fund": True})
    assert strict["passed"] is False

    lenient = leakage_audit(panel, enforce_lastupdated_cutoff_by_prefix={"fund": False})
    assert lenient["passed"] is True


def _validation_cfg() -> ValidationConfig:
    return ValidationConfig(
        min_equity_eligible_unique=1,
        min_equity_eligible_rows=1,
        extreme_jump_threshold=0.8,
        max_extreme_jump_ratio=1.0,
        max_missing_active_ratio=1.0,
        max_zero_volume_ratio=1.0,
        min_fundamental_coverage_ratio=0.0,
        max_start_date_gap_days=3650,
        market_cap_alignment_sample_size=1000,
        max_market_cap_missing_ratio=0.0,
        max_market_cap_mismatch_ratio=0.0,
        market_cap_match_rtol=1e-6,
        market_cap_match_atol=1e-3,
    )


def test_universe_market_cap_alignment_audit_fails_on_misalignment():
    panel = pd.DataFrame(
        {
            "permaticker": [1, 2],
            "asof_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
            "fund_market_cap": [100.0, 200.0],
        }
    )
    universe = pd.DataFrame(
        {
            "permaticker": [1, 2],
            "asof_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
            "market_cap": [200.0, 100.0],
        }
    )

    out = universe_market_cap_alignment_audit(panel, universe, _validation_cfg())
    assert out["passed"] is False
    assert out["mismatch_count"] == 2


def test_universe_market_cap_alignment_audit_passes_when_equal():
    panel = pd.DataFrame(
        {
            "permaticker": [1, 2],
            "asof_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
            "fund_market_cap": [100.0, 200.0],
        }
    )
    universe = pd.DataFrame(
        {
            "permaticker": [1, 2],
            "asof_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
            "market_cap": [100.0, 200.0],
        }
    )

    out = universe_market_cap_alignment_audit(panel, universe, _validation_cfg())
    assert out["passed"] is True
    assert out["mismatch_count"] == 0
