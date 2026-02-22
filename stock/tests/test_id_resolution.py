import pandas as pd

from stock_pipeline.ingest_facts import _resolve_permaticker_from_history as resolve_fact_perm
from stock_pipeline.ingest_prices import _resolve_permaticker_from_history as resolve_price_perm


def _ticker_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "permaticker": [101, 202],
            "start_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "end_date": [pd.Timestamp("2025-12-31"), pd.Timestamp("2025-12-31")],
        }
    )


def test_fact_permaticker_resolution_handles_noncontiguous_index():
    table = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "availability_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]},
        index=[5, 7],
    )
    out = resolve_fact_perm(table, _ticker_history(), date_col="availability_date")
    assert list(out.index) == [5, 7]
    assert out.loc[5] == 101
    assert out.loc[7] == 202


def test_price_permaticker_resolution_handles_noncontiguous_index():
    table = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]},
        index=[9, 11],
    )
    out = resolve_price_perm(table, _ticker_history())
    assert list(out.index) == [9, 11]
    assert out.loc[9] == 101
    assert out.loc[11] == 202
