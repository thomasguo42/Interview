import pandas as pd

from stock_pipeline.asof import deterministic_asof_join


def test_deterministic_asof_join_respects_lastupdated_cutoff():
    panel = pd.DataFrame(
        {
            "permaticker": [1, 1],
            "asof_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-06")],
        }
    )

    fact = pd.DataFrame(
        {
            "permaticker": [1, 1],
            "knowledge_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "lastupdated": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-03")],
            "metric": [100.0, 200.0],
        }
    )

    out = deterministic_asof_join(panel, fact, prefix="fund")

    # On 2024-01-03, row with lastupdated=2024-01-05 is not yet eligible.
    assert out.loc[0, "fund_metric"] == 200.0

    # On 2024-01-06, later revision is eligible and should be selected.
    assert out.loc[1, "fund_metric"] == 100.0


def test_deterministic_asof_join_can_ignore_lastupdated_cutoff():
    panel = pd.DataFrame(
        {
            "permaticker": [1],
            "asof_date": [pd.Timestamp("2024-01-03")],
        }
    )
    fact = pd.DataFrame(
        {
            "permaticker": [1],
            "knowledge_date": [pd.Timestamp("2024-01-02")],
            "lastupdated": [pd.Timestamp("2024-01-10")],
            "metric": [123.0],
        }
    )

    out = deterministic_asof_join(panel, fact, enforce_lastupdated_cutoff=False, prefix="fund")
    assert out.loc[0, "fund_metric"] == 123.0


def test_deterministic_asof_join_does_not_prefix_left_payload_columns():
    panel = pd.DataFrame(
        {
            "permaticker": [1],
            "asof_date": [pd.Timestamp("2024-01-03")],
            "left_feature": [7.0],
        }
    )
    fact = pd.DataFrame(
        {
            "permaticker": [1],
            "knowledge_date": [pd.Timestamp("2024-01-02")],
            "metric": [11.0],
        }
    )

    out = deterministic_asof_join(panel, fact, prefix="fund")
    assert "left_feature" in out.columns
    assert "fund_left_feature" not in out.columns
    assert out.loc[0, "fund_metric"] == 11.0
