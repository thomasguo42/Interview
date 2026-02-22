from stock_pipeline.schema_lock import DatasetSchemaLock, required_pull_columns


def test_required_pull_columns_merges_and_dedupes_case_insensitive():
    lock = DatasetSchemaLock(
        alias="sample",
        datatable="SHARADAR/TEST",
        columns=[],
        resolved_fields={
            "ticker": "ticker",
            "permaticker": "permaticker",
            "date": "date",
            "close": "close",
            "optional": None,
        },
        required_columns=["Ticker", "DATE"],
    )

    cols = required_pull_columns(lock, extra_columns=["volume", "ticker", "VOLUME"])
    assert cols == ["Ticker", "DATE", "permaticker", "close", "volume"]
