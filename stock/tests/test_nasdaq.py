from stock_pipeline.nasdaq import _datatable_to_zip_glob, _filter_columns_from_params, _resolve_fallback_usecols


def test_filter_columns_from_params_extracts_base_fields():
    params = {
        "date.gte": "2020-01-01",
        "date.lte": "2020-12-31",
        "ticker": "SPY",
        "dimension": "ARQ",
        "qopts.per_page": 10000,
    }
    cols = _filter_columns_from_params(params)
    assert cols == ["date", "ticker", "dimension"]


def test_resolve_fallback_usecols_includes_filter_columns():
    csv_columns = ["date", "ticker", "close", "dimension", "other"]
    params = {"date.gte": "2020-01-01", "dimension": "ARQ"}
    selected = ["ticker", "close"]
    usecols = _resolve_fallback_usecols(csv_columns, params, selected)
    assert usecols == ["ticker", "close", "date", "dimension"]


def test_datatable_zip_glob_uses_exact_table_prefix():
    assert _datatable_to_zip_glob("SHARADAR/SF3") == "SHARADAR_SF3_*.zip"
