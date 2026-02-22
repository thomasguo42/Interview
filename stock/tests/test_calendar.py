import pandas as pd

from stock_pipeline.calendar import TradingCalendar, build_nyse_calendar


def test_next_trading_day_skips_weekend_and_holiday():
    calendar_df = build_nyse_calendar("2024-01-01", "2024-01-10")
    cal = TradingCalendar.from_dataframe(calendar_df)

    # 2024-01-05 is Friday, next trading day is Monday 2024-01-08.
    assert cal.next_trading_day(pd.Timestamp("2024-01-05")) == pd.Timestamp("2024-01-08")

    # 2024-01-01 is NYSE holiday, next trading day is 2024-01-02.
    assert cal.next_trading_day(pd.Timestamp("2024-01-01")) == pd.Timestamp("2024-01-02")
