from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


@dataclass
class TradingCalendar:
    dates: pd.DatetimeIndex

    @classmethod
    def from_dataframe(cls, calendar_df: pd.DataFrame, date_col: str = "trade_date") -> "TradingCalendar":
        dates = pd.to_datetime(calendar_df[date_col]).sort_values().drop_duplicates()
        return cls(dates=pd.DatetimeIndex(dates))

    def next_trading_day(self, date: pd.Timestamp | str | pd.Series) -> pd.Timestamp | pd.Series:
        if isinstance(date, pd.Series):
            values = pd.to_datetime(date).to_numpy(dtype="datetime64[ns]")
            out = self._next_trading_day_numpy(values)
            return pd.Series(out, index=date.index)

        ts = pd.Timestamp(date).normalize().to_datetime64()
        out = self._next_trading_day_numpy(np.array([ts]))
        return pd.Timestamp(out[0]) if out[0] != np.datetime64("NaT") else pd.NaT

    def _next_trading_day_numpy(self, dates: np.ndarray) -> np.ndarray:
        if dates.size == 0:
            return dates
        trading_days = self.dates.to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(trading_days, dates, side="right")
        out = np.full(dates.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = positions < len(trading_days)
        out[valid] = trading_days[positions[valid]]
        return out


def build_nyse_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    cal = mcal.get_calendar("XNYS")
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    trading_days = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    return pd.DataFrame({"trade_date": trading_days, "is_trading_day": True})
