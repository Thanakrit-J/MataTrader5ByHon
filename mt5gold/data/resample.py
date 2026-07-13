from __future__ import annotations
import pandas as pd
from mt5gold.live.broker import TIMEFRAME_MINUTES

_AGG = {"open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum", "spread": "mean"}


def resample_ohlcv(df_m1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    minutes = TIMEFRAME_MINUTES[timeframe]
    rule = f"{minutes}min"
    g = (df_m1.set_index("time")
              .resample(rule, label="left", closed="left")
              .agg(_AGG)
              .dropna(subset=["open"]))          # drop empty (weekend/holiday) buckets
    return g.reset_index()
