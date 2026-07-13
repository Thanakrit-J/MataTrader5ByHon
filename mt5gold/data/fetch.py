"""Fetch historical rates in date chunks via a Broker. No MT5 import here."""
from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def _empty() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_rates(broker, symbol: str, timeframe: str,
                start: datetime, end: datetime, chunk_days: int = 30) -> pd.DataFrame:
    frames = []
    cursor = start
    step = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        rates = broker.copy_rates_range(symbol, timeframe, cursor, chunk_end)
        if rates is not None and len(rates) > 0:
            frames.append(pd.DataFrame(rates))
        cursor = chunk_end
    if not frames:
        return _empty()
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[COLUMNS]
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df
