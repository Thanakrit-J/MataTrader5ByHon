from datetime import datetime, timezone
import numpy as np, pandas as pd
from mt5gold.data.resample import resample_ohlcv

UTC = timezone.utc


def _m1(n, start=datetime(2020, 1, 1, tzinfo=UTC)):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    base = np.arange(n, dtype=float)
    return pd.DataFrame({"time": idx, "open": 2000 + base, "high": 2000 + base + 0.5,
                         "low": 2000 + base - 0.5, "close": 2000 + base + 0.2,
                         "tick_volume": np.ones(n), "spread": np.full(n, 20.0)})


def test_resample_m5_aggregates_5_bars():
    out = resample_ohlcv(_m1(15), "M5")
    assert len(out) == 3
    assert out["open"].iloc[0] == 2000.0            # first of first 5
    assert out["close"].iloc[0] == 2000.0 + 4 + 0.2  # last of first 5
    assert out["high"].iloc[0] == 2000.0 + 4 + 0.5
    assert out["tick_volume"].iloc[0] == 5


def test_resample_drops_empty_buckets():
    # two clusters with a 3-hour gap → no phantom bars in between
    a = _m1(10, datetime(2020, 1, 3, 21, 55, tzinfo=UTC))   # Fri late
    b = _m1(10, datetime(2020, 1, 6, 1, 0, tzinfo=UTC))     # Mon
    out = resample_ohlcv(pd.concat([a, b], ignore_index=True), "M15")
    # gap must not create bars; count of M15 buckets equals only populated ones
    assert out["time"].is_monotonic_increasing
    assert (out["time"].diff().dropna() > pd.Timedelta("15min")).any()  # a real gap exists
