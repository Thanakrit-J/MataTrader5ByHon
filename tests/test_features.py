import numpy as np, pandas as pd
from datetime import datetime, timezone
from mt5gold.core.features import build_features, feature_row, WARMUP_BARS

UTC = timezone.utc


def _df(n=800):   # must exceed WARMUP_BARS so trailing-window equality has room
    idx = pd.date_range(datetime(2020, 1, 1, tzinfo=UTC), periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"time": idx, "open": close, "high": close + 1,
                         "low": close - 1, "close": close, "tick_volume": 1, "spread": 20.0})


def test_build_features_has_expected_columns():
    f = build_features(_df())
    for c in ["ema9", "ema21", "ema50", "rsi14", "atr14", "atr_pctile", "swing_high", "swing_low"]:
        assert c in f.columns


def test_features_are_causal_truncation_equality():
    df = _df()
    full = build_features(df)
    for t in [WARMUP_BARS + 5, WARMUP_BARS + 50, len(df) - 1]:
        trunc = build_features(df.iloc[:t + 1])
        for c in full.columns:
            assert abs(float(full[c].iloc[t]) - float(trunc[c].iloc[t])) < 1e-6, (c, t)


def test_feature_row_matches_build_features():
    df = _df()
    full = build_features(df)
    t = len(df) - 1
    row = feature_row(df.iloc[t - WARMUP_BARS:t + 1])
    for c in full.columns:
        assert abs(float(row[c]) - float(full[c].iloc[t])) < 1e-6, c
