from __future__ import annotations
import numpy as np
import pandas as pd

_ATR_PCTILE_WINDOW = 252
# Warmup = longest explicit lookback (ATR percentile window) + a convergence
# margin large enough that adjust=False EWMs reproduce the full-series value
# from a trailing window to 1e-6. The slowest is the span-50 EMA: its seed
# influence decays like (1 - 2/51)^n, so ~250 extra bars are needed to make a
# feature_row window match build_features (the train/live no-skew guarantee).
WARMUP_BARS = _ATR_PCTILE_WINDOW + 250


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s, period=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame, df_htf=None) -> pd.DataFrame:
    c = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["ema9"] = _ema(c, 9)
    out["ema21"] = _ema(c, 21)
    out["ema50"] = _ema(c, 50)
    out["rsi14"] = _rsi(c, 14)
    out["atr14"] = _atr(df, 14)
    out["atr_pctile"] = (out["atr14"]
        .rolling(_ATR_PCTILE_WINDOW, min_periods=20)
        .apply(lambda w: (w.iloc[-1] >= w).mean(), raw=False))
    out["swing_high"] = df["high"].rolling(20, min_periods=1).max()   # trailing only
    out["swing_low"] = df["low"].rolling(20, min_periods=1).min()
    return out


def feature_row(window: pd.DataFrame, htf_window=None) -> pd.Series:
    return build_features(window, htf_window).iloc[-1]
