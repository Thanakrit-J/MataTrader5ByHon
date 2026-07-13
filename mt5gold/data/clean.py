"""Normalize broker time to UTC, validate integrity, flag spread anomalies.
Never fabricates weekend/holiday bars (spec §4.2)."""
from __future__ import annotations
import numpy as np
import pandas as pd


class DataValidationError(Exception):
    pass


def validate_rates(df: pd.DataFrame) -> None:
    if not df["time"].is_monotonic_increasing:
        raise DataValidationError("time column is not sorted ascending")
    if df["time"].duplicated().any():
        raise DataValidationError("duplicate time values present")
    hi_ok = df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9
    if not hi_ok.all():
        raise DataValidationError(f"high < max(open,close) at {(~hi_ok).sum()} bars")
    lo_ok = df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9
    if not lo_ok.all():
        raise DataValidationError(f"low > min(open,close) at {(~lo_ok).sum()} bars")
    if (df["spread"] <= 0).any() or df["spread"].isna().any():
        raise DataValidationError("spread column has missing or non-positive values")


def clean_rates(df: pd.DataFrame, broker_tz_offset_hours: int) -> pd.DataFrame:
    out = df.copy()
    out["time"] = out["time"] - pd.Timedelta(hours=broker_tz_offset_hours)
    out = out.sort_values("time").reset_index(drop=True)
    validate_rates(out)
    # Flag anomalous spread via median absolute deviation (robust to outliers).
    med = out["spread"].median()
    mad = (out["spread"] - med).abs().median()
    scale = mad if mad > 0 else 1.0
    out["spread_anomaly"] = ((out["spread"] - med).abs() / scale) > 10.0
    return out
