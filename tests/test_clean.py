from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from mt5gold.data.clean import clean_rates, validate_rates, DataValidationError
from mt5gold.data.fetch import fetch_rates
from tests.fakes import FakeBroker, make_rates

UTC = timezone.utc


def _df(n=50, spread=20):
    broker = FakeBroker(make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=n, spread=spread))
    return fetch_rates(broker, "XAUUSD", "M1",
                       datetime(2020, 1, 1, tzinfo=UTC),
                       datetime(2020, 1, 2, tzinfo=UTC))


def test_validate_passes_on_good_data():
    validate_rates(_df())  # no raise


def test_validate_rejects_bad_high():
    df = _df()
    df.loc[5, "high"] = df.loc[5, "close"] - 1.0     # high below close
    with pytest.raises(DataValidationError, match="high"):
        validate_rates(df)


def test_validate_rejects_nonpositive_spread():
    df = _df()
    df.loc[3, "spread"] = 0
    with pytest.raises(DataValidationError, match="spread"):
        validate_rates(df)


def test_clean_shifts_broker_offset_to_utc():
    # Broker stamps are UTC+3; cleaning must subtract 3h to land on true UTC
    df = _df()
    first_before = df["time"].iloc[0]
    cleaned = clean_rates(df, broker_tz_offset_hours=3)
    assert cleaned["time"].iloc[0] == first_before - pd.Timedelta(hours=3)
    assert "spread_anomaly" in cleaned.columns
    assert cleaned["spread_anomaly"].dtype == bool


def test_clean_imputes_nonpositive_spread():
    # Real MT5 feeds report spread==0 on many bars; cleaning imputes the median
    # of positive spreads and records provenance, so validation then passes.
    df = _df(n=50, spread=20)
    df.loc[10, "spread"] = 0
    df.loc[11, "spread"] = 0
    cleaned = clean_rates(df, broker_tz_offset_hours=0)
    assert (cleaned["spread"] > 0).all()              # nothing non-positive remains
    assert cleaned["spread_imputed"].dtype == bool
    assert cleaned["spread_imputed"].sum() == 2        # two bars flagged
    assert bool(cleaned.loc[10, "spread_imputed"]) is True
    assert cleaned.loc[10, "spread"] == 20             # imputed to positive median


def test_clean_flags_spread_spike():
    df = _df(n=50, spread=20)
    df.loc[25, "spread"] = 900                        # news-time blowout
    cleaned = clean_rates(df, broker_tz_offset_hours=0)
    assert bool(cleaned.loc[25, "spread_anomaly"]) is True
    assert cleaned["spread_anomaly"].sum() == 1
