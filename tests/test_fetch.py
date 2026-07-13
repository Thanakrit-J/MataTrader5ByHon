from datetime import datetime, timezone
import pandas as pd
from mt5gold.data.fetch import fetch_rates
from tests.fakes import FakeBroker, make_rates

UTC = timezone.utc


def test_fetch_rates_returns_sorted_unique_utc():
    rates = make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=100)
    broker = FakeBroker(rates)
    df = fetch_rates(broker, "XAUUSD", "M1",
                     datetime(2020, 1, 1, tzinfo=UTC),
                     datetime(2020, 1, 1, 3, tzinfo=UTC),
                     chunk_days=1)
    assert list(df.columns) == ["time", "open", "high", "low", "close",
                                "tick_volume", "spread", "real_volume"]
    assert str(df["time"].dt.tz) == "UTC"
    assert df["time"].is_monotonic_increasing
    assert not df["time"].duplicated().any()
    assert len(df) == 100


def test_fetch_rates_empty_when_no_data():
    broker = FakeBroker(make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=0))
    df = fetch_rates(broker, "XAUUSD", "M1",
                     datetime(2021, 1, 1, tzinfo=UTC),
                     datetime(2021, 1, 2, tzinfo=UTC))
    assert df.empty
    assert list(df.columns) == ["time", "open", "high", "low", "close",
                                "tick_volume", "spread", "real_volume"]


def test_fetch_rates_merges_chunks_without_gaps_or_dupes():
    # 3 days of M1 across a 1-day chunk size must stitch seamlessly
    rates = make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=3 * 1440)
    broker = FakeBroker(rates)
    df = fetch_rates(broker, "XAUUSD", "M1",
                     datetime(2020, 1, 1, tzinfo=UTC),
                     datetime(2020, 1, 4, tzinfo=UTC),
                     chunk_days=1)
    assert len(df) == 3 * 1440
    assert not df["time"].duplicated().any()
