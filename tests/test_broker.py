from datetime import datetime, timezone
from mt5gold.live.broker import TIMEFRAME_MINUTES, Broker
from tests.fakes import FakeBroker, make_rates


def test_timeframe_minutes_table():
    assert TIMEFRAME_MINUTES["M1"] == 1
    assert TIMEFRAME_MINUTES["H1"] == 60


def test_fakebroker_satisfies_protocol_and_filters_by_range():
    rates = make_rates(datetime(2020, 1, 1, tzinfo=timezone.utc), n=10)
    broker: Broker = FakeBroker(rates)
    broker.connect()
    got = broker.copy_rates_range(
        "XAUUSD", "M1",
        datetime(2020, 1, 1, 0, 2, tzinfo=timezone.utc),
        datetime(2020, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    assert len(got) == 4                      # minutes 2,3,4,5 inclusive
    assert broker.symbol_info("XAUUSD")["point"] == 0.01
