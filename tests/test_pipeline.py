from datetime import datetime, timezone
import pandas as pd
from mt5gold.data.pipeline import run_ingest
from mt5gold.data.store import read_dataset
from mt5gold.config import DataConfig
from tests.fakes import FakeBroker, make_rates

UTC = timezone.utc


def test_run_ingest_end_to_end(tmp_path):
    rates = make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=500)
    broker = FakeBroker(rates, tz_offset_hours=2)
    cfg = DataConfig(symbol="XAUUSD", base_timeframe="M1",
                     history_start="2020-01-01", history_end="2020-01-02")
    path = run_ingest(broker, cfg, tmp_path,
                      created_utc=datetime(2026, 7, 4, tzinfo=UTC))
    assert path.exists()

    df, manifest = read_dataset(tmp_path, "XAUUSD", "M1")
    assert manifest["tz_offset_hours"] == 2          # detected from account_info
    assert manifest["rows"] == len(df) > 0
    assert manifest["contract"]["contract_size"] == 100.0
    # tz shift applied: broker stamp 2020-01-01 00:00 UTC minus 2h offset
    assert df["time"].iloc[0] == pd.Timestamp("2019-12-31 22:00", tz="UTC")
    assert str(df["time"].dt.tz) == "UTC"


def test_run_ingest_is_reproducible(tmp_path):
    rates = make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=300)
    cfg = DataConfig(history_start="2020-01-01", history_end="2020-01-02")
    created = datetime(2026, 7, 4, tzinfo=UTC)
    run_ingest(FakeBroker(rates, 0), cfg, tmp_path / "a", created_utc=created)
    run_ingest(FakeBroker(rates, 0), cfg, tmp_path / "b", created_utc=created)
    _, m1 = read_dataset(tmp_path / "a", "XAUUSD", "M1")
    _, m2 = read_dataset(tmp_path / "b", "XAUUSD", "M1")
    assert m1["data_hash"] == m2["data_hash"]         # same inputs → same hash
