from datetime import datetime, timezone
import json
import pandas as pd
from mt5gold.data.store import (write_dataset, read_dataset, contract_snapshot,
                                dataframe_hash)
from mt5gold.data.clean import clean_rates
from mt5gold.data.fetch import fetch_rates
from tests.fakes import FakeBroker, make_rates

UTC = timezone.utc


def _clean_df():
    broker = FakeBroker(make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=200))
    raw = fetch_rates(broker, "XAUUSD", "M1",
                      datetime(2020, 1, 1, tzinfo=UTC),
                      datetime(2020, 1, 2, tzinfo=UTC))
    return clean_rates(raw, broker_tz_offset_hours=0)


def test_contract_snapshot_captures_spec_fields():
    broker = FakeBroker(make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=1))
    snap = contract_snapshot(broker, "XAUUSD")
    assert snap["contract_size"] == 100.0       # mapped from MT5 trade_contract_size
    assert snap["point"] == 0.01                # engine needs point for points->price
    assert snap["swap_long"] == -3.0
    assert snap["account_currency"] == "USD"


def test_dataframe_hash_is_deterministic():
    df = _clean_df()
    assert dataframe_hash(df) == dataframe_hash(df.copy())


def test_write_then_read_roundtrip(tmp_path):
    df = _clean_df()
    broker = FakeBroker(make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=1))
    contract = contract_snapshot(broker, "XAUUSD")
    created = datetime(2026, 7, 4, tzinfo=UTC)
    path = write_dataset(df, tmp_path, "XAUUSD", "M1", contract,
                         tz_offset_hours=0, created_utc=created)
    assert path.exists()

    back, manifest = read_dataset(tmp_path, "XAUUSD", "M1")
    pd.testing.assert_frame_equal(back, df)
    assert manifest["rows"] == len(df)
    assert manifest["data_hash"] == dataframe_hash(df)
    assert manifest["contract"]["contract_size"] == 100.0
    assert manifest["tz_offset_hours"] == 0
    assert manifest["created_utc"] == "2026-07-04T00:00:00+00:00"


def test_manifest_is_valid_json(tmp_path):
    df = _clean_df()
    broker = FakeBroker(make_rates(datetime(2020, 1, 1, tzinfo=UTC), n=1))
    write_dataset(df, tmp_path, "XAUUSD", "M1",
                  contract_snapshot(broker, "XAUUSD"),
                  tz_offset_hours=0, created_utc=datetime(2026, 7, 4, tzinfo=UTC))
    manifest_path = tmp_path / "clean" / "XAUUSD" / "M1" / "manifest.json"
    json.loads(manifest_path.read_text(encoding="utf-8"))  # no raise
