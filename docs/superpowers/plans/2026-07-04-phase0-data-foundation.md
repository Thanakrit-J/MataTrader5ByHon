# Phase 0 — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible XAUUSD M1 data pipeline (fetch → clean → store as Parquet with a hashed manifest) behind a mockable MT5 wrapper, so every later phase reads one honest source of truth.

**Architecture:** A single `broker.py` module is the only code that imports `MetaTrader5`; everything else depends on a narrow `Broker` protocol so tests run with a `FakeBroker` and no live terminal. `fetch.py` pulls M1 in date chunks, `clean.py` normalizes timezone/gaps/spread and validates, `store.py` writes partitioned Parquet plus a JSON manifest carrying data hash, timezone offset, and a broker contract-spec snapshot.

**Tech Stack:** Python 3.11, MetaTrader5, pandas, numpy, pyarrow, pytz, pytest.

## Global Constraints

- Python 3.11; dependencies limited to: `MetaTrader5`, `pandas`, `numpy`, `pyarrow`, `pytz`, `pytest` (copied from spec §3 Tech stack).
- `broker.py` is the ONLY module allowed to `import MetaTrader5` (spec Principle 1 / finding 30). All other modules take a `Broker` instance or read from stored data.
- All stored timestamps are timezone-aware UTC; the broker's UTC offset is detected and recorded, never assumed (spec §4.2 item 1).
- MT5 `copy_rates_*` OHLC is BID-based; the per-bar `spread` column (points) MUST be preserved (spec §4.1, §4.2 item 4).
- No hardcoded secrets; DB/credentials come from environment variables (spec §7.1, §9).
- Parquet layout: `data/{raw|clean}/{symbol}/{timeframe}/…`; every clean dataset has a sibling `manifest.json` (spec §4.3).
- Package name is `mt5gold`; run tests with `python -m pytest`.

---

### Task 1: Package scaffold + typed config

**Files:**
- Create: `mt5gold/__init__.py`, `mt5gold/core/__init__.py`, `mt5gold/data/__init__.py`, `mt5gold/live/__init__.py`
- Create: `mt5gold/config.py`
- Create: `tests/__init__.py`, `tests/test_config.py`
- Create: `pytest.ini`

**Interfaces:**
- Produces: `DataConfig` (frozen dataclass: `symbol: str`, `base_timeframe: str`, `history_start: str`, `history_end: str | None`, `broker_tz_offset_hours: int | None`); `config_hash(obj) -> str` returning a stable 12-char hex over any frozen dataclass; `require_env(name: str) -> str` raising `RuntimeError` if unset.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest
from mt5gold.config import DataConfig, config_hash, require_env


def test_dataconfig_defaults_are_xauusd_m1():
    cfg = DataConfig()
    assert cfg.symbol == "XAUUSD"
    assert cfg.base_timeframe == "M1"


def test_config_hash_is_stable_and_sensitive():
    a = DataConfig(symbol="XAUUSD")
    b = DataConfig(symbol="XAUUSD")
    c = DataConfig(symbol="EURUSD")
    assert config_hash(a) == config_hash(b)          # deterministic
    assert config_hash(a) != config_hash(c)          # sensitive to fields
    assert len(config_hash(a)) == 12


def test_require_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MT5GOLD_TEST_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="MT5GOLD_TEST_SECRET"):
        require_env("MT5GOLD_TEST_SECRET")


def test_require_env_returns_value(monkeypatch):
    monkeypatch.setenv("MT5GOLD_TEST_SECRET", "abc")
    assert require_env("MT5GOLD_TEST_SECRET") == "abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mt5gold'`

- [ ] **Step 3: Write minimal implementation**

Create the empty `__init__.py` files listed above. Then:

```python
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
```

```python
# mt5gold/config.py
"""Typed configuration. SHARED configs must be byte-identical across
backtest/train/live; LIVE-only configs never affect a backtest (spec finding 31)."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """SHARED/pinned. Governs which data is fetched and how it is keyed."""
    symbol: str = "XAUUSD"
    base_timeframe: str = "M1"
    history_start: str = "2020-01-01"      # ISO date, UTC
    history_end: str | None = None          # None = up to now
    broker_tz_offset_hours: int | None = None  # detected at fetch time, then pinned


def config_hash(obj) -> str:
    """Stable 12-char hex hash over a frozen dataclass's fields."""
    payload = json.dumps(dataclasses.asdict(obj), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def require_env(name: str) -> str:
    """Read a required environment variable or fail fast (never log the value)."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add mt5gold/ tests/ pytest.ini
git commit -m "feat(phase0): package scaffold + typed config with stable hashing"
```

---

### Task 2: `broker.py` — the only MetaTrader5 boundary

**Files:**
- Create: `mt5gold/live/broker.py`
- Create: `tests/fakes.py`
- Create: `tests/test_broker.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TIMEFRAME_MINUTES: dict[str, int]` mapping `"M1"→1, "M5"→5, "M15"→15, "H1"→60`.
  - `class Broker(Protocol)` with methods: `copy_rates_range(symbol: str, timeframe: str, start: datetime, end: datetime) -> np.ndarray`, `symbol_info(symbol: str) -> dict`, `account_info() -> dict`, `connect() -> None`, `shutdown() -> None`.
  - `class Mt5Broker` implementing the protocol against the real library (imports MetaTrader5 lazily inside methods).
  - `tests/fakes.py`: `FakeBroker` returning deterministic synthetic rates for tests, plus `make_rates(start, n, minutes=1, spread=20)` helper producing an MT5-shaped structured array.

- [ ] **Step 1: Write the failing test**

```python
# tests/fakes.py
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np

RATE_DTYPE = np.dtype([
    ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
    ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8"),
])


def make_rates(start: datetime, n: int, minutes: int = 1, spread: int = 20) -> np.ndarray:
    """Build an MT5-shaped bid-based structured array of n bars."""
    rows = []
    base = int(start.replace(tzinfo=timezone.utc).timestamp())
    price = 2000.0
    for i in range(n):
        o = price
        h = o + 0.5
        l = o - 0.5
        c = o + 0.2
        rows.append((base + i * minutes * 60, o, h, l, c, 100 + i, spread, 0))
        price = c
    return np.array(rows, dtype=RATE_DTYPE)


class FakeBroker:
    def __init__(self, rates: np.ndarray, tz_offset_hours: int = 0):
        self._rates = rates
        self._tz_offset_hours = tz_offset_hours
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def shutdown(self) -> None:
        self.connected = False

    def copy_rates_range(self, symbol, timeframe, start, end):
        s = int(start.timestamp())
        e = int(end.timestamp())
        return self._rates[(self._rates["time"] >= s) & (self._rates["time"] <= e)]

    def symbol_info(self, symbol):
        return {
            "point": 0.01, "digits": 2, "contract_size": 100.0,
            "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
            "trade_stops_level": 0, "trade_freeze_level": 0,
            "swap_long": -3.0, "swap_short": -1.0, "swap_mode": 1,
            "trade_tick_value": 1.0, "trade_tick_size": 0.01,
        }

    def account_info(self):
        # server_tz lets tests exercise offset detection
        return {"login": 1, "currency": "USD", "trade_mode": 0,
                "server_tz_offset_hours": self._tz_offset_hours}
```

```python
# tests/test_broker.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_broker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mt5gold.live.broker'`

- [ ] **Step 3: Write minimal implementation**

```python
# mt5gold/live/broker.py
"""The ONLY module that talks to MetaTrader5 (spec Principle 1 / finding 30).
Everything else depends on the Broker protocol so tests can inject a fake."""
from __future__ import annotations
from datetime import datetime
from typing import Protocol, runtime_checkable
import numpy as np

TIMEFRAME_MINUTES: dict[str, int] = {"M1": 1, "M5": 5, "M15": 15, "H1": 60}


@runtime_checkable
class Broker(Protocol):
    def connect(self) -> None: ...
    def shutdown(self) -> None: ...
    def copy_rates_range(self, symbol: str, timeframe: str,
                         start: datetime, end: datetime) -> np.ndarray: ...
    def symbol_info(self, symbol: str) -> dict: ...
    def account_info(self) -> dict: ...


class Mt5Broker:
    """Real implementation. Imports MetaTrader5 lazily so importing this
    module (and running tests) does not require the library/terminal."""

    def __init__(self, reconnect_wait: int = 5, max_reconnect: int = 10):
        self.reconnect_wait = reconnect_wait
        self.max_reconnect = max_reconnect

    def _mt5(self):
        import MetaTrader5 as mt5
        return mt5

    def connect(self) -> None:
        import time
        mt5 = self._mt5()
        for attempt in range(1, self.max_reconnect + 1):
            if mt5.initialize():
                return
            time.sleep(self.reconnect_wait)
        raise RuntimeError("Cannot connect to MetaTrader 5 terminal")

    def shutdown(self) -> None:
        self._mt5().shutdown()

    def copy_rates_range(self, symbol, timeframe, start, end):
        mt5 = self._mt5()
        tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None:
            return np.empty(0)
        return rates

    def symbol_info(self, symbol):
        info = self._mt5().symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info returned None for {symbol}")
        return info._asdict()

    def account_info(self):
        acc = self._mt5().account_info()
        if acc is None:
            raise RuntimeError("account_info returned None")
        return acc._asdict()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_broker.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add mt5gold/live/broker.py tests/fakes.py tests/test_broker.py
git commit -m "feat(phase0): broker.py MT5 boundary + FakeBroker for tests"
```

---

### Task 3: `data/fetch.py` — chunked M1 fetch

**Files:**
- Create: `mt5gold/data/fetch.py`
- Create: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `Broker` (Task 2), `TIMEFRAME_MINUTES`.
- Produces: `fetch_rates(broker, symbol, timeframe, start, end, chunk_days=30) -> pd.DataFrame` returning columns `["time","open","high","low","close","tick_volume","spread","real_volume"]` with `time` as tz-aware UTC datetime, sorted ascending, de-duplicated on `time`. Empty input → empty DataFrame with those columns.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mt5gold.data.fetch'`

- [ ] **Step 3: Write minimal implementation**

```python
# mt5gold/data/fetch.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fetch.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add mt5gold/data/fetch.py tests/test_fetch.py
git commit -m "feat(phase0): chunked M1 fetch returning sorted unique UTC bars"
```

---

### Task 4: `data/clean.py` — tz normalization, gaps, spread, validation

**Files:**
- Create: `mt5gold/data/clean.py`
- Create: `tests/test_clean.py`

**Interfaces:**
- Consumes: DataFrame from `fetch_rates` (Task 3).
- Produces:
  - `validate_rates(df) -> None` raising `DataValidationError` on: unsorted/duplicate time, `high < max(open,close)`, `low > min(open,close)`, missing/non-positive `spread`.
  - `class DataValidationError(Exception)`.
  - `clean_rates(df, broker_tz_offset_hours: int) -> pd.DataFrame`: shifts broker time to UTC using the offset, validates, flags anomalous spread in a boolean `spread_anomaly` column (median-absolute-deviation rule), returns validated frame. Weekend/holiday bars are left as-is (no phantom-bar creation); this function must NOT reindex to a continuous range.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clean.py
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


def test_clean_flags_spread_spike():
    df = _df(n=50, spread=20)
    df.loc[25, "spread"] = 900                        # news-time blowout
    cleaned = clean_rates(df, broker_tz_offset_hours=0)
    assert bool(cleaned.loc[25, "spread_anomaly"]) is True
    assert cleaned["spread_anomaly"].sum() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mt5gold.data.clean'`

- [ ] **Step 3: Write minimal implementation**

```python
# mt5gold/data/clean.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_clean.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add mt5gold/data/clean.py tests/test_clean.py
git commit -m "feat(phase0): clean.py tz-to-UTC, integrity validation, spread anomaly flag"
```

---

### Task 5: `data/store.py` — Parquet + hashed manifest with contract snapshot

**Files:**
- Create: `mt5gold/data/store.py`
- Create: `tests/test_store.py`

**Interfaces:**
- Consumes: cleaned DataFrame (Task 4), `Broker.symbol_info`/`account_info` (Task 2), `config_hash` (Task 1).
- Produces:
  - `contract_snapshot(broker, symbol) -> dict` capturing spec fields (spec §4.3 / finding 43): `contract_size, tick_size, tick_value, volume_min, volume_max, volume_step, trade_stops_level, trade_freeze_level, filling_mode, swap_long, swap_short, swap_mode, account_currency`.
  - `write_dataset(df, root, symbol, timeframe, contract, tz_offset_hours) -> Path` writing `root/clean/{symbol}/{timeframe}/data.parquet` + sibling `manifest.json` containing `{symbol, timeframe, rows, start, end, tz_offset_hours, data_hash, contract, created_utc}`. `created_utc` is passed in (not read from the clock) so runs are reproducible.
  - `read_dataset(root, symbol, timeframe) -> tuple[pd.DataFrame, dict]`.
  - `dataframe_hash(df) -> str` (stable 12-char hex over content).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
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
    assert snap["contract_size"] == 100.0
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mt5gold.data.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# mt5gold/data/store.py
"""Persist cleaned data as Parquet + a hashed JSON manifest (spec §4.3)."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

_CONTRACT_FIELDS = {
    "contract_size": "contract_size", "tick_size": "trade_tick_size",
    "tick_value": "trade_tick_value", "volume_min": "volume_min",
    "volume_max": "volume_max", "volume_step": "volume_step",
    "trade_stops_level": "trade_stops_level", "trade_freeze_level": "trade_freeze_level",
    "swap_long": "swap_long", "swap_short": "swap_short", "swap_mode": "swap_mode",
}


def contract_snapshot(broker, symbol: str) -> dict:
    info = broker.symbol_info(symbol)
    acc = broker.account_info()
    snap = {out: info.get(src) for out, src in _CONTRACT_FIELDS.items()}
    snap["filling_mode"] = info.get("filling_mode")
    snap["account_currency"] = acc.get("currency")
    return snap


def dataframe_hash(df: pd.DataFrame) -> str:
    content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(content).hexdigest()[:12]


def _dir(root, symbol, timeframe) -> Path:
    return Path(root) / "clean" / symbol / timeframe


def write_dataset(df: pd.DataFrame, root, symbol: str, timeframe: str,
                  contract: dict, tz_offset_hours: int,
                  created_utc: datetime) -> Path:
    out_dir = _dir(root, symbol, timeframe)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.parquet"
    df.to_parquet(data_path, index=False)
    manifest = {
        "symbol": symbol, "timeframe": timeframe, "rows": int(len(df)),
        "start": df["time"].iloc[0].isoformat() if len(df) else None,
        "end": df["time"].iloc[-1].isoformat() if len(df) else None,
        "tz_offset_hours": tz_offset_hours,
        "data_hash": dataframe_hash(df),
        "contract": contract,
        "created_utc": created_utc.isoformat(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return data_path


def read_dataset(root, symbol: str, timeframe: str) -> tuple[pd.DataFrame, dict]:
    out_dir = _dir(root, symbol, timeframe)
    df = pd.read_parquet(out_dir / "data.parquet")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return df, manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add mt5gold/data/store.py tests/test_store.py
git commit -m "feat(phase0): parquet store + hashed manifest with contract snapshot"
```

---

### Task 6: `scripts/fetch_data.py` — end-to-end entrypoint + integration test

**Files:**
- Create: `mt5gold/data/pipeline.py`
- Create: `scripts/fetch_data.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_ingest(broker, cfg: DataConfig, root, created_utc, detect_offset=True) -> Path` that: detects the broker tz offset from `account_info()["server_tz_offset_hours"]` (when `detect_offset`), fetches → cleans → writes, and returns the dataset path. `scripts/fetch_data.py` is a thin `__main__` wrapper constructing `Mt5Broker` + `DataConfig` from env/args and calling `run_ingest`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mt5gold.data.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# mt5gold/data/pipeline.py
"""End-to-end ingest: detect offset -> fetch -> clean -> store."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from mt5gold.config import DataConfig
from mt5gold.data.fetch import fetch_rates
from mt5gold.data.clean import clean_rates
from mt5gold.data.store import write_dataset, contract_snapshot


def run_ingest(broker, cfg: DataConfig, root, created_utc: datetime,
               detect_offset: bool = True) -> Path:
    broker.connect()
    try:
        offset = cfg.broker_tz_offset_hours or 0
        if detect_offset:
            offset = int(broker.account_info().get("server_tz_offset_hours", offset))

        start = pd.Timestamp(cfg.history_start, tz="UTC").to_pydatetime()
        end = (pd.Timestamp(cfg.history_end, tz="UTC").to_pydatetime()
               if cfg.history_end else created_utc)

        raw = fetch_rates(broker, cfg.symbol, cfg.base_timeframe, start, end)
        cleaned = clean_rates(raw, broker_tz_offset_hours=offset)
        contract = contract_snapshot(broker, cfg.symbol)
        return write_dataset(cleaned, root, cfg.symbol, cfg.base_timeframe,
                             contract, tz_offset_hours=offset, created_utc=created_utc)
    finally:
        broker.shutdown()
```

```python
# scripts/fetch_data.py
"""CLI entrypoint: python scripts/fetch_data.py [--root data] [--start 2020-01-01]"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from mt5gold.config import DataConfig
from mt5gold.data.pipeline import run_ingest
from mt5gold.live.broker import Mt5Broker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    cfg = DataConfig(symbol=args.symbol, base_timeframe="M1",
                     history_start=args.start, history_end=args.end)
    path = run_ingest(Mt5Broker(), cfg, args.root,
                      created_utc=datetime.now(timezone.utc))
    print(f"Wrote dataset: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the whole suite and commit**

Run: `python -m pytest -v`
Expected: PASS (all Phase 0 tests green)

```bash
git add mt5gold/data/pipeline.py scripts/fetch_data.py tests/test_pipeline.py
git commit -m "feat(phase0): end-to-end ingest pipeline + fetch_data CLI"
```

---

## Phase 0 Milestone (from spec §11)

Running `python scripts/fetch_data.py` against a live MT5 demo terminal produces a partitioned Parquet dataset with a `manifest.json` carrying a content hash, detected tz offset, and broker contract snapshot — a **reproducible source of truth** that every later phase reads. All logic is covered by tests that run without a live terminal (via `FakeBroker`).

## Self-Review

- **Spec coverage (Phase 0 rows):** scaffold ✓ (T1), config + secrets via env ✓ (T1), `broker.py` isolation ✓ (T2, finding 30), M1 fetch ✓ (T3), tz→UTC + gap-safe + spread column + validation ✓ (T4, §4.2), Parquet + manifest + hash + contract snapshot ✓ (T5, §4.3 / finding 43), reproducibility (created_utc injected, deterministic hash) ✓ (T5/T6, Principle 5).
- **Placeholder scan:** none — every step ships runnable code and exact commands.
- **Type consistency:** `Broker` protocol methods (T2) match `FakeBroker`/`Mt5Broker`; `fetch_rates` columns (T3) feed `clean_rates` (T4) feed `write_dataset` (T5) feed `run_ingest` (T6); `contract_snapshot`/`dataframe_hash`/`config_hash` names are stable across tasks.
- **Deferred to later phases (correctly out of Phase 0 scope):** resample M1→M5/M15/H1 (Phase 1 needs it for the engine), `core/types.py` Signal/StrategyState (Phase 1–2), shared-config hash binding into model/live (Phase 2–3). Noted so a reader does not expect them here.
