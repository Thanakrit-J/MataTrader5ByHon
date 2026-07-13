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
        # Field names mirror MT5's real symbol_info (e.g. trade_contract_size,
        # not contract_size) so the fake exercises the same mapping as production.
        return {
            "point": 0.01, "digits": 2, "trade_contract_size": 100.0,
            "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
            "trade_stops_level": 0, "trade_freeze_level": 0,
            "swap_long": -3.0, "swap_short": -1.0, "swap_mode": 1,
            "trade_tick_value": 1.0, "trade_tick_size": 0.01,
        }

    def account_info(self):
        # server_tz lets tests exercise offset detection
        return {"login": 1, "currency": "USD", "trade_mode": 0,
                "server_tz_offset_hours": self._tz_offset_hours}
