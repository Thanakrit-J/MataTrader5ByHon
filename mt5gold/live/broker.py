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
