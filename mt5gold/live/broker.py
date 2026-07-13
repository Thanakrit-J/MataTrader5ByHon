"""The ONLY module that talks to MetaTrader5 (spec Principle 1 / finding 30).
Everything else depends on the Broker protocol so tests can inject a fake."""
from __future__ import annotations
from datetime import datetime, timezone
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
    # Live-trading surface (data-only consumers such as the Phase-0 pipeline
    # and FakeBroker do not need these; structural typing lets them omit them).
    def symbol_info_tick(self, symbol: str) -> dict: ...
    def positions_get(self, symbol: str | None = None) -> list[dict]: ...
    def order_send(self, request: dict) -> dict: ...


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

    def symbol_info_tick(self, symbol):
        tick = self._mt5().symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick returned None for {symbol}")
        return tick._asdict()

    def positions_get(self, symbol=None):
        mt5 = self._mt5()
        ps = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not ps:
            return []
        out = []
        for p in ps:
            d = p._asdict()
            out.append({
                "type": "BUY" if d.get("type") == mt5.POSITION_TYPE_BUY else "SELL",
                "entry_price": d.get("price_open"),
                "sl": d.get("sl"), "tp": d.get("tp"),
                "lot": d.get("volume"),
                "entry_time": datetime.fromtimestamp(d.get("time", 0), tz=timezone.utc),
            })
        return out

    def order_send(self, request):
        mt5 = self._mt5()
        order_type = mt5.ORDER_TYPE_BUY if request["type"] == "BUY" else mt5.ORDER_TYPE_SELL
        mt5_req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request["symbol"],
            "volume": float(request["volume"]),
            "type": order_type,
            "price": request["price"],
            "sl": request["sl"],
            "tp": request["tp"],
            "deviation": request["deviation"],
            "magic": request["magic"],
            "comment": request["comment"],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": request["type_filling"],
        }
        result = mt5.order_send(mt5_req)
        if result is None:
            return {"ok": False, "retcode": None, "ticket": None}
        return {"ok": result.retcode == mt5.TRADE_RETCODE_DONE,
                "retcode": result.retcode, "ticket": getattr(result, "order", None)}
