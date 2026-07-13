from mt5gold.live.execution import ExecConfig, build_order_request
from mt5gold.core.types import Signal


class FB:
    def symbol_info(self, s): return {"filling_mode": 1, "digits": 2, "point": 0.01}
    def symbol_info_tick(self, s): return {"ask": 2000.50, "bid": 2000.30}


def test_build_request_buy_uses_ask_and_filling_mode():
    sig = Signal("BUY", 1.0, 1990.0, 2020.0, {})
    req = build_order_request(FB(), "XAUUSD", sig, 0.1, {"digits": 2, "point": 0.01},
                              ExecConfig(deviation=20, magic=42, comment="MLBOT"))
    assert req["type"] == "BUY" and req["price"] == 2000.50
    assert req["type_filling"] == 1 and req["deviation"] == 20 and req["magic"] == 42
    assert req["sl"] == 1990.0 and req["tp"] == 2020.0 and req["volume"] == 0.1
