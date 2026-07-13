from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecConfig:
    deviation: int = 20
    magic: int = 20260704
    comment: str = "MT5GOLD"


def build_order_request(broker, symbol, signal, lot, spec, cfg: ExecConfig) -> dict:
    tick = broker.symbol_info_tick(symbol)
    info = broker.symbol_info(symbol)
    digits = info["digits"]
    price = tick["ask"] if signal.side == "BUY" else tick["bid"]
    return {
        "symbol": symbol, "type": signal.side, "volume": lot, "price": price,
        "sl": round(signal.sl_price, digits), "tp": round(signal.tp_price, digits),
        "deviation": cfg.deviation, "magic": cfg.magic, "comment": cfg.comment,
        "type_filling": info["filling_mode"],
    }


def place_order(broker, request) -> dict:
    return broker.order_send(request)
