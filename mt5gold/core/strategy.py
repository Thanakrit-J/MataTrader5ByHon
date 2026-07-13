from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import pandas as pd
from mt5gold.core.types import Signal, StrategyState


@dataclass(frozen=True)
class StrategyConfig:
    k_tp: float = 3.0
    k_sl: float = 1.0
    atr_window: int = 14
    rsi_buy: float = 55.0
    rsi_sell: float = 45.0


class Strategy(Protocol):
    def generate_signal(self, features_row: pd.Series, state: StrategyState) -> Signal: ...


def _levels(side, price, atr, cfg):
    if side == "BUY":
        return price - cfg.k_sl * atr, price + cfg.k_tp * atr
    return price + cfg.k_sl * atr, price - cfg.k_tp * atr


def _flat():
    return Signal("FLAT", 0.0, None, None, {})


class RuleBasedStrategy:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def generate_signal(self, f, state) -> Signal:
        if state.open_position is not None or not state.warmup_ready:
            return _flat()
        price, atr = float(f["close"]), float(f["atr14"])
        up = f["ema9"] > f["ema21"] > f["ema50"] and f["rsi14"] >= self.cfg.rsi_buy
        dn = f["ema9"] < f["ema21"] < f["ema50"] and f["rsi14"] <= self.cfg.rsi_sell
        if up:
            sl, tp = _levels("BUY", price, atr, self.cfg)
            return Signal("BUY", 1.0, sl, tp, {"r": "trend_up"})
        if dn:
            sl, tp = _levels("SELL", price, atr, self.cfg)
            return Signal("SELL", 1.0, sl, tp, {"r": "trend_dn"})
        return _flat()


class LegacyReconstructionStrategy:
    """B0: reproduces legacy zone -> EMA/RSI -> smart-guess-every-candle fallback."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def generate_signal(self, f, state) -> Signal:
        if state.open_position is not None or not state.warmup_ready:
            return _flat()
        price, atr = float(f["close"]), float(f["atr14"])
        up = f["ema9"] > f["ema21"] and f["rsi14"] > 50
        side = "BUY" if up else "SELL"        # smart-guess: always picks a side
        sl, tp = _levels(side, price, atr, self.cfg)
        return Signal(side, 0.5, sl, tp, {"r": "legacy_guess"})
