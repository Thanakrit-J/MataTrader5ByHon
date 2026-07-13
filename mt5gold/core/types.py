from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Side = Literal["BUY", "SELL", "FLAT"]


@dataclass(frozen=True)
class Signal:
    side: Side
    confidence: float
    sl_price: float | None
    tp_price: float | None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Position:
    side: Side
    entry_price: float
    sl: float
    tp: float
    lot: float
    entry_time: datetime


@dataclass(frozen=True)
class Trade:
    side: Side
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    lot: float
    pnl: float
    exit_reason: str
    costs: float


@dataclass(frozen=True)
class StrategyState:
    open_position: Position | None
    bars_held: int
    warmup_ready: bool
    bar_index: int
    bar_time: datetime
    balance: float
    equity: float
    trades_today: int
    daily_pnl: float
