from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SafetyConfig:
    max_spread_points: float = 45
    daily_loss_limit: float = -100.0
    daily_profit_target: float = 100.0
    max_drawdown_pct: float = 0.15
    weekend_flat: bool = True
    friday_close_utc_hour: int = 21


def assert_demo(broker) -> None:
    if broker.account_info().get("trade_mode", 1) != 0:
        raise RuntimeError("LIVE account detected — this bot runs on DEMO only")


def spread_ok(spread_points, cfg: SafetyConfig) -> bool:
    return spread_points <= cfg.max_spread_points


def daily_halt(state, cfg: SafetyConfig) -> bool:
    return state.daily_pnl <= cfg.daily_loss_limit or state.daily_pnl >= cfg.daily_profit_target


def drawdown_halt(equity, peak_equity, cfg: SafetyConfig) -> bool:
    if peak_equity <= 0:
        return False
    return (equity - peak_equity) / peak_equity <= -cfg.max_drawdown_pct


def should_force_flat(bar_time: datetime, cfg: SafetyConfig) -> bool:
    return cfg.weekend_flat and bar_time.weekday() == 4 and bar_time.hour >= cfg.friday_close_utc_hour
