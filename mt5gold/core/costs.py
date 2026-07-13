from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class CostConfig:
    commission_per_lot: float = 0.0
    slippage_base_points: float = 5.0
    slippage_atr_mult: float = 2.0
    slippage_exit_mult: float = 3.0


def _round_step(lot, step):
    return round(round(lot / step) * step, 2)


def position_size(balance, risk_pct, sl_distance_price, spec) -> float:
    if sl_distance_price <= 0:
        return spec["volume_min"]
    money_risk = balance * risk_pct
    money_per_lot = sl_distance_price * spec["contract_size"]
    lot = money_risk / money_per_lot if money_per_lot > 0 else spec["volume_min"]
    lot = _round_step(lot, spec["volume_step"])
    return max(spec["volume_min"], min(lot, spec["volume_max"]))


def entry_fill(open_price, side, spread_points, spec) -> float:
    return open_price + spread_points * spec["point"] if side == "BUY" else open_price


def exit_slippage_points(atr_percentile, spread_points, cfg: CostConfig, is_stop: bool) -> float:
    base = cfg.slippage_base_points + cfg.slippage_atr_mult * atr_percentile + 0.1 * spread_points
    return base * (cfg.slippage_exit_mult if is_stop else 1.0)


def swap_cost(side, lots, entry_time: datetime, exit_time: datetime, spec) -> float:
    """Charge swap per rollover (00:00 UTC boundary) crossed; Wednesday = triple."""
    rate = spec["swap_long"] if side == "BUY" else spec["swap_short"]
    total = 0.0
    day = entry_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while day <= exit_time:
        mult = 3 if day.weekday() == 2 else 1   # Wed = triple swap
        total += rate * lots * mult
        day += timedelta(days=1)
    return total


def enforce_min_stop_distance(entry, sl, tp, side, spec):
    min_dist = max(spec.get("trade_stops_level", 0), spec.get("trade_freeze_level", 0)) * spec["point"]
    if abs(entry - sl) < min_dist or abs(tp - entry) < min_dist:
        return sl, tp, "skip"
    return sl, tp, "ok"
