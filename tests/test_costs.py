from datetime import datetime, timezone
from mt5gold.core.costs import (CostConfig, position_size, entry_fill,
    exit_slippage_points, swap_cost, enforce_min_stop_distance)

UTC = timezone.utc
SPEC = {"point": 0.01, "contract_size": 100.0, "tick_value": 1.0, "tick_size": 0.01,
        "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
        "trade_stops_level": 50, "trade_freeze_level": 0,
        "swap_long": -3.0, "swap_short": -1.0, "swap_mode": 1}


def test_position_size_respects_risk_and_step():
    # risk 1% of $1000 = $10; SL distance $2 on gold: $/lot per $1 move = contract_size=100
    # money per lot at SL = 2 * 100 = $200 → lot = 10/200 = 0.05
    lot = position_size(1000.0, 0.01, 2.0, SPEC)
    assert abs(lot - 0.05) < 1e-9


def test_entry_fill_buy_pays_spread_sell_does_not():
    assert abs(entry_fill(2000.0, "BUY", 20, SPEC) - (2000.0 + 20 * 0.01)) < 1e-9
    assert entry_fill(2000.0, "SELL", 20, SPEC) == 2000.0


def test_exit_slippage_larger_for_stops():
    cfg = CostConfig(commission_per_lot=0.0, slippage_base_points=5,
                     slippage_atr_mult=2.0, slippage_exit_mult=3.0)
    entry_sl = exit_slippage_points(0.9, 20, cfg, is_stop=True)
    entry_no = exit_slippage_points(0.9, 20, cfg, is_stop=False)
    assert entry_sl > entry_no


def test_swap_charges_per_night_and_triple_wednesday():
    # hold Tue 22:00 -> Thu 02:00 crosses Tue->Wed (triple) + Wed->Thu nights
    entry = datetime(2020, 1, 7, 22, 0, tzinfo=UTC)   # Tue
    exit_ = datetime(2020, 1, 9, 2, 0, tzinfo=UTC)    # Thu
    cost = swap_cost("BUY", 1.0, entry, exit_, SPEC)
    assert cost < 0                               # negative swap accrues loss


def test_min_stop_distance_skips_when_too_tight():
    # stops_level 50 points * 0.01 = 0.5 price; SL only 0.2 away -> skip
    sl, tp, action = enforce_min_stop_distance(2000.0, 1999.8, 2001.0, "BUY", SPEC)
    assert action == "skip"
    sl, tp, action = enforce_min_stop_distance(2000.0, 1998.0, 2004.0, "BUY", SPEC)
    assert action == "ok"
