from datetime import datetime, timezone
import pytest
from mt5gold.live.safety import (SafetyConfig, assert_demo, spread_ok,
    daily_halt, drawdown_halt, should_force_flat)
from mt5gold.core.types import StrategyState

UTC = timezone.utc
CFG = SafetyConfig(max_spread_points=45, daily_loss_limit=-100.0, daily_profit_target=100.0,
                   max_drawdown_pct=0.15, weekend_flat=True, friday_close_utc_hour=21)


class _Acc:
    def __init__(self, mode): self.mode = mode
    def account_info(self): return {"trade_mode": self.mode}


def test_assert_demo_blocks_live():
    with pytest.raises(RuntimeError): assert_demo(_Acc(1))   # 0=demo,1=contest/real
    assert_demo(_Acc(0))                                     # no raise


def test_spread_guard(): assert spread_ok(40, CFG) and not spread_ok(50, CFG)


def test_daily_halt_on_loss():
    st = StrategyState(None, 0, True, 0, datetime(2020, 1, 1, tzinfo=UTC), 1000, 1000, 3, -120.0)
    assert daily_halt(st, CFG)


def test_drawdown_halt(): assert drawdown_halt(830, 1000, CFG) and not drawdown_halt(900, 1000, CFG)


def test_force_flat_friday_late():
    assert should_force_flat(datetime(2020, 1, 3, 21, 30, tzinfo=UTC), CFG)   # Fri 21:30
    assert not should_force_flat(datetime(2020, 1, 1, 12, 0, tzinfo=UTC), CFG)  # Wed
