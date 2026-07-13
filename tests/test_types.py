from datetime import datetime, timezone
from mt5gold.core.types import Signal, Position, StrategyState


def test_signal_flat_needs_no_levels():
    s = Signal(side="FLAT", confidence=0.0, sl_price=None, tp_price=None, meta={})
    assert s.side == "FLAT"


def test_signal_is_frozen():
    s = Signal(side="BUY", confidence=1.0, sl_price=1.0, tp_price=2.0, meta={})
    try:
        s.side = "SELL"          # type: ignore
        assert False, "should be frozen"
    except Exception:
        pass


def test_state_carries_position():
    pos = Position("BUY", 2000.0, 1990.0, 2020.0, 0.1, datetime(2020, 1, 1, tzinfo=timezone.utc))
    st = StrategyState(open_position=pos, bars_held=3, warmup_ready=True, bar_index=100,
                       bar_time=datetime(2020, 1, 1, tzinfo=timezone.utc), balance=1000.0,
                       equity=1010.0, trades_today=1, daily_pnl=10.0)
    assert st.open_position.side == "BUY" and st.bars_held == 3
