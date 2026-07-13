from datetime import datetime, timezone
import pandas as pd
from mt5gold.core.types import StrategyState
from mt5gold.core.strategy import RuleBasedStrategy, LegacyReconstructionStrategy, StrategyConfig

UTC = timezone.utc


def _state(): return StrategyState(None, 0, True, 300, datetime(2020, 1, 1, tzinfo=UTC), 1000, 1000, 0, 0.0)


def _row(**kw):
    base = {"ema9": 2010, "ema21": 2005, "ema50": 2000, "rsi14": 60, "atr14": 2.0,
            "atr_pctile": 0.5, "swing_high": 2015, "swing_low": 1990, "close": 2011}
    base.update(kw)
    return pd.Series(base)


def test_rulebased_returns_flat_when_no_edge():
    s = RuleBasedStrategy(StrategyConfig())
    sig = s.generate_signal(_row(ema9=2000, ema21=2001, rsi14=50, close=2000), _state())
    assert sig.side == "FLAT"


def test_rulebased_buy_on_uptrend_momentum():
    s = RuleBasedStrategy(StrategyConfig())
    sig = s.generate_signal(_row(), _state())
    assert sig.side == "BUY"
    assert sig.sl_price is not None and sig.tp_price is not None


def test_rulebased_does_not_trade_when_position_open():
    s = RuleBasedStrategy(StrategyConfig())
    from mt5gold.core.types import Position
    st = StrategyState(Position("BUY", 2000, 1990, 2020, 0.1, datetime(2020, 1, 1, tzinfo=UTC)),
                       1, True, 300, datetime(2020, 1, 1, tzinfo=UTC), 1000, 1000, 1, 0.0)
    assert s.generate_signal(_row(), st).side == "FLAT"


def test_legacy_reconstruction_always_trades_when_flat():
    # B0 must reproduce the "enter every candle" behavior: never FLAT when no position
    s = LegacyReconstructionStrategy(StrategyConfig())
    sig = s.generate_signal(_row(ema9=2000, ema21=2001, rsi14=50, close=2000), _state())
    assert sig.side in ("BUY", "SELL")
