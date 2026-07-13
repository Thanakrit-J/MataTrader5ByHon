from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
from mt5gold.backtest.engine import run_backtest, BacktestConfig
from mt5gold.core.costs import CostConfig
from mt5gold.core.types import Signal

UTC = timezone.utc
SPEC = {"point": 0.01, "contract_size": 100.0, "tick_value": 1.0, "tick_size": 0.01,
        "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
        "trade_stops_level": 0, "trade_freeze_level": 0, "swap_long": -3.0, "swap_short": -1.0, "swap_mode": 1}


class AlwaysBuyOnce:
    def __init__(self): self.fired = False
    def generate_signal(self, f, state):
        if state.open_position is None and not self.fired:
            self.fired = True
            return Signal("BUY", 1.0, sl_price=1990.0, tp_price=2010.0, meta={})
        return Signal("FLAT", 0.0, None, None, {})


def _prices(n=10):
    idx = pd.date_range(datetime(2020, 1, 1, tzinfo=UTC), periods=n, freq="5min", tz="UTC")
    # flat then a jump up to hit TP at bar 3
    close = np.array([2000, 2000, 2000, 2011, 2011, 2011, 2011, 2011, 2011, 2011], float)
    return pd.DataFrame({"time": idx, "open": close, "high": close + 1, "low": close - 1,
                         "close": close, "tick_volume": 1, "spread": 20.0})


def test_engine_opens_and_closes_on_tp():
    price = _prices()
    feats = pd.DataFrame({"atr14": [2.0] * len(price), "close": price["close"]})
    trades = run_backtest(AlwaysBuyOnce(), feats, price, SPEC,
                          CostConfig(commission_per_lot=0.0),
                          BacktestConfig(risk_pct=0.01, starting_balance=1000.0))
    assert len(trades) == 1
    assert trades[0].side == "BUY"
    assert trades[0].exit_reason == "TP"
    assert trades[0].pnl > 0


def test_engine_no_trade_when_strategy_flat():
    price = _prices()
    feats = pd.DataFrame({"atr14": [2.0] * len(price), "close": price["close"]})
    class Flat:
        def generate_signal(self, f, s): return Signal("FLAT", 0.0, None, None, {})
    assert run_backtest(Flat(), feats, price, SPEC, CostConfig(),
                        BacktestConfig(0.01, 1000.0)) == []
