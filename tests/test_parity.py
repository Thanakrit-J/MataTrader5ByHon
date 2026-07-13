from datetime import datetime, timezone
import numpy as np, pandas as pd
from mt5gold.backtest.replay import replay_live_signals, backtest_signals
from mt5gold.core.features import build_features, WARMUP_BARS
from mt5gold.core.strategy import RuleBasedStrategy, StrategyConfig
from mt5gold.live.bot import LiveBot
from mt5gold.live.safety import SafetyConfig
from mt5gold.live.execution import ExecConfig
from mt5gold.core.costs import CostConfig

UTC = timezone.utc
SPEC = {"point": 0.01, "contract_size": 100.0, "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
        "trade_stops_level": 0, "trade_freeze_level": 0, "swap_long": -3.0, "swap_short": -1.0,
        "tick_value": 1.0, "tick_size": 0.01, "digits": 2, "filling_mode": 1}


class FB:
    def account_info(self): return {"trade_mode": 0, "balance": 1000.0, "equity": 1000.0, "currency": "USD"}
    def positions_get(self, symbol=None): return []


def _price(n=WARMUP_BARS + 60):   # must exceed WARMUP_BARS so the comparison is non-empty
    idx = pd.date_range(datetime(2020, 1, 1, tzinfo=UTC), periods=n, freq="5min", tz="UTC")
    c = 2000 + np.cumsum(np.random.default_rng(3).normal(0, 1, n))
    return pd.DataFrame({"time": idx, "open": c, "high": c + 1, "low": c - 1, "close": c, "tick_volume": 1, "spread": 20.0})


def test_live_and_backtest_signals_match_after_warmup():
    price = _price()
    strat = RuleBasedStrategy(StrategyConfig())
    feats = build_features(price); feats["close"] = price["close"].values
    bt = backtest_signals(strat, feats, price)
    import tempfile, os
    bot = LiveBot(FB(), strat, SPEC, CostConfig(), ExecConfig(), SafetyConfig(weekend_flat=False),
                  os.path.join(tempfile.mkdtemp(), "j.jsonl"))
    live = replay_live_signals(bot, price)
    # compare from WARMUP_BARS onward
    for i in range(WARMUP_BARS + 5, len(price)):
        assert bt[i].side == live[i].side, i
