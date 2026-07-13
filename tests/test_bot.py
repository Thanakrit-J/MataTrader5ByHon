from datetime import datetime, timezone
import numpy as np, pandas as pd
from mt5gold.live.bot import LiveBot
from mt5gold.core.features import WARMUP_BARS
from mt5gold.core.strategy import RuleBasedStrategy, StrategyConfig
from mt5gold.live.safety import SafetyConfig
from mt5gold.live.execution import ExecConfig
from mt5gold.core.costs import CostConfig

UTC = timezone.utc
SPEC = {"point": 0.01, "contract_size": 100.0, "tick_value": 1.0, "tick_size": 0.01, "volume_min": 0.01,
        "volume_max": 50.0, "volume_step": 0.01, "trade_stops_level": 0, "trade_freeze_level": 0,
        "swap_long": -3.0, "swap_short": -1.0, "swap_mode": 1, "digits": 2, "filling_mode": 1}


class FB:
    def account_info(self): return {"trade_mode": 0, "balance": 1000.0, "equity": 1000.0, "currency": "USD"}
    def positions_get(self, symbol=None): return []


def _buf(n):
    idx = pd.date_range(datetime(2020, 1, 1, tzinfo=UTC), periods=n, freq="5min", tz="UTC")
    c = 2000 + np.arange(n) * 0.1
    return pd.DataFrame({"time": idx, "open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
                         "tick_volume": 1, "spread": 20.0})


def test_bot_flat_before_warmup(tmp_path):
    bot = LiveBot(FB(), RuleBasedStrategy(StrategyConfig()), SPEC, CostConfig(),
                  ExecConfig(), SafetyConfig(), tmp_path / "j.jsonl")
    sig = bot.on_new_closed_bar(_buf(WARMUP_BARS - 10))
    assert sig.side == "FLAT"


def test_bot_signal_after_warmup(tmp_path):
    bot = LiveBot(FB(), RuleBasedStrategy(StrategyConfig()), SPEC, CostConfig(),
                  ExecConfig(), SafetyConfig(), tmp_path / "j.jsonl")
    sig = bot.on_new_closed_bar(_buf(WARMUP_BARS + 50))
    assert sig.side in ("BUY", "SELL", "FLAT")   # deterministic given features; journaled
    from mt5gold.live.journal import read_journal
    assert len(read_journal(tmp_path / "j.jsonl")) == 1
