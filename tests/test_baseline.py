from datetime import datetime, timezone
import json, numpy as np, pandas as pd
from mt5gold.backtest.baseline import run_and_freeze
from mt5gold.backtest.engine import BacktestConfig
from mt5gold.core.costs import CostConfig
from mt5gold.core.strategy import RuleBasedStrategy, StrategyConfig

UTC = timezone.utc
SPEC = {"point": 0.01, "contract_size": 100.0, "tick_value": 1.0, "tick_size": 0.01, "volume_min": 0.01,
        "volume_max": 50.0, "volume_step": 0.01, "trade_stops_level": 0, "trade_freeze_level": 0,
        "swap_long": -3.0, "swap_short": -1.0, "swap_mode": 1}


def test_run_and_freeze_writes_artifact(tmp_path):
    n = 300
    idx = pd.date_range(datetime(2020, 1, 1, tzinfo=UTC), periods=n, freq="5min", tz="UTC")
    close = 2000 + np.cumsum(np.random.default_rng(1).normal(0, 1, n))
    price = pd.DataFrame({"time": idx, "open": close, "high": close + 1, "low": close - 1, "close": close,
                          "tick_volume": 1, "spread": 20.0})
    feats = pd.DataFrame({"ema9": close, "ema21": close, "ema50": close, "rsi14": 50.0,
                          "atr14": 2.0, "atr_pctile": 0.5, "swing_high": close, "swing_low": close, "close": close})
    art = run_and_freeze(RuleBasedStrategy(StrategyConfig()), "B1", feats, price, SPEC,
                         CostConfig(), BacktestConfig(0.01, 1000.0), data_hash="abc123", out_dir=tmp_path)
    assert (tmp_path / "baseline_B1.json").exists()
    saved = json.loads((tmp_path / "baseline_B1.json").read_text(encoding="utf-8"))
    assert saved["name"] == "B1" and saved["data_hash"] == "abc123"
    assert "expectancy" in saved["metrics"]
