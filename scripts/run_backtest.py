from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repo root: put the repo root on
# sys.path so `import mt5gold` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt5gold.data.store import read_dataset
from mt5gold.data.resample import resample_ohlcv
from mt5gold.core.features import build_features
from mt5gold.core.strategy import RuleBasedStrategy, LegacyReconstructionStrategy, StrategyConfig
from mt5gold.core.costs import CostConfig
from mt5gold.backtest.engine import BacktestConfig
from mt5gold.backtest.baseline import run_and_freeze, go_no_go_verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="M15", help="timeframe to trade/backtest on")
    p.add_argument("--base-timeframe", default=None,
                   help="stored dataset timeframe (default = --timeframe). Set to a "
                        "finer TF (e.g. M1) to resample up to --timeframe.")
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()

    base_tf = args.base_timeframe or args.timeframe
    stored, manifest = read_dataset(args.root, args.symbol, base_tf)
    price = stored if base_tf == args.timeframe else resample_ohlcv(stored, args.timeframe)
    feats = build_features(price)
    feats["close"] = price["close"].values
    spec = manifest["contract"]
    dh = manifest["data_hash"]
    cost, bt, scfg = CostConfig(), BacktestConfig(0.01, 1000.0), StrategyConfig()

    b0 = run_and_freeze(LegacyReconstructionStrategy(scfg), "B0", feats, price, spec, cost, bt, dh, args.out)
    b1 = run_and_freeze(RuleBasedStrategy(scfg), "B1", feats, price, spec, cost, bt, dh, args.out)
    for art in (b0, b1):
        m = art["metrics"]
        print(f"{art['name']}: n_trades={m['n_trades']} expectancy={m['expectancy']:.3f} "
              f"CI={m['expectancy_ci']} PF={m['profit_factor']:.3f} PF_CI={m['pf_ci']} "
              f"maxDD={m['max_drawdown']:.1f}")
    _, msg = go_no_go_verdict(b1["metrics"])
    print(f"GO/NO-GO: {msg}")


if __name__ == "__main__":
    main()
