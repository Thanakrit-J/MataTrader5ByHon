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
from mt5gold.backtest.baseline import run_and_freeze


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="M15")
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()

    m1, manifest = read_dataset(args.root, args.symbol, "M1")
    price = resample_ohlcv(m1, args.timeframe)
    feats = build_features(price)
    feats["close"] = price["close"].values
    spec = manifest["contract"]
    dh = manifest["data_hash"]
    cost, bt, scfg = CostConfig(), BacktestConfig(0.01, 1000.0), StrategyConfig()

    b0 = run_and_freeze(LegacyReconstructionStrategy(scfg), "B0", feats, price, spec, cost, bt, dh, args.out)
    b1 = run_and_freeze(RuleBasedStrategy(scfg), "B1", feats, price, spec, cost, bt, dh, args.out)
    print(f"B0 expectancy={b0['metrics']['expectancy']} PF={b0['metrics']['profit_factor']}")
    print(f"B1 expectancy={b1['metrics']['expectancy']} PF={b1['metrics']['profit_factor']}")
    verdict = "PROCEED to ML" if b1['metrics']['expectancy'] > 0 else "STOP — B1 has no edge (spec §11 gate)"
    print(f"GO/NO-GO: {verdict}")


if __name__ == "__main__":
    main()
