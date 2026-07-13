"""Disciplined B1 variant sweep — search a PRE-DECLARED grid for a rule config
with a real edge, honestly.

Guardrails against fooling ourselves:
  * fixed grid declared up front (no expanding it until something passes),
  * in-sample / out-of-sample split: a variant must clear the CI gate on BOTH,
  * deflated Sharpe uses n_trials = grid size (penalizes multiple testing),
  * gate = expectancy CI lower bound > 0 AND profit-factor CI lower bound >= 1.0.

Reads one stored dataset (default M15) and can also test H1 by resampling it.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt5gold.data.store import read_dataset
from mt5gold.data.resample import resample_ohlcv
from mt5gold.core.features import build_features, WARMUP_BARS
from mt5gold.core.strategy import RuleBasedStrategy, StrategyConfig
from mt5gold.core.costs import CostConfig
from mt5gold.backtest.engine import run_backtest, BacktestConfig
from mt5gold.backtest.metrics import compute_metrics

# --- pre-declared grid -------------------------------------------------------
TIMEFRAMES = ["M15", "H1"]
RR = [(1.5, 1.0), (2.0, 1.0), (3.0, 1.0)]      # (k_tp, k_sl)
RSI = [(55.0, 45.0), (60.0, 40.0)]             # (rsi_buy, rsi_sell)


def _prep(df, timeframe, base_tf):
    price = df if base_tf == timeframe else resample_ohlcv(df, timeframe)
    price = price.reset_index(drop=True)
    feats = build_features(price)
    feats["close"] = price["close"].values
    return price, feats.reset_index(drop=True)


def _slice(price, feats, a, b):
    return price.iloc[a:b].reset_index(drop=True), feats.iloc[a:b].reset_index(drop=True)


def _gate(m):
    return m["expectancy_ci"][0] > 0.0 and m["pf_ci"][0] >= 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--base-timeframe", default="M15")
    p.add_argument("--train-frac", type=float, default=0.7)
    args = p.parse_args()

    stored, manifest = read_dataset(args.root, args.symbol, args.base_timeframe)
    spec = manifest["contract"]
    cost, bt = CostConfig(), BacktestConfig(0.01, 1000.0)

    variants = [(tf, ktp, ksl, rb, rs)
                for tf in TIMEFRAMES for (ktp, ksl) in RR for (rb, rs) in RSI]
    n_trials = len(variants)
    prepared = {tf: _prep(stored, tf, args.base_timeframe) for tf in TIMEFRAMES}

    print(f"Sweep: {n_trials} variants | base={args.base_timeframe} | train_frac={args.train_frac}")
    hdr = f"{'tf':>3} {'kTP':>4} {'kSL':>4} {'rsiB':>4} {'rsiS':>4} | {'n':>5} {'exp':>7} {'expCIlo':>8} {'PFlo':>6} {'DSR':>6} {'gate':>5}"
    print("IN-SAMPLE:\n" + hdr)
    passers = []
    for (tf, ktp, ksl, rb, rs) in variants:
        price, feats = prepared[tf]
        n = len(price)
        split = WARMUP_BARS + int((n - WARMUP_BARS) * args.train_frac)
        ip, ifeat = _slice(price, feats, WARMUP_BARS, split)
        cfg = StrategyConfig(k_tp=ktp, k_sl=ksl, rsi_buy=rb, rsi_sell=rs)
        m = compute_metrics(run_backtest(RuleBasedStrategy(cfg), ifeat, ip, spec, cost, bt), n_trials=n_trials)
        g = _gate(m)
        print(f"{tf:>3} {ktp:>4} {ksl:>4} {rb:>4} {rs:>4} | {m['n_trades']:>5} {m['expectancy']:>7.3f} "
              f"{m['expectancy_ci'][0]:>8.3f} {m['pf_ci'][0]:>6.3f} {m['deflated_sharpe']:>6.3f} {str(g):>5}")
        if g:
            passers.append((tf, ktp, ksl, rb, rs, split))

    print(f"\nIn-sample gate passers: {len(passers)}")
    if not passers:
        print("VERDICT: no rule variant clears the CI gate in-sample -> STRONG STOP. "
              "Pure EMA/RSI trend rules on gold have no edge across this grid; the honest "
              "next step is a structurally different thesis (mean-reversion/regime) or "
              "accepting that technical-rule trading of gold has no edge here.")
        return

    print("\nOUT-OF-SAMPLE confirmation of in-sample passers:\n" + hdr)
    confirmed = []
    for (tf, ktp, ksl, rb, rs, split) in passers:
        price, feats = prepared[tf]
        op, ofeat = _slice(price, feats, split, len(price))
        cfg = StrategyConfig(k_tp=ktp, k_sl=ksl, rsi_buy=rb, rsi_sell=rs)
        m = compute_metrics(run_backtest(RuleBasedStrategy(cfg), ofeat, op, spec, cost, bt), n_trials=n_trials)
        g = _gate(m)
        print(f"{tf:>3} {ktp:>4} {ksl:>4} {rb:>4} {rs:>4} | {m['n_trades']:>5} {m['expectancy']:>7.3f} "
              f"{m['expectancy_ci'][0]:>8.3f} {m['pf_ci'][0]:>6.3f} {m['deflated_sharpe']:>6.3f} {str(g):>5}")
        if g:
            confirmed.append((tf, ktp, ksl, rb, rs))

    print(f"\nVERDICT: {len(confirmed)} variant(s) pass BOTH in-sample AND out-of-sample.")
    if confirmed:
        for c in confirmed:
            print(f"  CANDIDATE tf={c[0]} k_tp={c[1]} k_sl={c[2]} rsi={c[3]}/{c[4]} -> Phase 3 ML justified on this config")
    else:
        print("  None survive out-of-sample -> in-sample passers were multiple-testing noise. STOP stands.")


if __name__ == "__main__":
    main()
