from __future__ import annotations
import json, math
from pathlib import Path
from mt5gold.backtest.engine import run_backtest
from mt5gold.backtest.metrics import compute_metrics


def _json_safe(v):
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return str(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def run_and_freeze(strategy, name, features_df, price_df, spec, cost_cfg, bt_cfg,
                   data_hash, out_dir, n_trials=1) -> dict:
    trades = run_backtest(strategy, features_df, price_df, spec, cost_cfg, bt_cfg)
    metrics = {k: _json_safe(v) for k, v in compute_metrics(trades, n_trials).items()}
    artifact = {"name": name, "metrics": metrics, "data_hash": data_hash,
                "config": {"risk_pct": bt_cfg.risk_pct, "starting_balance": bt_cfg.starting_balance}}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"baseline_{name}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return artifact
