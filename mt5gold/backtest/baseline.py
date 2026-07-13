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


def go_no_go_verdict(metrics) -> tuple[bool, str]:
    """Spec §11 Go/No-Go gate, CI-aware. An edge counts only if the bootstrap
    confidence interval clears break-even, not merely the point estimate:
    require the expectancy CI lower bound > 0 AND the profit-factor CI lower
    bound >= 1.0. A positive point estimate whose CI straddles zero is noise."""
    exp_ci = metrics.get("expectancy_ci", [0.0, 0.0])
    pf_ci = metrics.get("pf_ci", [0.0, 0.0])
    exp_lo, exp_hi = float(exp_ci[0]), float(exp_ci[1])
    pf_lo, pf_hi = float(pf_ci[0]), float(pf_ci[1])
    proceed = exp_lo > 0.0 and pf_lo >= 1.0
    if proceed:
        msg = (f"PROCEED to ML — B1 edge is statistically positive after costs "
               f"(expectancy CI low={exp_lo:.3f} > 0, PF CI low={pf_lo:.3f} >= 1.0)")
    else:
        msg = (f"STOP — B1 edge not distinguishable from break-even after costs "
               f"(expectancy CI=[{exp_lo:.3f}, {exp_hi:.3f}], "
               f"PF CI=[{pf_lo:.3f}, {pf_hi:.3f}]); spec §11 gate")
    return proceed, msg


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
