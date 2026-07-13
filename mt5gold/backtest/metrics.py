from __future__ import annotations
import numpy as np


def _equity_curve(pnls):
    eq = np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    return eq, peak


def bootstrap_ci(values, stat_fn, n=1000, seed=0):
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, float)
    k = len(arr)
    stats = [stat_fn(arr[rng.integers(0, k, k)].tolist()) for _ in range(n)]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def _sharpe(pnls):
    a = np.asarray(pnls, float)
    return float(a.mean() / a.std(ddof=1)) if len(a) > 1 and a.std(ddof=1) > 0 else 0.0


def _deflated_sharpe(pnls, n_trials):
    # simple haircut: sharpe adjusted down by sqrt(2 ln(n_trials)) / sqrt(N)
    sr = _sharpe(pnls)
    N = len(pnls)
    if N < 2 or n_trials < 1:
        return 0.0
    haircut = np.sqrt(2 * np.log(max(n_trials, 1))) / np.sqrt(N)
    return float(sr - haircut)


def compute_metrics(trades, n_trials=1) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "profit_factor": 0, "expectancy": 0, "avg_win": 0,
                "avg_loss": 0, "max_drawdown": 0, "sharpe": 0, "sortino": 0, "deflated_sharpe": 0,
                "expectancy_ci": (0, 0), "pf_ci": (0, 0), "total_costs": 0}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    eq, peak = _equity_curve(pnls)
    downside = np.asarray([p for p in pnls if p < 0], float)
    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(pnls),
        "profit_factor": (gross_w / gross_l) if gross_l > 0 else float("inf"),
        "expectancy": float(np.mean(pnls)),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "max_drawdown": float((eq - peak).min()),
        "sharpe": _sharpe(pnls),
        "sortino": float(np.mean(pnls) / downside.std(ddof=1)) if len(downside) > 1 and downside.std(ddof=1) > 0 else 0.0,
        "deflated_sharpe": _deflated_sharpe(pnls, n_trials),
        "expectancy_ci": bootstrap_ci(pnls, lambda x: sum(x) / len(x)),
        "pf_ci": bootstrap_ci(pnls, lambda x: (sum(p for p in x if p > 0) / (abs(sum(p for p in x if p <= 0)) or 1))),
        "total_costs": float(sum(t.costs for t in trades)),
    }
