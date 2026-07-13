from datetime import datetime, timezone, timedelta
from mt5gold.core.types import Trade
from mt5gold.backtest.metrics import compute_metrics, bootstrap_ci

UTC = timezone.utc


def _t(pnl):
    z = datetime(2020, 1, 1, tzinfo=UTC)
    return Trade("BUY", z, z + timedelta(hours=1), 2000, 2001, 0.1, pnl, "TP", 0.1)


def test_metrics_basic():
    trades = [_t(2), _t(-1), _t(3), _t(-1), _t(2)]
    m = compute_metrics(trades)
    assert m["n_trades"] == 5
    assert m["win_rate"] == 0.6
    assert abs(m["profit_factor"] - (7 / 2)) < 1e-9      # wins 7, losses 2
    assert abs(m["expectancy"] - (5 / 5)) < 1e-9
    assert m["max_drawdown"] <= 0


def test_metrics_empty_is_safe():
    m = compute_metrics([])
    assert m["n_trades"] == 0 and m["profit_factor"] == 0


def test_bootstrap_ci_is_seeded_and_ordered():
    vals = [2, -1, 3, -1, 2, 1, -2, 4, -1, 2]
    lo, hi = bootstrap_ci(vals, lambda x: sum(x) / len(x), n=500, seed=0)
    lo2, hi2 = bootstrap_ci(vals, lambda x: sum(x) / len(x), n=500, seed=0)
    assert (lo, hi) == (lo2, hi2) and lo <= hi
