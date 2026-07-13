import pandas as pd
from mt5gold.core.barriers import resolve_barrier_hit


def _bar(o, h, l, c): return {"open": o, "high": h, "low": l, "close": c}


def test_only_tp_in_range_returns_tp():
    assert resolve_barrier_hit(_bar(2000, 2005, 1999, 2004), None, sl=1990, tp=2003, side="BUY") == 1


def test_only_sl_in_range_returns_sl():
    assert resolve_barrier_hit(_bar(2000, 2001, 1994, 1996), None, sl=1995, tp=2020, side="BUY") == -1


def test_both_in_range_no_m1_is_pessimistic_sl():
    assert resolve_barrier_hit(_bar(2000, 2010, 1990, 2005), None, sl=1995, tp=2005, side="BUY") == -1


def test_both_in_range_with_m1_uses_path_order():
    m1 = pd.DataFrame({"high": [2001, 2006], "low": [1999, 2004]})  # TP(2005) touched in 2nd bar first
    assert resolve_barrier_hit(_bar(2000, 2010, 1990, 2005), m1, sl=1985, tp=2005, side="BUY") == 1


def test_neither_touched_returns_zero():
    assert resolve_barrier_hit(_bar(2000, 2001, 1999, 2000), None, sl=1990, tp=2010, side="BUY") == 0
