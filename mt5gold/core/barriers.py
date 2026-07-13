from __future__ import annotations
import pandas as pd


def _touch(hi, lo, sl, tp, side):
    if side == "BUY":
        return (hi >= tp, lo <= sl)
    return (lo <= tp, hi >= sl)   # SELL: tp below, sl above


def resolve_barrier_hit(bar, m1_path, sl, tp, side) -> int:
    if m1_path is not None and len(m1_path) > 0:
        for _, r in m1_path.iterrows():
            tp_hit, sl_hit = _touch(r["high"], r["low"], sl, tp, side)
            if tp_hit and sl_hit:
                return -1              # ambiguous within one M1 bar → pessimistic
            if tp_hit:
                return 1
            if sl_hit:
                return -1
        return 0
    tp_hit, sl_hit = _touch(bar["high"], bar["low"], sl, tp, side)
    if tp_hit and sl_hit:
        return -1                      # both in range, no path → pessimistic SL-first
    if tp_hit:
        return 1
    if sl_hit:
        return -1
    return 0
