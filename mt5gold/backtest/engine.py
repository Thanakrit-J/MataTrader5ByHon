from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from mt5gold.core.types import Signal, Position, Trade, StrategyState
from mt5gold.core.costs import (position_size, entry_fill, exit_slippage_points,
                                swap_cost, enforce_min_stop_distance)
from mt5gold.core.barriers import resolve_barrier_hit


@dataclass(frozen=True)
class BacktestConfig:
    risk_pct: float
    starting_balance: float
    weekend_policy: str = "hold"


def _pnl(side, entry, exit_, lot, spec):
    diff = (exit_ - entry) if side == "BUY" else (entry - exit_)
    return diff * lot * spec["contract_size"]


def run_backtest(strategy, features_df, price_df, spec, cost_cfg, bt_cfg, m1_by_bar=None):
    trades: list[Trade] = []
    balance = bt_cfg.starting_balance
    pos: Position | None = None
    entry_idx = 0
    n = len(price_df)
    for t in range(n - 1):
        prow = price_df.iloc[t]
        if pos is None:
            frow = features_df.iloc[t]
            state = StrategyState(None, 0, True, t, prow["time"], balance, balance, 0, 0.0)
            sig: Signal = strategy.generate_signal(frow, state)
            if sig.side in ("BUY", "SELL"):
                nxt = price_df.iloc[t + 1]
                fill = entry_fill(nxt["open"], sig.side, nxt["spread"], spec)
                sl, tp, action = enforce_min_stop_distance(fill, sig.sl_price, sig.tp_price, sig.side, spec)
                if action == "skip":
                    continue
                lot = position_size(balance, bt_cfg.risk_pct, abs(fill - sl), spec)
                pos = Position(sig.side, fill, sl, tp, lot, nxt["time"])
                entry_idx = t + 1
            continue
        # position open: resolve on this bar
        bar = price_df.iloc[t]
        m1 = m1_by_bar.get(t) if m1_by_bar else None
        # gap-through at open
        exit_price = exit_reason = None
        if (pos.side == "BUY" and bar["open"] <= pos.sl) or (pos.side == "SELL" and bar["open"] >= pos.sl):
            slip = exit_slippage_points(0.5, bar["spread"], cost_cfg, is_stop=True) * spec["point"]
            exit_price = bar["open"] - slip if pos.side == "BUY" else bar["open"] + slip
            exit_reason = "SL_GAP"
        else:
            hit = resolve_barrier_hit(bar, m1, pos.sl, pos.tp, pos.side)
            if hit == -1:
                slip = exit_slippage_points(0.5, bar["spread"], cost_cfg, is_stop=True) * spec["point"]
                exit_price = pos.sl - slip if pos.side == "BUY" else pos.sl + slip
                exit_reason = "SL"
            elif hit == 1:
                exit_price = pos.tp                      # limit fills exactly
                exit_reason = "TP"
        if exit_price is not None:
            gross = _pnl(pos.side, pos.entry_price, exit_price, pos.lot, spec)
            costs = cost_cfg.commission_per_lot * pos.lot
            swap = swap_cost(pos.side, pos.lot, pos.entry_time, bar["time"], spec)
            pnl = gross - costs + swap
            balance += pnl
            trades.append(Trade(pos.side, pos.entry_time, bar["time"], pos.entry_price,
                                exit_price, pos.lot, pnl, exit_reason, costs - swap))
            pos = None
    return trades
