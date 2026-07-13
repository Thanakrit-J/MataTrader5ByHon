from __future__ import annotations
from mt5gold.core.types import StrategyState
from mt5gold.core.features import feature_row


def replay_live_signals(bot, price_df):
    sigs = []
    for t in range(len(price_df)):
        buf = price_df.iloc[:t + 1]
        sigs.append(bot.on_new_closed_bar(buf))
    return sigs


def backtest_signals(strategy, features_df, price_df):
    sigs = []
    for t in range(len(price_df)):
        row = features_df.iloc[t]
        state = StrategyState(None, 0, True, t, price_df.iloc[t]["time"], 1000.0, 1000.0, 0, 0.0)
        sigs.append(strategy.generate_signal(row, state))
    return sigs
