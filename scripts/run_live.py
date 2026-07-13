"""Live demo entrypoint (manual smoke on the dev machine, per DEV_SETUP §6).

Drives the SAME core path as the backtest: rolling buffer -> feature_row ->
strategy.generate_signal -> safety gates -> order via execution.py. All MT5 I/O
goes through broker.py. DEMO-ONLY (assert_demo shuts down on a live account).
Unit tests only import this module; real behavior is validated manually.
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt5gold.live.broker import Mt5Broker, TIMEFRAME_MINUTES
from mt5gold.live.bot import LiveBot
from mt5gold.live.safety import assert_demo, SafetyConfig
from mt5gold.live.execution import ExecConfig, build_order_request, place_order
from mt5gold.live.journal import record_decision
from mt5gold.core.strategy import RuleBasedStrategy, StrategyConfig
from mt5gold.core.costs import CostConfig, position_size
from mt5gold.core.features import WARMUP_BARS
from mt5gold.data.fetch import fetch_rates
from mt5gold.data.store import contract_snapshot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="M5")
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--journal", default="journal/live.jsonl")
    p.add_argument("--poll-seconds", type=int, default=5)
    args = p.parse_args()

    broker = Mt5Broker()
    broker.connect()
    assert_demo(broker)                         # safe-lock: refuse live accounts
    spec = contract_snapshot(broker, args.symbol)
    spec["symbol"] = args.symbol
    bot = LiveBot(broker, RuleBasedStrategy(StrategyConfig()), spec,
                  CostConfig(), ExecConfig(), SafetyConfig(), args.journal)

    tf_minutes = TIMEFRAME_MINUTES[args.timeframe]
    buffer_bars = WARMUP_BARS + 50
    last_bar_time = None
    print(f"Live loop on {args.symbol} {args.timeframe} (DEMO). Ctrl+C to stop.")
    try:
        while True:
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=tf_minutes * (buffer_bars + 5))
            buf = fetch_rates(broker, args.symbol, args.timeframe, start, end)
            if len(buf) >= 2:
                closed = buf.iloc[:-1]           # last row may still be forming
                bar_time = closed.iloc[-1]["time"]
                if last_bar_time is None or bar_time > last_bar_time:
                    last_bar_time = bar_time
                    sig = bot.on_new_closed_bar(closed)
                    if sig.side in ("BUY", "SELL"):
                        balance = broker.account_info()["balance"]
                        lot = position_size(balance, args.risk_pct,
                                            abs(float(closed.iloc[-1]["close"]) - sig.sl_price), spec)
                        req = build_order_request(broker, args.symbol, sig, lot, spec, ExecConfig())
                        result = place_order(broker, req)
                        record_decision(args.journal, {"bar_time": bar_time.isoformat(),
                            "side": sig.side, "action": "order",
                            "retcode": result.get("retcode"), "ticket": result.get("ticket")})
                        print(f"{bar_time} {sig.side} lot={lot} -> {result}")
            time.sleep(args.poll_seconds)
    finally:
        broker.shutdown()


if __name__ == "__main__":
    main()
