"""CLI entrypoint: python scripts/fetch_data.py [--root data] [--start 2020-01-01]"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from mt5gold.config import DataConfig
from mt5gold.data.pipeline import run_ingest
from mt5gold.live.broker import Mt5Broker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    cfg = DataConfig(symbol=args.symbol, base_timeframe="M1",
                     history_start=args.start, history_end=args.end)
    path = run_ingest(Mt5Broker(), cfg, args.root,
                      created_utc=datetime.now(timezone.utc))
    print(f"Wrote dataset: {path}")


if __name__ == "__main__":
    main()
