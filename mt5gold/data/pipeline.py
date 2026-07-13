"""End-to-end ingest: detect offset -> fetch -> clean -> store."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from mt5gold.config import DataConfig
from mt5gold.data.fetch import fetch_rates
from mt5gold.data.clean import clean_rates
from mt5gold.data.store import write_dataset, contract_snapshot


def run_ingest(broker, cfg: DataConfig, root, created_utc: datetime,
               detect_offset: bool = True) -> Path:
    broker.connect()
    try:
        offset = cfg.broker_tz_offset_hours or 0
        if detect_offset:
            offset = int(broker.account_info().get("server_tz_offset_hours", offset))

        start = pd.Timestamp(cfg.history_start, tz="UTC").to_pydatetime()
        end = (pd.Timestamp(cfg.history_end, tz="UTC").to_pydatetime()
               if cfg.history_end else created_utc)

        raw = fetch_rates(broker, cfg.symbol, cfg.base_timeframe, start, end)
        cleaned = clean_rates(raw, broker_tz_offset_hours=offset)
        contract = contract_snapshot(broker, cfg.symbol)
        return write_dataset(cleaned, root, cfg.symbol, cfg.base_timeframe,
                             contract, tz_offset_hours=offset, created_utc=created_utc)
    finally:
        broker.shutdown()
