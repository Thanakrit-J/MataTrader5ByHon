"""Persist cleaned data as Parquet + a hashed JSON manifest (spec §4.3)."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# Left = normalized snapshot key (what the engine/costs read); right = the actual
# field name on MT5's symbol_info. Names differ from the plan because real MT5
# exposes `trade_contract_size` (not `contract_size`) and the engine also needs
# `point` for points->price conversion.
_CONTRACT_FIELDS = {
    "point": "point", "contract_size": "trade_contract_size",
    "tick_size": "trade_tick_size", "tick_value": "trade_tick_value",
    "volume_min": "volume_min", "volume_max": "volume_max", "volume_step": "volume_step",
    "trade_stops_level": "trade_stops_level", "trade_freeze_level": "trade_freeze_level",
    "swap_long": "swap_long", "swap_short": "swap_short", "swap_mode": "swap_mode",
}


def contract_snapshot(broker, symbol: str) -> dict:
    info = broker.symbol_info(symbol)
    acc = broker.account_info()
    snap = {out: info.get(src) for out, src in _CONTRACT_FIELDS.items()}
    snap["filling_mode"] = info.get("filling_mode")
    snap["account_currency"] = acc.get("currency")
    return snap


def dataframe_hash(df: pd.DataFrame) -> str:
    content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(content).hexdigest()[:12]


def _dir(root, symbol, timeframe) -> Path:
    return Path(root) / "clean" / symbol / timeframe


def write_dataset(df: pd.DataFrame, root, symbol: str, timeframe: str,
                  contract: dict, tz_offset_hours: int,
                  created_utc: datetime) -> Path:
    out_dir = _dir(root, symbol, timeframe)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.parquet"
    df.to_parquet(data_path, index=False)
    manifest = {
        "symbol": symbol, "timeframe": timeframe, "rows": int(len(df)),
        "start": df["time"].iloc[0].isoformat() if len(df) else None,
        "end": df["time"].iloc[-1].isoformat() if len(df) else None,
        "tz_offset_hours": tz_offset_hours,
        "data_hash": dataframe_hash(df),
        "contract": contract,
        "created_utc": created_utc.isoformat(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return data_path


def read_dataset(root, symbol: str, timeframe: str) -> tuple[pd.DataFrame, dict]:
    out_dir = _dir(root, symbol, timeframe)
    df = pd.read_parquet(out_dir / "data.parquet")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return df, manifest
