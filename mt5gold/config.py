"""Typed configuration. SHARED configs must be byte-identical across
backtest/train/live; LIVE-only configs never affect a backtest (spec finding 31)."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """SHARED/pinned. Governs which data is fetched and how it is keyed."""
    symbol: str = "XAUUSD"
    base_timeframe: str = "M1"
    history_start: str = "2020-01-01"      # ISO date, UTC
    history_end: str | None = None          # None = up to now
    broker_tz_offset_hours: int | None = None  # detected at fetch time, then pinned


def config_hash(obj) -> str:
    """Stable 12-char hex hash over a frozen dataclass's fields."""
    payload = json.dumps(dataclasses.asdict(obj), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def require_env(name: str) -> str:
    """Read a required environment variable or fail fast (never log the value)."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value
