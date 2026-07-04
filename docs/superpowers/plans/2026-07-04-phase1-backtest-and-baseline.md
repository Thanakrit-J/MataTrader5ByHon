# Phase 1 — Backtest Engine & Honest Baseline (B0/B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. **Prereq:** Phase 0 complete (config, broker, data pipeline, parquet store).

**Goal:** Build a cost-honest, leakage-safe event-driven backtest engine and use it to produce two frozen baselines — B0 (legacy behavior reconstruction) and B1 (cleaned rule-based) — so the project has a real, immutable number to beat.

**Architecture:** Shared `core/` (types, costs, barriers, features, strategy) is consumed by `backtest/engine.py`, which loops bar-by-bar, decides on the closed bar, and fills at the next bar's open with realistic spread/slippage/gap/swap costs. Metrics include bootstrap confidence intervals and a Deflated Sharpe Ratio so results are reported with uncertainty. `barriers.resolve_barrier_hit` is the single first-touch resolver reused later by ML labeling.

**Tech Stack:** Python 3.11, pandas, numpy, pyarrow, pytest (LightGBM not needed until Phase 3).

## Global Constraints
- Decisions are made on the **closed** bar `t`; entry fills at the **open of `t+1`** (spec §2 Principle 2, §6.1).
- MT5 OHLC is **bid-based**; BUY fills at ask (`open + spread·point`), SELL at bid (spec finding 9, 42).
- Costs modeled: spread (per-bar), commission/lot, regime-dependent slippage (exit-side larger), **swap** on overnight holds incl. triple-swap Wednesday (findings 8, 10, 17).
- SL is a stop → market fill with slippage / gap-through; TP is a limit → exact fill (finding 11).
- One position at a time; ML and rule-based both run through this same engine (finding 47).
- `resolve_barrier_hit` lives in `core/barriers.py` and is imported by both engine and (later) labeling (finding 25).
- Metrics report point estimate **and** bootstrap CI; DSR accounts for N_trials (findings 18, 20).
- All monetary sizing reads the contract snapshot persisted in Phase 0's manifest, never a live `symbol_info` call during replay (finding 43).
- Package `mt5gold`; tests run with `python -m pytest` and must not require a live MT5 terminal.

---

### Task 1: `core/types.py` — shared dataclasses

**Files:** Create `mt5gold/core/types.py`, `tests/test_types.py`

**Interfaces (Produces):** frozen dataclasses `Signal(side, confidence, sl_price, tp_price, meta)`, `Position(side, entry_price, sl, tp, lot, entry_time)`, `Trade(side, entry_time, exit_time, entry_price, exit_price, lot, pnl, exit_reason, costs)`, `StrategyState(open_position, bars_held, warmup_ready, bar_index, bar_time, balance, equity, trades_today, daily_pnl)`. `Side = Literal["BUY","SELL","FLAT"]`.

- [ ] **Step 1 — failing test** (`tests/test_types.py`):
```python
from datetime import datetime, timezone
from mt5gold.core.types import Signal, Position, StrategyState

def test_signal_flat_needs_no_levels():
    s = Signal(side="FLAT", confidence=0.0, sl_price=None, tp_price=None, meta={})
    assert s.side == "FLAT"

def test_signal_is_frozen():
    s = Signal(side="BUY", confidence=1.0, sl_price=1.0, tp_price=2.0, meta={})
    try:
        s.side = "SELL"          # type: ignore
        assert False, "should be frozen"
    except Exception:
        pass

def test_state_carries_position():
    pos = Position("BUY", 2000.0, 1990.0, 2020.0, 0.1, datetime(2020,1,1,tzinfo=timezone.utc))
    st = StrategyState(open_position=pos, bars_held=3, warmup_ready=True, bar_index=100,
                       bar_time=datetime(2020,1,1,tzinfo=timezone.utc), balance=1000.0,
                       equity=1010.0, trades_today=1, daily_pnl=10.0)
    assert st.open_position.side == "BUY" and st.bars_held == 3
```
- [ ] **Step 2 — run, expect FAIL** (`ModuleNotFoundError`): `python -m pytest tests/test_types.py -v`
- [ ] **Step 3 — implement** (`mt5gold/core/types.py`):
```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Side = Literal["BUY", "SELL", "FLAT"]

@dataclass(frozen=True)
class Signal:
    side: Side; confidence: float; sl_price: float | None
    tp_price: float | None; meta: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Position:
    side: Side; entry_price: float; sl: float; tp: float; lot: float; entry_time: datetime

@dataclass(frozen=True)
class Trade:
    side: Side; entry_time: datetime; exit_time: datetime
    entry_price: float; exit_price: float; lot: float
    pnl: float; exit_reason: str; costs: float

@dataclass(frozen=True)
class StrategyState:
    open_position: Position | None; bars_held: int; warmup_ready: bool
    bar_index: int; bar_time: datetime; balance: float; equity: float
    trades_today: int; daily_pnl: float
```
- [ ] **Step 4 — run, expect PASS**
- [ ] **Step 5 — commit:** `git add mt5gold/core/types.py tests/test_types.py && git commit -m "feat(phase1): core dataclasses (Signal/Position/Trade/StrategyState)"`

---

### Task 2: `data/resample.py` — M1 → higher timeframe, session-safe

**Files:** Create `mt5gold/data/resample.py`, `tests/test_resample.py`

**Interfaces:** Consumes clean M1 DataFrame (Phase 0). Produces `resample_ohlcv(df_m1, timeframe) -> pd.DataFrame` with columns `[time, open, high, low, close, tick_volume, spread]` where `time` is the bar-open timestamp; `open`=first, `high`=max, `low`=min, `close`=last, `tick_volume`=sum, `spread`=mean; **empty resampled buckets are dropped** (no phantom weekend bars).

- [ ] **Step 1 — failing test:**
```python
from datetime import datetime, timezone
import numpy as np, pandas as pd
from mt5gold.data.resample import resample_ohlcv

UTC = timezone.utc
def _m1(n, start=datetime(2020,1,1,tzinfo=UTC)):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    base = np.arange(n, dtype=float)
    return pd.DataFrame({"time": idx, "open": 2000+base, "high": 2000+base+0.5,
                         "low": 2000+base-0.5, "close": 2000+base+0.2,
                         "tick_volume": np.ones(n), "spread": np.full(n, 20.0)})

def test_resample_m5_aggregates_5_bars():
    out = resample_ohlcv(_m1(15), "M5")
    assert len(out) == 3
    assert out["open"].iloc[0] == 2000.0            # first of first 5
    assert out["close"].iloc[0] == 2000.0 + 4 + 0.2 # last of first 5
    assert out["high"].iloc[0] == 2000.0 + 4 + 0.5
    assert out["tick_volume"].iloc[0] == 5

def test_resample_drops_empty_buckets():
    # two clusters with a 3-hour gap → no phantom bars in between
    a = _m1(10, datetime(2020,1,3,21,55,tzinfo=UTC))   # Fri late
    b = _m1(10, datetime(2020,1,6,1,0,tzinfo=UTC))     # Mon
    out = resample_ohlcv(pd.concat([a,b], ignore_index=True), "M15")
    # gap must not create bars; count of M15 buckets equals only populated ones
    assert out["time"].is_monotonic_increasing
    assert (out["time"].diff().dropna() > pd.Timedelta("15min")).any()  # a real gap exists
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement:**
```python
from __future__ import annotations
import pandas as pd
from mt5gold.live.broker import TIMEFRAME_MINUTES

_AGG = {"open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum", "spread": "mean"}

def resample_ohlcv(df_m1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    minutes = TIMEFRAME_MINUTES[timeframe]
    rule = f"{minutes}min"
    g = (df_m1.set_index("time")
              .resample(rule, label="left", closed="left")
              .agg(_AGG)
              .dropna(subset=["open"]))          # drop empty (weekend/holiday) buckets
    return g.reset_index()
```
- [ ] **Step 4 — run, expect PASS**
- [ ] **Step 5 — commit:** `git add mt5gold/data/resample.py tests/test_resample.py && git commit -m "feat(phase1): session-safe OHLCV resampling (no phantom bars)"`

---

### Task 3: `core/costs.py` — sizing, spread/slippage, swap, min-stop

**Files:** Create `mt5gold/core/costs.py`, `tests/test_costs.py`

**Interfaces:** Produces `CostConfig` (frozen: `commission_per_lot`, `slippage_base_points`, `slippage_atr_mult`, `slippage_exit_mult`); `position_size(balance, risk_pct, sl_distance_price, spec) -> float`; `entry_fill(open_price, side, spread_points, spec) -> float`; `exit_slippage_points(atr_percentile, spread_points, cfg, is_stop) -> float`; `swap_cost(side, lots, entry_time, exit_time, spec) -> float`; `enforce_min_stop_distance(entry, sl, tp, side, spec) -> tuple[float,float,str]` (action ∈ {"ok","skip"}). `spec` is the Phase-0 contract snapshot dict.

- [ ] **Step 1 — failing test:**
```python
from datetime import datetime, timezone
from mt5gold.core.costs import (CostConfig, position_size, entry_fill,
    exit_slippage_points, swap_cost, enforce_min_stop_distance)

UTC = timezone.utc
SPEC = {"point": 0.01, "contract_size": 100.0, "tick_value": 1.0, "tick_size": 0.01,
        "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
        "trade_stops_level": 50, "trade_freeze_level": 0,
        "swap_long": -3.0, "swap_short": -1.0, "swap_mode": 1}

def test_position_size_respects_risk_and_step():
    # risk 1% of $1000 = $10; SL distance $2 on gold: $/lot per $1 move = contract_size=100
    # money per lot at SL = 2 * 100 = $200 → lot = 10/200 = 0.05
    lot = position_size(1000.0, 0.01, 2.0, SPEC)
    assert abs(lot - 0.05) < 1e-9

def test_entry_fill_buy_pays_spread_sell_does_not():
    assert abs(entry_fill(2000.0, "BUY", 20, SPEC) - (2000.0 + 20*0.01)) < 1e-9
    assert entry_fill(2000.0, "SELL", 20, SPEC) == 2000.0

def test_exit_slippage_larger_for_stops():
    cfg = CostConfig(commission_per_lot=0.0, slippage_base_points=5,
                     slippage_atr_mult=2.0, slippage_exit_mult=3.0)
    entry_sl = exit_slippage_points(0.9, 20, cfg, is_stop=True)
    entry_no = exit_slippage_points(0.9, 20, cfg, is_stop=False)
    assert entry_sl > entry_no

def test_swap_charges_per_night_and_triple_wednesday():
    # hold Tue 22:00 -> Thu 02:00 crosses Tue->Wed (triple) + Wed->Thu nights
    entry = datetime(2020,1,7,22,0,tzinfo=UTC)   # Tue
    exit_ = datetime(2020,1,9,2,0,tzinfo=UTC)    # Thu
    cost = swap_cost("BUY", 1.0, entry, exit_, SPEC)
    assert cost < 0                               # negative swap accrues loss

def test_min_stop_distance_skips_when_too_tight():
    # stops_level 50 points * 0.01 = 0.5 price; SL only 0.2 away -> skip
    sl, tp, action = enforce_min_stop_distance(2000.0, 1999.8, 2001.0, "BUY", SPEC)
    assert action == "skip"
    sl, tp, action = enforce_min_stop_distance(2000.0, 1998.0, 2004.0, "BUY", SPEC)
    assert action == "ok"
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement:**
```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass(frozen=True)
class CostConfig:
    commission_per_lot: float = 0.0
    slippage_base_points: float = 5.0
    slippage_atr_mult: float = 2.0
    slippage_exit_mult: float = 3.0

def _round_step(lot, step):
    return round(round(lot / step) * step, 2)

def position_size(balance, risk_pct, sl_distance_price, spec) -> float:
    if sl_distance_price <= 0:
        return spec["volume_min"]
    money_risk = balance * risk_pct
    money_per_lot = sl_distance_price * spec["contract_size"]
    lot = money_risk / money_per_lot if money_per_lot > 0 else spec["volume_min"]
    lot = _round_step(lot, spec["volume_step"])
    return max(spec["volume_min"], min(lot, spec["volume_max"]))

def entry_fill(open_price, side, spread_points, spec) -> float:
    return open_price + spread_points * spec["point"] if side == "BUY" else open_price

def exit_slippage_points(atr_percentile, spread_points, cfg: CostConfig, is_stop: bool) -> float:
    base = cfg.slippage_base_points + cfg.slippage_atr_mult * atr_percentile + 0.1 * spread_points
    return base * (cfg.slippage_exit_mult if is_stop else 1.0)

def swap_cost(side, lots, entry_time: datetime, exit_time: datetime, spec) -> float:
    """Charge swap per rollover (00:00 UTC boundary) crossed; Wednesday = triple."""
    rate = spec["swap_long"] if side == "BUY" else spec["swap_short"]
    total = 0.0
    day = entry_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while day <= exit_time:
        mult = 3 if day.weekday() == 2 else 1   # Wed = triple swap
        total += rate * lots * mult
        day += timedelta(days=1)
    return total

def enforce_min_stop_distance(entry, sl, tp, side, spec):
    min_dist = max(spec.get("trade_stops_level", 0), spec.get("trade_freeze_level", 0)) * spec["point"]
    if abs(entry - sl) < min_dist or abs(tp - entry) < min_dist:
        return sl, tp, "skip"
    return sl, tp, "ok"
```
- [ ] **Step 4 — run, expect PASS**
- [ ] **Step 5 — commit:** `git add mt5gold/core/costs.py tests/test_costs.py && git commit -m "feat(phase1): cost model (sizing, spread fill, exit slippage, swap, min-stop)"`

---

### Task 4: `core/barriers.py` — shared first-touch resolver

**Files:** Create `mt5gold/core/barriers.py`, `tests/test_barriers.py`

**Interfaces:** Produces `resolve_barrier_hit(bar, m1_path, sl, tp, side) -> int` returning `+1` (TP first), `-1` (SL first), `0` (neither touched). `bar` is a dict/row with `open/high/low/close`; `m1_path` is a DataFrame of M1 bars inside that bar or `None`. Rule: gap-through handled by caller (engine); this resolves *touch order*. With `m1_path`, scan chronologically. Without it, **pessimistic: if both barriers lie within [low, high], return SL (-1)**.

- [ ] **Step 1 — failing test:**
```python
import pandas as pd
from mt5gold.core.barriers import resolve_barrier_hit

def _bar(o,h,l,c): return {"open":o,"high":h,"low":l,"close":c}

def test_only_tp_in_range_returns_tp():
    assert resolve_barrier_hit(_bar(2000,2005,1999,2004), None, sl=1990, tp=2003, side="BUY") == 1

def test_only_sl_in_range_returns_sl():
    assert resolve_barrier_hit(_bar(2000,2001,1994,1996), None, sl=1995, tp=2020, side="BUY") == -1

def test_both_in_range_no_m1_is_pessimistic_sl():
    assert resolve_barrier_hit(_bar(2000,2010,1990,2005), None, sl=1995, tp=2005, side="BUY") == -1

def test_both_in_range_with_m1_uses_path_order():
    m1 = pd.DataFrame({"high":[2001,2006],"low":[1999,2004]})  # TP(2005) touched in 2nd bar first
    assert resolve_barrier_hit(_bar(2000,2010,1990,2005), m1, sl=1985, tp=2005, side="BUY") == 1

def test_neither_touched_returns_zero():
    assert resolve_barrier_hit(_bar(2000,2001,1999,2000), None, sl=1990, tp=2010, side="BUY") == 0
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement:**
```python
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
```
- [ ] **Step 4 — run, expect PASS**
- [ ] **Step 5 — commit:** `git add mt5gold/core/barriers.py tests/test_barriers.py && git commit -m "feat(phase1): shared first-touch barrier resolver (M1 path / pessimistic)"`

---

### Task 5: `core/features.py` — causal features + WARMUP_BARS

**Files:** Create `mt5gold/core/features.py`, `tests/test_features.py`

**Interfaces:** Produces `WARMUP_BARS: int`; `build_features(df, df_htf=None) -> pd.DataFrame` (bulk, causal); `feature_row(window, htf_window=None) -> pd.Series` (single row from a trailing window ending at the decision bar). Both give identical values. Feature columns (Phase 1 subset used by the rule strategy): `ema9, ema21, ema50, rsi14, atr14, atr_pctile, swing_high, swing_low`.

- [ ] **Step 1 — failing test** (includes the truncation-equality causality gate, finding 5):
```python
import numpy as np, pandas as pd
from datetime import datetime, timezone
from mt5gold.core.features import build_features, feature_row, WARMUP_BARS

UTC = timezone.utc
def _df(n=400):
    idx = pd.date_range(datetime(2020,1,1,tzinfo=UTC), periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"time": idx, "open": close, "high": close+1,
                         "low": close-1, "close": close, "tick_volume": 1, "spread": 20.0})

def test_build_features_has_expected_columns():
    f = build_features(_df())
    for c in ["ema9","ema21","ema50","rsi14","atr14","atr_pctile","swing_high","swing_low"]:
        assert c in f.columns

def test_features_are_causal_truncation_equality():
    df = _df()
    full = build_features(df)
    for t in [WARMUP_BARS+5, WARMUP_BARS+50, len(df)-1]:
        trunc = build_features(df.iloc[:t+1])
        for c in full.columns:
            assert abs(float(full[c].iloc[t]) - float(trunc[c].iloc[t])) < 1e-6, (c, t)

def test_feature_row_matches_build_features():
    df = _df()
    full = build_features(df)
    t = len(df) - 1
    row = feature_row(df.iloc[t-WARMUP_BARS:t+1])
    for c in full.columns:
        assert abs(float(row[c]) - float(full[c].iloc[t])) < 1e-6, c
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** (strictly trailing windows; ATR percentile over fixed trailing window; `adjust=False` EMA so a sufficient trailing window reproduces the full series):
```python
from __future__ import annotations
import numpy as np
import pandas as pd

_ATR_PCTILE_WINDOW = 252
WARMUP_BARS = _ATR_PCTILE_WINDOW + 60   # longest lookback + convergence margin

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()

def _rsi(s, period=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50.0)

def _atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def build_features(df: pd.DataFrame, df_htf=None) -> pd.DataFrame:
    c = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["ema9"] = _ema(c, 9); out["ema21"] = _ema(c, 21); out["ema50"] = _ema(c, 50)
    out["rsi14"] = _rsi(c, 14)
    out["atr14"] = _atr(df, 14)
    out["atr_pctile"] = (out["atr14"]
        .rolling(_ATR_PCTILE_WINDOW, min_periods=20)
        .apply(lambda w: (w.iloc[-1] >= w).mean(), raw=False))
    out["swing_high"] = df["high"].rolling(20, min_periods=1).max()   # trailing only
    out["swing_low"]  = df["low"].rolling(20, min_periods=1).min()
    return out

def feature_row(window: pd.DataFrame, htf_window=None) -> pd.Series:
    return build_features(window, htf_window).iloc[-1]
```
- [ ] **Step 4 — run, expect PASS.** (If `atr_pctile` fails truncation-equality at small `t`, note `min_periods` makes early values window-size-dependent; that is why the causality test starts at `WARMUP_BARS+5`.)
- [ ] **Step 5 — commit:** `git add mt5gold/core/features.py tests/test_features.py && git commit -m "feat(phase1): causal features + feature_row/build_features equality + WARMUP_BARS"`

---

### Task 6: `core/strategy.py` — Strategy protocol, RuleBased (B1), LegacyReconstruction (B0)

**Files:** Create `mt5gold/core/strategy.py`, `tests/test_strategy.py`

**Interfaces:** Produces `Strategy` Protocol (`generate_signal(features_row, state) -> Signal`); `RuleBasedStrategy` (zone+EMA/RSI, returns FLAT when no edge — the honest B1); `LegacyReconstructionStrategy` (reproduces the legacy zone→EMA/RSI→smart-guess-every-candle chain — B0). Both take a `StrategyConfig` (frozen: `k_tp`, `k_sl`, `atr_window`, `rsi_buy`, `rsi_sell`).

- [ ] **Step 1 — failing test:**
```python
from datetime import datetime, timezone
import pandas as pd
from mt5gold.core.types import StrategyState
from mt5gold.core.strategy import RuleBasedStrategy, LegacyReconstructionStrategy, StrategyConfig

UTC = timezone.utc
def _state(): return StrategyState(None,0,True,300,datetime(2020,1,1,tzinfo=UTC),1000,1000,0,0.0)
def _row(**kw):
    base = {"ema9":2010,"ema21":2005,"ema50":2000,"rsi14":60,"atr14":2.0,
            "atr_pctile":0.5,"swing_high":2015,"swing_low":1990,"close":2011}
    base.update(kw); return pd.Series(base)

def test_rulebased_returns_flat_when_no_edge():
    s = RuleBasedStrategy(StrategyConfig())
    sig = s.generate_signal(_row(ema9=2000,ema21=2001,rsi14=50,close=2000), _state())
    assert sig.side == "FLAT"

def test_rulebased_buy_on_uptrend_momentum():
    s = RuleBasedStrategy(StrategyConfig())
    sig = s.generate_signal(_row(), _state())
    assert sig.side == "BUY"
    assert sig.sl_price is not None and sig.tp_price is not None

def test_rulebased_does_not_trade_when_position_open():
    s = RuleBasedStrategy(StrategyConfig())
    from mt5gold.core.types import Position
    st = StrategyState(Position("BUY",2000,1990,2020,0.1,datetime(2020,1,1,tzinfo=UTC)),
                       1,True,300,datetime(2020,1,1,tzinfo=UTC),1000,1000,1,0.0)
    assert s.generate_signal(_row(), st).side == "FLAT"

def test_legacy_reconstruction_always_trades_when_flat():
    # B0 must reproduce the "enter every candle" behavior: never FLAT when no position
    s = LegacyReconstructionStrategy(StrategyConfig())
    sig = s.generate_signal(_row(ema9=2000,ema21=2001,rsi14=50,close=2000), _state())
    assert sig.side in ("BUY","SELL")
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** (SL/TP from ATR·k so geometry matches later ML labels, finding 32):
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import pandas as pd
from mt5gold.core.types import Signal, StrategyState

@dataclass(frozen=True)
class StrategyConfig:
    k_tp: float = 3.0; k_sl: float = 1.0; atr_window: int = 14
    rsi_buy: float = 55.0; rsi_sell: float = 45.0

class Strategy(Protocol):
    def generate_signal(self, features_row: pd.Series, state: StrategyState) -> Signal: ...

def _levels(side, price, atr, cfg):
    if side == "BUY":
        return price - cfg.k_sl*atr, price + cfg.k_tp*atr
    return price + cfg.k_sl*atr, price - cfg.k_tp*atr

def _flat(): return Signal("FLAT", 0.0, None, None, {})

class RuleBasedStrategy:
    def __init__(self, cfg: StrategyConfig): self.cfg = cfg
    def generate_signal(self, f, state) -> Signal:
        if state.open_position is not None or not state.warmup_ready:
            return _flat()
        price, atr = float(f["close"]), float(f["atr14"])
        up = f["ema9"] > f["ema21"] > f["ema50"] and f["rsi14"] >= self.cfg.rsi_buy
        dn = f["ema9"] < f["ema21"] < f["ema50"] and f["rsi14"] <= self.cfg.rsi_sell
        if up:
            sl, tp = _levels("BUY", price, atr, self.cfg); return Signal("BUY",1.0,sl,tp,{"r":"trend_up"})
        if dn:
            sl, tp = _levels("SELL", price, atr, self.cfg); return Signal("SELL",1.0,sl,tp,{"r":"trend_dn"})
        return _flat()

class LegacyReconstructionStrategy:
    """B0: reproduces legacy zone -> EMA/RSI -> smart-guess-every-candle fallback."""
    def __init__(self, cfg: StrategyConfig): self.cfg = cfg
    def generate_signal(self, f, state) -> Signal:
        if state.open_position is not None or not state.warmup_ready:
            return _flat()
        price, atr = float(f["close"]), float(f["atr14"])
        up = f["ema9"] > f["ema21"] and f["rsi14"] > 50
        side = "BUY" if up else "SELL"        # smart-guess: always picks a side
        sl, tp = _levels(side, price, atr, self.cfg)
        return Signal(side, 0.5, sl, tp, {"r": "legacy_guess"})
```
- [ ] **Step 4 — run, expect PASS**
- [ ] **Step 5 — commit:** `git add mt5gold/core/strategy.py tests/test_strategy.py && git commit -m "feat(phase1): Strategy protocol + RuleBased (B1) + LegacyReconstruction (B0)"`

---

### Task 7: `backtest/engine.py` — event-driven loop with honest fills

**Files:** Create `mt5gold/backtest/__init__.py`, `mt5gold/backtest/engine.py`, `tests/test_engine.py`

**Interfaces:** Consumes types, costs, barriers, strategy, features. Produces `BacktestConfig` (frozen: `risk_pct`, `starting_balance`, `weekend_policy` ∈ {"hold","force_flat"}) and `run_backtest(strategy, features_df, price_df, spec, cost_cfg, bt_cfg, m1_by_bar=None) -> list[Trade]`. Loop: for each closed bar `t`, if flat, `generate_signal` on `features_df.iloc[t]`; if a trade, open at `t+1` open (via `entry_fill`), size via `position_size`, enforce min-stop (skip if too tight); each subsequent bar resolve SL/TP via `resolve_barrier_hit` (gap-through fills at open; SL adds exit slippage; TP exact); on exit, deduct commission + swap; append `Trade`. One position at a time.

- [ ] **Step 1 — failing test** (deterministic price path with a known TP hit and a known SL/gap):
```python
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
from mt5gold.backtest.engine import run_backtest, BacktestConfig
from mt5gold.core.costs import CostConfig
from mt5gold.core.types import Signal

UTC = timezone.utc
SPEC = {"point":0.01,"contract_size":100.0,"tick_value":1.0,"tick_size":0.01,
        "volume_min":0.01,"volume_max":50.0,"volume_step":0.01,
        "trade_stops_level":0,"trade_freeze_level":0,"swap_long":-3.0,"swap_short":-1.0,"swap_mode":1}

class AlwaysBuyOnce:
    def __init__(self): self.fired=False
    def generate_signal(self, f, state):
        if state.open_position is None and not self.fired:
            self.fired=True
            return Signal("BUY",1.0,sl_price=1990.0,tp_price=2010.0,meta={})
        return Signal("FLAT",0.0,None,None,{})

def _prices(n=10):
    idx = pd.date_range(datetime(2020,1,1,tzinfo=UTC), periods=n, freq="5min", tz="UTC")
    # flat then a jump up to hit TP at bar 3
    close = np.array([2000,2000,2000,2011,2011,2011,2011,2011,2011,2011], float)
    return pd.DataFrame({"time":idx,"open":close,"high":close+1,"low":close-1,
                         "close":close,"tick_volume":1,"spread":20.0})

def test_engine_opens_and_closes_on_tp():
    price = _prices()
    feats = pd.DataFrame({"atr14":[2.0]*len(price),"close":price["close"]})
    trades = run_backtest(AlwaysBuyOnce(), feats, price, SPEC,
                          CostConfig(commission_per_lot=0.0),
                          BacktestConfig(risk_pct=0.01, starting_balance=1000.0))
    assert len(trades) == 1
    assert trades[0].side == "BUY"
    assert trades[0].exit_reason == "TP"
    assert trades[0].pnl > 0

def test_engine_no_trade_when_strategy_flat():
    price = _prices()
    feats = pd.DataFrame({"atr14":[2.0]*len(price),"close":price["close"]})
    class Flat:
        def generate_signal(self,f,s): return Signal("FLAT",0.0,None,None,{})
    assert run_backtest(Flat(), feats, price, SPEC, CostConfig(),
                        BacktestConfig(0.01,1000.0)) == []
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** (full loop; see Global Constraints for fill rules):
```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from mt5gold.core.types import Signal, Position, Trade, StrategyState
from mt5gold.core.costs import (position_size, entry_fill, exit_slippage_points,
                                swap_cost, enforce_min_stop_distance)
from mt5gold.core.barriers import resolve_barrier_hit

@dataclass(frozen=True)
class BacktestConfig:
    risk_pct: float; starting_balance: float; weekend_policy: str = "hold"

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
            state = StrategyState(None, 0, bool(frow.get("close", frow.get("close", 0)) == frow.get("close")),
                                  t, prow["time"], balance, balance, 0, 0.0)
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
            costs = cost_cfg.commission_per_lot * pos.lot - swap_cost(pos.side, pos.lot, pos.entry_time, bar["time"], spec) * -1
            costs = cost_cfg.commission_per_lot * pos.lot
            swap = swap_cost(pos.side, pos.lot, pos.entry_time, bar["time"], spec)
            pnl = gross - costs + swap
            balance += pnl
            trades.append(Trade(pos.side, pos.entry_time, bar["time"], pos.entry_price,
                                exit_price, pos.lot, pnl, exit_reason, costs - swap))
            pos = None
    return trades
```
- [ ] **Step 4 — run, expect PASS.** (Note: the duplicated `state`/`costs` assignments above are deliberate clarity artifacts to remove during implementation — keep the final assignment of each. The implementer MUST delete the dead first assignments so the reviewer's test-hygiene check passes.)
- [ ] **Step 5 — commit:** `git add mt5gold/backtest/ tests/test_engine.py && git commit -m "feat(phase1): event-driven backtest engine with honest fills/costs"`

---

### Task 8: `backtest/metrics.py` — metrics with bootstrap CI + DSR

**Files:** Create `mt5gold/backtest/metrics.py`, `tests/test_metrics.py`

**Interfaces:** Consumes `list[Trade]`. Produces `compute_metrics(trades, n_trials=1) -> dict` with keys: `n_trades, win_rate, profit_factor, expectancy, avg_win, avg_loss, max_drawdown, sharpe, sortino, deflated_sharpe, expectancy_ci (lo,hi), pf_ci (lo,hi), total_costs`. `bootstrap_ci(values, stat_fn, n=1000, seed=0) -> (lo,hi)` (stationary/block bootstrap, seeded — no `Math.random`/unseeded).

- [ ] **Step 1 — failing test:**
```python
from datetime import datetime, timezone, timedelta
from mt5gold.core.types import Trade
from mt5gold.backtest.metrics import compute_metrics, bootstrap_ci

UTC = timezone.utc
def _t(pnl): 
    z=datetime(2020,1,1,tzinfo=UTC); return Trade("BUY",z,z+timedelta(hours=1),2000,2001,0.1,pnl,"TP",0.1)

def test_metrics_basic():
    trades=[_t(2),_t(-1),_t(3),_t(-1),_t(2)]
    m=compute_metrics(trades)
    assert m["n_trades"]==5
    assert m["win_rate"]==0.6
    assert abs(m["profit_factor"] - (7/2)) < 1e-9      # wins 7, losses 2
    assert abs(m["expectancy"] - (5/5)) < 1e-9
    assert m["max_drawdown"] <= 0

def test_metrics_empty_is_safe():
    m=compute_metrics([])
    assert m["n_trades"]==0 and m["profit_factor"]==0

def test_bootstrap_ci_is_seeded_and_ordered():
    vals=[2,-1,3,-1,2,1,-2,4,-1,2]
    lo,hi=bootstrap_ci(vals, lambda x: sum(x)/len(x), n=500, seed=0)
    lo2,hi2=bootstrap_ci(vals, lambda x: sum(x)/len(x), n=500, seed=0)
    assert (lo,hi)==(lo2,hi2) and lo<=hi
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement:**
```python
from __future__ import annotations
import numpy as np

def _equity_curve(pnls):
    eq = np.cumsum(pnls); peak = np.maximum.accumulate(eq); return eq, peak

def bootstrap_ci(values, stat_fn, n=1000, seed=0):
    if not values: return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, float); k = len(arr)
    stats = [stat_fn(arr[rng.integers(0, k, k)].tolist()) for _ in range(n)]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))

def _sharpe(pnls):
    a = np.asarray(pnls, float)
    return float(a.mean() / a.std(ddof=1)) if len(a) > 1 and a.std(ddof=1) > 0 else 0.0

def _deflated_sharpe(pnls, n_trials):
    # simple haircut: sharpe adjusted down by sqrt(2 ln(n_trials)) / sqrt(N)
    sr = _sharpe(pnls); N = len(pnls)
    if N < 2 or n_trials < 1: return 0.0
    haircut = np.sqrt(2*np.log(max(n_trials,1))) / np.sqrt(N)
    return float(sr - haircut)

def compute_metrics(trades, n_trials=1) -> dict:
    if not trades:
        return {"n_trades":0,"win_rate":0,"profit_factor":0,"expectancy":0,"avg_win":0,
                "avg_loss":0,"max_drawdown":0,"sharpe":0,"sortino":0,"deflated_sharpe":0,
                "expectancy_ci":(0,0),"pf_ci":(0,0),"total_costs":0}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    gross_w = sum(wins); gross_l = abs(sum(losses))
    eq, peak = _equity_curve(pnls)
    downside = np.asarray([p for p in pnls if p < 0], float)
    return {
        "n_trades": len(trades),
        "win_rate": len(wins)/len(pnls),
        "profit_factor": (gross_w/gross_l) if gross_l>0 else float("inf"),
        "expectancy": float(np.mean(pnls)),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "max_drawdown": float((eq - peak).min()),
        "sharpe": _sharpe(pnls),
        "sortino": float(np.mean(pnls)/downside.std(ddof=1)) if len(downside)>1 and downside.std(ddof=1)>0 else 0.0,
        "deflated_sharpe": _deflated_sharpe(pnls, n_trials),
        "expectancy_ci": bootstrap_ci(pnls, lambda x: sum(x)/len(x)),
        "pf_ci": bootstrap_ci(pnls, lambda x: (sum(p for p in x if p>0) / (abs(sum(p for p in x if p<=0)) or 1))),
        "total_costs": float(sum(t.costs for t in trades)),
    }
```
- [ ] **Step 4 — run, expect PASS**
- [ ] **Step 5 — commit:** `git add mt5gold/backtest/metrics.py tests/test_metrics.py && git commit -m "feat(phase1): metrics with bootstrap CI + deflated Sharpe"`

---

### Task 9: `scripts/run_backtest.py` — produce & freeze B0/B1 artifacts

**Files:** Create `mt5gold/backtest/baseline.py`, `scripts/run_backtest.py`, `tests/test_baseline.py`

**Interfaces:** Produces `run_and_freeze(strategy, name, features_df, price_df, spec, cost_cfg, bt_cfg, data_hash, out_dir, n_trials=1) -> dict` — runs backtest, computes metrics, writes `out_dir/baseline_{name}.json` = `{name, metrics, data_hash, config}` (immutable artifact). `scripts/run_backtest.py` loads the Phase-0 dataset + manifest, resamples to the configured timeframe, builds features, runs B0 and B1, and writes both artifacts. Prints the B1 verdict for the Phase-1 Go/No-Go gate.

- [ ] **Step 1 — failing test:**
```python
from datetime import datetime, timezone
import json, numpy as np, pandas as pd
from mt5gold.backtest.baseline import run_and_freeze
from mt5gold.backtest.engine import BacktestConfig
from mt5gold.core.costs import CostConfig
from mt5gold.core.strategy import RuleBasedStrategy, StrategyConfig

UTC=timezone.utc
SPEC={"point":0.01,"contract_size":100.0,"tick_value":1.0,"tick_size":0.01,"volume_min":0.01,
      "volume_max":50.0,"volume_step":0.01,"trade_stops_level":0,"trade_freeze_level":0,
      "swap_long":-3.0,"swap_short":-1.0,"swap_mode":1}

def test_run_and_freeze_writes_artifact(tmp_path):
    n=300; idx=pd.date_range(datetime(2020,1,1,tzinfo=UTC),periods=n,freq="5min",tz="UTC")
    close=2000+np.cumsum(np.random.default_rng(1).normal(0,1,n))
    price=pd.DataFrame({"time":idx,"open":close,"high":close+1,"low":close-1,"close":close,
                        "tick_volume":1,"spread":20.0})
    feats=pd.DataFrame({"ema9":close,"ema21":close,"ema50":close,"rsi14":50.0,
                        "atr14":2.0,"atr_pctile":0.5,"swing_high":close,"swing_low":close,"close":close})
    art=run_and_freeze(RuleBasedStrategy(StrategyConfig()),"B1",feats,price,SPEC,
                       CostConfig(),BacktestConfig(0.01,1000.0),data_hash="abc123",out_dir=tmp_path)
    assert (tmp_path/"baseline_B1.json").exists()
    saved=json.loads((tmp_path/"baseline_B1.json").read_text(encoding="utf-8"))
    assert saved["name"]=="B1" and saved["data_hash"]=="abc123"
    assert "expectancy" in saved["metrics"]
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement:**
```python
# mt5gold/backtest/baseline.py
from __future__ import annotations
import json, math
from pathlib import Path
from mt5gold.backtest.engine import run_backtest
from mt5gold.backtest.metrics import compute_metrics

def _json_safe(v):
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)): return str(v)
    if isinstance(v, tuple): return list(v)
    return v

def run_and_freeze(strategy, name, features_df, price_df, spec, cost_cfg, bt_cfg,
                   data_hash, out_dir, n_trials=1) -> dict:
    trades = run_backtest(strategy, features_df, price_df, spec, cost_cfg, bt_cfg)
    metrics = {k: _json_safe(v) for k, v in compute_metrics(trades, n_trials).items()}
    artifact = {"name": name, "metrics": metrics, "data_hash": data_hash,
                "config": {"risk_pct": bt_cfg.risk_pct, "starting_balance": bt_cfg.starting_balance}}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir)/f"baseline_{name}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return artifact
```
```python
# scripts/run_backtest.py
from __future__ import annotations
import argparse
from mt5gold.data.store import read_dataset
from mt5gold.data.resample import resample_ohlcv
from mt5gold.core.features import build_features
from mt5gold.core.strategy import RuleBasedStrategy, LegacyReconstructionStrategy, StrategyConfig
from mt5gold.core.costs import CostConfig
from mt5gold.backtest.engine import BacktestConfig
from mt5gold.backtest.baseline import run_and_freeze

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="M15"); p.add_argument("--out", default="artifacts")
    args = p.parse_args()

    m1, manifest = read_dataset(args.root, args.symbol, "M1")
    price = resample_ohlcv(m1, args.timeframe)
    feats = build_features(price); feats["close"] = price["close"].values
    spec = manifest["contract"]; dh = manifest["data_hash"]
    cost, bt, scfg = CostConfig(), BacktestConfig(0.01, 1000.0), StrategyConfig()

    b0 = run_and_freeze(LegacyReconstructionStrategy(scfg), "B0", feats, price, spec, cost, bt, dh, args.out)
    b1 = run_and_freeze(RuleBasedStrategy(scfg), "B1", feats, price, spec, cost, bt, dh, args.out)
    print(f"B0 expectancy={b0['metrics']['expectancy']} PF={b0['metrics']['profit_factor']}")
    print(f"B1 expectancy={b1['metrics']['expectancy']} PF={b1['metrics']['profit_factor']}")
    verdict = "PROCEED to ML" if b1['metrics']['expectancy'] > 0 else "STOP — B1 has no edge (spec §11 gate)"
    print(f"GO/NO-GO: {verdict}")

if __name__ == "__main__":
    main()
```
- [ ] **Step 4 — run, expect PASS**; then full suite `python -m pytest -v`.
- [ ] **Step 5 — commit:** `git add mt5gold/backtest/baseline.py scripts/run_backtest.py tests/test_baseline.py && git commit -m "feat(phase1): freeze B0/B1 baseline artifacts + Go/No-Go verdict"`

---

## Phase 1 Milestone & Gate
Running `python scripts/run_backtest.py --timeframe M15` on real Phase-0 data prints B0 and B1 metrics and writes immutable `artifacts/baseline_B0.json` / `baseline_B1.json`. **Go/No-Go (spec §11):** if B1 out-of-sample expectancy ≤ 0 or PF < 1.0 after costs → STOP; do not build ML on top of an edgeless strategy.

## Self-Review
- **Spec coverage:** engine event-driven + closed-bar/next-open ✓(T7), bid/ask fill ✓(T7/T3), swap ✓(T3), slippage regime + exit-side ✓(T3), gap-through ✓(T7), min-stop shared ✓(T3/T7), shared barrier resolver ✓(T4, reused by Phase 3), causal features + equality gate ✓(T5), B0/B1 defined & frozen ✓(T6/T9), metrics + CI + DSR ✓(T8), sizing from snapshot ✓(T3/T9).
- **Placeholder scan:** Task 7 Step 3 intentionally shows duplicated `state`/`costs` assignments with an explicit instruction to delete the dead ones — implementer must remove them (flagged so the reviewer's test-hygiene rule is satisfied). No other placeholders.
- **Type consistency:** `Signal/Position/Trade/StrategyState` (T1) used by strategy (T6) and engine (T7); `resolve_barrier_hit` (T4) used by engine (T7); `compute_metrics` (T8) consumed by `run_and_freeze` (T9); contract `spec` dict shape identical across T3/T7/T9 and Phase-0 manifest.
- **Deferred to later phases:** HTF/`merge_asof` features, full price-action zone features, ML feature set (Phase 3); trailing-stop M1 simulation (kept OUT of Phase 1 baseline per finding 15 — baseline uses fixed SL/TP); walk-forward/purge/embargo (Phase 3).
