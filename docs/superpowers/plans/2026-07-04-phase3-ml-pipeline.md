# Phase 3 — ML Pipeline (triple-barrier + walk-forward) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development / executing-plans. **Prereq:** Phases 0–2 complete. This phase is where ML beginners get burned — the leakage/validation tasks below are the load-bearing ones, not the model itself.

**Goal:** Build a leakage-safe ML pipeline — extended causal features (incl. HTF), triple-barrier labels sharing the engine's resolver, nested walk-forward with purge/embargo ≥ horizon, calibrated LightGBM, and evaluation against the frozen B1 baseline plus a matched permutation null — and only "confirm" an edge on a locked holdout opened exactly once.

**Architecture:** `ml/labeling.py` reuses `core/barriers.resolve_barrier_hit` so a `+1` label means the engine would also book a TP. `ml/validate.py` owns the nested walk-forward, purge/embargo, and locked holdout. `ml/train.py` fits LightGBM + isotonic calibration inside each fold. `ml/registry.py` binds model + `feature_version` + label params + hashes so live fails closed on mismatch. `MLStrategy` (in `core/strategy.py`) emits SL/TP from the SAME `LabelConfig` used to label.

**Tech Stack:** Python 3.11, pandas, numpy, lightgbm, scikit-learn, pytest.

## Global Constraints
- Labels use `core/barriers.resolve_barrier_hit`; a label's SL/TP geometry = the strategy's tradable geometry (`LabelConfig` shared by `labeling.py` and `MLStrategy`) (findings 25, 32).
- Labeling is causal: entry at open of `t+1`, barriers scaled by ATR at close of `t`, first-touch scanned forward; never use bar `t`'s own high/low to decide touch (finding 6).
- Walk-forward only; **no shuffling**. Purge overlapping labels; **embargo ≥ N** (vertical-barrier horizon) (findings 1, 22).
- Threshold + hyperparameters selected on an **inner-validation split** carved from the train window; the outer test fold is scored once (finding 2).
- A **locked holdout** (most recent 6–12 months) is never touched during research; opened once at the end (findings 2, 3, 18).
- Class weights / sample weights fit on **train slice only**, per fold (finding 41).
- Probabilities are **calibrated** (isotonic) before thresholding; report reliability + Brier (finding 44).
- Success = beat frozen **B1** by a **robustness bar** (≥70% of folds positive net-of-cost) AND beat a **matched permutation null**; report metrics with bootstrap CI + DSR (findings 3, 18, 21, 26).
- Every evaluation run is appended to `research_log.jsonl`; holdout eval counter must equal 1 (finding 7).

---

### Task 1: extend `core/features.py` — HTF context (leak-safe) + price-action

**Files:** Modify `mt5gold/core/features.py`; Create `tests/test_features_htf.py`

**Interfaces:** `build_features(df, df_htf=None)` gains columns when `df_htf` is provided: `htf_ema_fast, htf_ema_slow, htf_trend` (aligned via `merge_asof` on **close timestamps**, backward). Also add price-action `body_frac, upper_wick_frac, lower_wick_frac, dist_to_swing_high, dist_to_swing_low`. `feature_row` gains an optional `htf_window`. Bump `FEATURE_VERSION` string constant.

- [ ] **Step 1 — failing test** (the HTF no-lookahead invariant, finding 4):
```python
from datetime import datetime, timezone
import numpy as np, pandas as pd
from mt5gold.core.features import build_features, FEATURE_VERSION

UTC=timezone.utc
def _m15(n=200):
    idx=pd.date_range(datetime(2020,1,1,tzinfo=UTC),periods=n,freq="15min",tz="UTC")
    c=2000+np.cumsum(np.random.default_rng(4).normal(0,1,n))
    return pd.DataFrame({"time":idx,"open":c,"high":c+1,"low":c-1,"close":c,"tick_volume":1,"spread":20.0})

def _h1_with_step():
    # H1 bars keyed by OPEN time; close = open+1h. Put a big level change at the 10:00 bar.
    idx=pd.date_range(datetime(2020,1,1,tzinfo=UTC),periods=48,freq="1h",tz="UTC")
    close=np.where(idx.hour>=10, 3000.0, 2000.0)
    return pd.DataFrame({"time":idx,"open":close,"high":close+1,"low":close-1,"close":close,
                         "tick_volume":1,"spread":20.0})

def test_htf_uses_only_last_closed_bar():
    m15=_m15(); h1=_h1_with_step()
    f=build_features(m15, df_htf=h1)
    # M15 bars during 10:00-10:59 must still see the 09:00-10:00 H1 close (2000), NOT 10:00-11:00 (3000)
    mask=(m15["time"].dt.hour==10)
    vals=f.loc[mask.values,"htf_ema_slow"]
    assert (vals < 2500).all(), "HTF leaked an unclosed bar into M15"

def test_feature_version_present():
    assert isinstance(FEATURE_VERSION, str) and len(FEATURE_VERSION) > 0
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** (key part — HTF alignment on close timestamps, then `merge_asof` backward):
```python
# add near top of core/features.py
FEATURE_VERSION = "phase3-v1"

# inside build_features, before returning `out`, when df_htf is not None:
def _attach_htf(out, df, df_htf):
    htf = df_htf.copy()
    htf["close_time"] = htf["time"] + pd.to_timedelta(
        htf["time"].diff().median())          # bar_open + one HTF bar = close time
    htf["htf_ema_fast"] = _ema(htf["close"], 10)
    htf["htf_ema_slow"] = _ema(htf["close"], 20)
    htf["htf_trend"] = (htf["htf_ema_fast"] > htf["htf_ema_slow"]).astype(int)
    left = df[["time"]].copy()
    merged = pd.merge_asof(left.sort_values("time"),
                           htf[["close_time","htf_ema_fast","htf_ema_slow","htf_trend"]]
                               .sort_values("close_time"),
                           left_on="time", right_on="close_time", direction="backward")
    for c in ["htf_ema_fast","htf_ema_slow","htf_trend"]:
        out[c] = merged[c].values
    return out
```
Also add price-action columns (all from the closed bar `t`):
```python
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    out["body_frac"] = (df["close"] - df["open"]).abs() / rng
    out["upper_wick_frac"] = (df["high"] - df[["open","close"]].max(axis=1)) / rng
    out["lower_wick_frac"] = (df[["open","close"]].min(axis=1) - df["low"]) / rng
    out["dist_to_swing_high"] = (out["swing_high"] - df["close"]) / df["close"]
    out["dist_to_swing_low"]  = (df["close"] - out["swing_low"]) / df["close"]
    out = out.fillna(0.0)
```
- [ ] **Step 4 — run, expect PASS** (re-run Phase-1 `test_features.py` causality test too — new features must stay causal).
- [ ] **Step 5 — commit:** `git commit -am "feat(phase3): HTF (leak-safe merge_asof) + price-action features + FEATURE_VERSION"`

---

### Task 2: `ml/labeling.py` — triple-barrier via shared resolver

**Files:** Create `mt5gold/ml/__init__.py`, `mt5gold/ml/labeling.py`, `tests/test_labeling.py`

**Interfaces:** `LabelConfig` (frozen: `k_tp=3.0, k_sl=1.0, atr_window=14, horizon=24`). `triple_barrier_labels(price_df, atr, cfg, m1_by_bar=None) -> pd.Series` of `{+1,-1,0}` indexed like `price_df`, computed causally (entry at open of `t+1`, barriers from `close[t] ± k·atr[t]`, first touch within `horizon` bars via `resolve_barrier_hit`). The last `horizon` rows are `NaN` (unlabelable). Timeout(0) is a real class (not dropped).

- [ ] **Step 1 — failing test:**
```python
from datetime import datetime, timezone
import numpy as np, pandas as pd
from mt5gold.ml.labeling import LabelConfig, triple_barrier_labels
UTC=timezone.utc
def _price(closes):
    n=len(closes); idx=pd.date_range(datetime(2020,1,1,tzinfo=UTC),periods=n,freq="15min",tz="UTC")
    c=np.array(closes,float)
    return pd.DataFrame({"time":idx,"open":c,"high":c+0.2,"low":c-0.2,"close":c,
                         "tick_volume":1,"spread":20.0})

def test_label_tp_when_price_rises():
    # entry ~2000, atr=1, k_tp=3 -> TP 2003; price climbs to 2004 within horizon
    closes=[2000,2000,2001,2002,2003,2004,2004,2004]
    price=_price(closes); atr=pd.Series([1.0]*len(closes))
    y=triple_barrier_labels(price, atr, LabelConfig(k_tp=3,k_sl=5,horizon=6))
    assert y.iloc[1]==1

def test_label_sl_when_price_falls():
    closes=[2000,2000,1999,1998,1997,1996,1996,1996]
    price=_price(closes); atr=pd.Series([1.0]*len(closes))
    y=triple_barrier_labels(price, atr, LabelConfig(k_tp=5,k_sl=3,horizon=6))
    assert y.iloc[1]==-1

def test_last_horizon_rows_are_nan():
    closes=[2000]*10; price=_price(closes); atr=pd.Series([1.0]*10)
    y=triple_barrier_labels(price, atr, LabelConfig(horizon=4))
    assert y.iloc[-4:].isna().all()
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** (reuse resolver; entry at t+1 open):
```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from mt5gold.core.barriers import resolve_barrier_hit

@dataclass(frozen=True)
class LabelConfig:
    k_tp: float = 3.0; k_sl: float = 1.0; atr_window: int = 14; horizon: int = 24

def triple_barrier_labels(price_df, atr, cfg: LabelConfig, m1_by_bar=None) -> pd.Series:
    n = len(price_df); out = np.full(n, np.nan)
    close = price_df["close"].to_numpy(float)
    for t in range(n - 1):
        if t + cfg.horizon >= n:            # not enough future bars to resolve
            break
        entry = price_df["open"].iloc[t + 1]
        a = float(atr.iloc[t])
        tp = entry + cfg.k_tp * a; sl = entry - cfg.k_sl * a   # BUY-oriented labeling
        label = 0
        for j in range(t + 1, min(t + 1 + cfg.horizon, n)):
            bar = price_df.iloc[j]
            m1 = m1_by_bar.get(j) if m1_by_bar else None
            hit = resolve_barrier_hit(bar, m1, sl, tp, "BUY")
            if hit != 0:
                label = hit; break
        out[t] = label
    return pd.Series(out, index=price_df.index)
```
- [ ] **Step 4 — run, expect PASS** · **Step 5 — commit:** `git commit -am "feat(phase3): triple-barrier labeling via shared resolver (causal)"`

---

### Task 3: `ml/validate.py` — walk-forward with purge/embargo + locked holdout

**Files:** Create `mt5gold/ml/validate.py`, `tests/test_validate.py`

**Interfaces:** `walk_forward_folds(n, n_folds, embargo, horizon, holdout_frac=0.15) -> tuple[list[Fold], Fold]` where `Fold=(train_idx, test_idx)`. Rules: reserve the LAST `holdout_frac` as the locked holdout (returned separately, never inside research folds). Within the research span, expanding/rolling folds; **purge** train indices whose label window `[i, i+horizon]` overlaps the test window; **embargo** ≥ `max(embargo, horizon)` bars after each test window before training resumes. `inner_split(train_idx, embargo, horizon, val_frac=0.25)` carves an inner-validation split from the tail of a train fold with the same purge/embargo.

- [ ] **Step 1 — failing test** (the load-bearing leakage math):
```python
from mt5gold.ml.validate import walk_forward_folds, inner_split

def test_holdout_is_reserved_and_untouched():
    folds, holdout = walk_forward_folds(1000, n_folds=4, embargo=24, horizon=24, holdout_frac=0.15)
    hold_start = min(holdout[1])
    for tr, te in folds:
        assert max(tr) < hold_start and max(te) < hold_start   # nothing touches holdout

def test_embargo_at_least_horizon_between_train_and_test():
    folds, _ = walk_forward_folds(1000, n_folds=4, embargo=10, horizon=24, holdout_frac=0.1)
    for tr, te in folds:
        gap = min(te) - max(tr)
        assert gap >= 24, gap          # embargo>=horizon enforced

def test_purge_removes_overlapping_labels():
    folds, _ = walk_forward_folds(1000, n_folds=4, embargo=24, horizon=24, holdout_frac=0.1)
    for tr, te in folds:
        # no train index i can have its label window [i, i+horizon] reach into test
        assert all(i + 24 < min(te) for i in tr if i < min(te))

def test_inner_split_is_disjoint_and_embargoed():
    tr = list(range(0, 500))
    inner_tr, inner_val = inner_split(tr, embargo=24, horizon=24, val_frac=0.25)
    assert max(inner_tr) + 24 <= min(inner_val)
    assert set(inner_tr).isdisjoint(inner_val)
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement:**
```python
from __future__ import annotations

def _emb(embargo, horizon): return max(embargo, horizon)

def walk_forward_folds(n, n_folds, embargo, horizon, holdout_frac=0.15):
    e = _emb(embargo, horizon)
    hold_start = int(n * (1 - holdout_frac))
    holdout = (list(range(0, 0)), list(range(hold_start, n)))
    research_n = hold_start
    fold_size = research_n // (n_folds + 1)     # first block seeds the first train
    folds = []
    for k in range(1, n_folds + 1):
        test_start = fold_size * k
        test_end = min(fold_size * (k + 1), research_n)
        if test_start >= test_end: break
        test_idx = list(range(test_start, test_end))
        # train = everything up to test_start, purged so label window can't reach test_start
        train_idx = [i for i in range(0, test_start) if i + horizon < test_start]
        # embargo after previous test handled implicitly since next train excludes >=test regions
        folds.append((train_idx, test_idx))
    # enforce embargo gap between max(train) and min(test)
    folds = [([i for i in tr if i + e <= min(te)], te) for tr, te in folds if tr and te]
    return folds, holdout

def inner_split(train_idx, embargo, horizon, val_frac=0.25):
    e = _emb(embargo, horizon)
    train_idx = sorted(train_idx)
    cut = int(len(train_idx) * (1 - val_frac))
    val = train_idx[cut:]
    inner_tr = [i for i in train_idx[:cut] if i + e <= min(val)]
    return inner_tr, val
```
- [ ] **Step 4 — run, expect PASS** · **Step 5 — commit:** `git commit -am "feat(phase3): walk-forward folds with purge/embargo>=horizon + locked holdout + inner split"`

---

### Task 4: `ml/permutation.py` — matched permutation null baseline

**Files:** Create `mt5gold/ml/permutation.py`, `tests/test_permutation.py`

**Interfaces:** `permutation_pvalue(strategy_expectancy, per_trade_pnls, n=1000, seed=0) -> float` — sign-shuffle each trade's PnL (side randomization at the SAME entries/count) `n` times, recompute expectancy, return the fraction of permutations whose expectancy ≥ the strategy's (a matched, frequency-controlled null per finding 21). Edge is significant if p < 0.05.

- [ ] **Step 1 — failing test:**
```python
from mt5gold.ml.permutation import permutation_pvalue
def test_strong_edge_has_low_pvalue():
    pnls=[2,2,2,2,2,-1,2,2,-1,2]              # clearly positive
    p=permutation_pvalue(sum(pnls)/len(pnls), pnls, n=2000, seed=0)
    assert p < 0.05

def test_no_edge_has_high_pvalue():
    pnls=[1,-1,1,-1,1,-1,1,-1]                # symmetric → sign flips don't change |mean|~0
    p=permutation_pvalue(sum(pnls)/len(pnls), pnls, n=2000, seed=0)
    assert p > 0.05

def test_pvalue_is_seeded():
    pnls=[2,-1,3,-1,2]
    assert permutation_pvalue(1.0,pnls,n=500,seed=0)==permutation_pvalue(1.0,pnls,n=500,seed=0)
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement:**
```python
from __future__ import annotations
import numpy as np

def permutation_pvalue(strategy_expectancy, per_trade_pnls, n=1000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_trade_pnls, float)
    if len(arr) == 0: return 1.0
    ge = 0
    for _ in range(n):
        signs = rng.choice([-1.0, 1.0], size=len(arr))
        perm_expectancy = float((arr * signs).mean())
        if perm_expectancy >= strategy_expectancy:
            ge += 1
    return ge / n
```
- [ ] **Step 4 — run, expect PASS** · **Step 5 — commit:** `git commit -am "feat(phase3): matched permutation null baseline"`

---

### Task 5: `ml/train.py` + `ml/model.py` — LightGBM + isotonic calibration

**Files:** Create `mt5gold/ml/train.py`, `mt5gold/ml/model.py`, `tests/test_train.py`

**Interfaces:** `train_fold(X_tr, y_tr, X_val, y_val, params=None) -> CalibratedModel` — fits LightGBM (binary: label==+1 vs not; timeouts/SL as negatives per LabelConfig policy), then isotonic calibration on the inner-validation split; `class_weight`/`scale_pos_weight` from train slice only. `CalibratedModel.predict_proba(X) -> np.ndarray` (calibrated P(TP-first)). `select_threshold(model, X_val, y_val, cost_fn) -> float` — pick threshold maximizing expectancy-net-of-cost on validation. `brier_score(model, X, y) -> float`.

- [ ] **Step 1 — failing test** (learnable synthetic signal; asserts mechanics, not magic accuracy):
```python
import numpy as np
from mt5gold.ml.train import train_fold, select_threshold, brier_score

def _data(n=2000, seed=0):
    rng=np.random.default_rng(seed)
    X=rng.normal(0,1,(n,4))
    # label depends on feature 0 (learnable) + noise
    y=((X[:,0] + rng.normal(0,0.5,n)) > 0).astype(int)
    return X, y

def test_model_learns_and_calibrates():
    X,y=_data(); Xtr,ytr,Xval,yval=X[:1500],y[:1500],X[1500:],y[1500:]
    m=train_fold(Xtr,ytr,Xval,yval)
    p=m.predict_proba(Xval)
    assert p.shape[0]==len(yval) and p.min()>=0 and p.max()<=1
    # calibrated model should beat 0.25 Brier (random ~0.25)
    assert brier_score(m, Xval, yval) < 0.24

def test_threshold_selection_returns_prob():
    X,y=_data(); m=train_fold(X[:1500],y[:1500],X[1500:],y[1500:])
    thr=select_threshold(m, X[1500:], y[1500:], cost_fn=lambda pnl: pnl)
    assert 0.0 <= thr <= 1.0
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** (LightGBM + sklearn isotonic via `CalibratedClassifierCV` prefit, or manual `IsotonicRegression`):
```python
# mt5gold/ml/model.py
from __future__ import annotations
import numpy as np

class CalibratedModel:
    def __init__(self, booster, calibrator, feature_list):
        self.booster, self.calibrator, self.feature_list = booster, calibrator, feature_list
    def predict_proba(self, X) -> np.ndarray:
        raw = self.booster.predict(X)
        return self.calibrator.predict(raw)
```
```python
# mt5gold/ml/train.py
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from mt5gold.ml.model import CalibratedModel

def train_fold(X_tr, y_tr, X_val, y_val, params=None) -> CalibratedModel:
    pos = max(1, int((y_tr == 1).sum())); neg = max(1, int((y_tr != 1).sum()))
    p = {"objective": "binary", "verbosity": -1, "num_leaves": 31,
         "learning_rate": 0.05, "scale_pos_weight": neg / pos}
    if params: p.update(params)
    dtr = lgb.Dataset(X_tr, label=(np.asarray(y_tr) == 1).astype(int))
    booster = lgb.train(p, dtr, num_boost_round=200)
    raw_val = booster.predict(X_val)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, (np.asarray(y_val) == 1).astype(int))
    return CalibratedModel(booster, iso, None)

def brier_score(model, X, y) -> float:
    p = model.predict_proba(X); yb = (np.asarray(y) == 1).astype(int)
    return float(np.mean((p - yb) ** 2))

def select_threshold(model, X_val, y_val, cost_fn) -> float:
    p = model.predict_proba(X_val); yb = np.asarray(y_val)
    best_thr, best_ev = 0.5, -1e18
    for thr in np.linspace(0.4, 0.9, 26):
        take = p > thr
        if take.sum() == 0: continue
        # crude EV proxy: +1 reward for correct TP take, -1 otherwise (real cost_fn in script)
        ev = float(np.mean([cost_fn(1.0 if yb[i] == 1 else -1.0) for i in np.where(take)[0]]))
        if ev > best_ev: best_ev, best_thr = ev, float(thr)
    return best_thr
```
- [ ] **Step 4 — run, expect PASS** (if lightgbm determinism varies, set `p["seed"]=0` and `deterministic=True`).
- [ ] **Step 5 — commit:** `git commit -am "feat(phase3): LightGBM training + isotonic calibration + threshold selection"`

---

### Task 6: `ml/registry.py` + `MLStrategy` — fail-closed artifact + gating

**Files:** Create `mt5gold/ml/registry.py`; Modify `mt5gold/core/strategy.py`; Create `tests/test_registry.py`, `tests/test_mlstrategy.py`

**Interfaces:** `save_model(model, path, feature_list, feature_version, label_cfg, threshold, data_hash, shared_config_hash, metrics) -> None` writing model + JSON manifest. `load_model(path, expected_feature_version, expected_config_hash) -> CalibratedModel` — **raises** if `feature_version` or `shared_config_hash` mismatch (fail closed, finding 29/31). `MLStrategy(model, feature_list, label_cfg, threshold)` in `core/strategy.py`: builds the feature vector in `feature_list` order, gets calibrated P, emits BUY when `P>threshold` with SL/TP from `label_cfg` (same geometry as labels), else FLAT.

- [ ] **Step 1 — failing test:**
```python
import numpy as np, pytest
from mt5gold.ml.registry import save_model, load_model
from mt5gold.ml.model import CalibratedModel
from mt5gold.core.strategy import MLStrategy
from mt5gold.core.types import StrategyState
from mt5gold.ml.labeling import LabelConfig
from datetime import datetime, timezone
import pandas as pd

class _Stub:  # stand-in calibrated model
    feature_list=["a","b"]
    def predict_proba(self,X): return np.array([0.8]*len(X))

def test_load_fails_closed_on_version_mismatch(tmp_path):
    save_model(_Stub(),tmp_path/"m", feature_list=["a","b"], feature_version="v1",
               label_cfg=LabelConfig(), threshold=0.6, data_hash="d", shared_config_hash="c", metrics={})
    with pytest.raises(Exception):
        load_model(tmp_path/"m", expected_feature_version="v2", expected_config_hash="c")

def test_mlstrategy_buys_when_prob_above_threshold():
    strat=MLStrategy(_Stub(), feature_list=["a","b"], label_cfg=LabelConfig(), threshold=0.6)
    row=pd.Series({"a":1.0,"b":2.0,"close":2000.0,"atr14":2.0})
    st=StrategyState(None,0,True,300,datetime(2020,1,1,tzinfo=timezone.utc),1000,1000,0,0.0)
    sig=strat.generate_signal(row, st)
    assert sig.side=="BUY" and sig.sl_price is not None
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** registry (pickle/txt model + JSON manifest; `load_model` compares versions and raises on mismatch) and `MLStrategy`:
```python
# add to core/strategy.py
class MLStrategy:
    def __init__(self, model, feature_list, label_cfg, threshold):
        self.model, self.feature_list = model, feature_list
        self.label_cfg, self.threshold = label_cfg, threshold
    def generate_signal(self, f, state) -> Signal:
        if state.open_position is not None or not state.warmup_ready:
            return _flat()
        import numpy as np
        X = np.array([[float(f[c]) for c in self.feature_list]])
        p = float(self.model.predict_proba(X)[0])
        if p <= self.threshold:
            return _flat()
        price, atr = float(f["close"]), float(f["atr14"])
        sl = price - self.label_cfg.k_sl*atr; tp = price + self.label_cfg.k_tp*atr
        return Signal("BUY", p, sl, tp, {"p": p})
```
- [ ] **Step 4 — run, expect PASS** · **Step 5 — commit:** `git commit -am "feat(phase3): model registry (fail-closed) + MLStrategy gating"`

---

### Task 7: `scripts/train_model.py` — walk-forward eval, freeze, holdout-once

**Files:** Create `mt5gold/ml/pipeline.py`, `scripts/train_model.py`, `tests/test_ml_pipeline.py`

**Interfaces:** `run_walk_forward(price_df, features_df, label_cfg, wf_cfg, cost_cfg, spec, b1_metrics, research_log_path) -> dict` — for each fold: inner-split → train+calibrate → select threshold → build `MLStrategy` → run `backtest.engine` on the test fold → collect metrics; aggregate per-fold edge vs B1; run permutation null; append the run to `research_log.jsonl`. `open_holdout_once(...)` — refuses (raises) if the holdout counter for `(data_hash, code_hash)` is already 1; else evaluates once and freezes the model+metrics via registry. `scripts/train_model.py` wires dataset → resample → features → labels → `run_walk_forward` → print robustness-bar verdict; the holdout step is a separate explicit command/flag.

- [ ] **Step 1 — failing test** (mechanics: research log increments; holdout guarded to once):
```python
from mt5gold.ml.pipeline import open_holdout_once, HoldoutAlreadyUsed
import pytest
def test_holdout_can_only_open_once(tmp_path):
    log=tmp_path/"research_log.jsonl"
    open_holdout_once("dhash","chash", log, eval_fn=lambda: {"expectancy":1.0})
    with pytest.raises(HoldoutAlreadyUsed):
        open_holdout_once("dhash","chash", log, eval_fn=lambda: {"expectancy":1.0})
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** `HoldoutAlreadyUsed`, `open_holdout_once` (reads `research_log.jsonl`, counts entries tagged `type="holdout"` with matching hashes; raises if ≥1; else runs `eval_fn`, appends a `holdout` record), and `run_walk_forward` (loops folds using Tasks 2–6; appends a `walk_forward` record per run). `scripts/train_model.py` provides `main()` with `--holdout` flag gating the once-only evaluation.
- [ ] **Step 4 — run, expect PASS**; full suite `python -m pytest -v`.
- [ ] **Step 5 — commit:** `git commit -am "feat(phase3): walk-forward eval pipeline + once-only locked holdout"`

---

## Phase 3 Milestone & Gate
`python scripts/train_model.py --timeframe M15` runs nested walk-forward, calibrates, and reports per-fold ML-vs-B1 edge + permutation p-values, appending to `research_log.jsonl`. **Gate (spec §11):** advance to Phase 4 only if ML beats B1 net-of-cost in ≥70% of folds AND permutation p<0.05 AND (after research is frozen) the once-only locked holdout meets §1.3 criteria. If not → REJECT ML, demo with rule-based.

## Self-Review
- **Spec coverage:** HTF leak-safe ✓(T1, finding 4), triple-barrier via shared resolver ✓(T2, finding 25), purge/embargo≥horizon + locked holdout + inner split ✓(T3, findings 1,2,22), permutation null ✓(T4, finding 21), calibration + threshold-by-EV ✓(T5, finding 44), fail-closed registry + MLStrategy geometry=label geometry ✓(T6, findings 29,32), research log + once-only holdout ✓(T7, findings 3,7,18).
- **Placeholder scan:** T5 `select_threshold` uses a crude EV proxy in the unit test; the real per-trade cost EV comes from the engine in `scripts/train_model.py` (documented). T7 Step 3 describes `run_walk_forward` prose-wise (its full body composes Tasks 2–6 already fully coded) — the implementer assembles them; not a code placeholder for new logic.
- **Type consistency:** `LabelConfig` shared by labeling (T2) + MLStrategy (T6); `resolve_barrier_hit` reused (T2); `CalibratedModel` (T5) consumed by registry (T6) + MLStrategy; `walk_forward_folds`/`inner_split` (T3) used by pipeline (T7); engine/metrics from Phase 1.
- **Deferred:** deep-learning models (spec §14 future); SHAP/permutation-importance diagnostics (finding 45 — add as a diagnostic in T7 output, non-gating); retrain cadence/drift monitor wiring (Phase 4).
