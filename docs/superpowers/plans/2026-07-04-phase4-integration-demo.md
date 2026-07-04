# Phase 4 — ML Integration & Demo Forward-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development / executing-plans. **Prereq:** Phases 0–3 complete and the Phase-3 holdout gate passed.

**Goal:** Run the calibrated ML strategy live on demo through the same shared core, then **prove "fewer mistakes" by quantified reconciliation** — signal agreement, realized-vs-modeled costs, and expectancy within the backtest's confidence interval — plus a drift monitor that throttles the bot when live edge decays.

**Architecture:** `run_live` loads the frozen model fail-closed (feature_version + shared-config hash) and runs `MLStrategy` through the Phase-2 `LiveBot`. A `reconciliation` module compares the decision journal + realized demo trades against a backtest replay of the same bars, emitting the three §13 gates. `safety.py` gains a drift monitor that compares rolling live expectancy to the OOS fold CI and reverts to FLAT/rule-based when live falls below it.

**Tech Stack:** Python 3.11, pandas, numpy, pytest.

## Global Constraints
- Live ML uses the SAME engine/feature/cost code as backtest (findings 13, 28, 47).
- Model load is fail-closed on `feature_version` and `shared_config_hash` (findings 29, 31).
- Demo↔backtest acceptance is the **quantified three-gate test** (finding 37): Gate 1 signal agreement ≥95%; Gate 2 realized slippage/spread ≤ modeled (median) within bound (p90); Gate 3 demo expectancy within backtest bootstrap CI once N≥30 (ideally ≥100).
- Drift monitor: rolling live expectancy below OOS fold CI for a sustained window → throttle to FLAT/rule-based + flag retrain (finding 24).
- No real-money trading (spec §1.2).

---

### Task 1: `live/reconciliation.py` — the three §13 gates

**Files:** Create `mt5gold/live/reconciliation.py`, `tests/test_reconciliation.py`

**Interfaces:**
- `signal_agreement(live_sides: list[str], backtest_sides: list[str]) -> float` — fraction of bars where sides match.
- `cost_gate(realized: list[dict], modeled: list[dict]) -> dict` — compares realized vs modeled slippage/spread; returns `{median_ok: bool, p90_realized, p90_modeled}`.
- `expectancy_gate(demo_pnls, backtest_ci, min_n=30) -> dict` — returns `{status: "pass"|"fail"|"insufficient_n", demo_expectancy}` where demo expectancy must lie within `backtest_ci` when `len>=min_n`.
- `reconcile(...) -> dict` combining all three into a `{gate1, gate2, gate3, verdict}`.

- [ ] **Step 1 — failing test:**
```python
from mt5gold.live.reconciliation import (signal_agreement, cost_gate, expectancy_gate)

def test_signal_agreement():
    assert signal_agreement(["BUY","FLAT","SELL","BUY"], ["BUY","FLAT","SELL","SELL"]) == 0.75

def test_cost_gate_flags_worse_realized():
    realized=[{"slippage":8},{"slippage":9},{"slippage":30}]
    modeled=[{"slippage":10},{"slippage":10},{"slippage":10}]
    g=cost_gate(realized, modeled)
    assert g["median_ok"] is True          # median realized (9) <= median modeled (10)
    assert g["p90_realized"] >= g["p90_modeled"]  # a tail breach exists

def test_expectancy_gate_insufficient_then_pass():
    assert expectancy_gate([1,2], (0.0,3.0), min_n=30)["status"] == "insufficient_n"
    many=[1.0]*40
    g=expectancy_gate(many, (0.5,1.5), min_n=30)
    assert g["status"] == "pass"
    g2=expectancy_gate([5.0]*40, (0.5,1.5), min_n=30)
    assert g2["status"] == "fail"          # 5.0 outside CI
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement:**
```python
from __future__ import annotations
import numpy as np

def signal_agreement(live_sides, backtest_sides) -> float:
    n = min(len(live_sides), len(backtest_sides))
    if n == 0: return 0.0
    return sum(1 for i in range(n) if live_sides[i] == backtest_sides[i]) / n

def cost_gate(realized, modeled) -> dict:
    rs = np.asarray([r["slippage"] for r in realized], float)
    ms = np.asarray([m["slippage"] for m in modeled], float)
    return {"median_ok": bool(np.median(rs) <= np.median(ms)),
            "p90_realized": float(np.percentile(rs, 90)),
            "p90_modeled": float(np.percentile(ms, 90))}

def expectancy_gate(demo_pnls, backtest_ci, min_n=30) -> dict:
    if len(demo_pnls) < min_n:
        return {"status": "insufficient_n", "demo_expectancy": None}
    e = float(np.mean(demo_pnls)); lo, hi = backtest_ci
    return {"status": "pass" if lo <= e <= hi else "fail", "demo_expectancy": e}

def reconcile(live_sides, backtest_sides, realized, modeled, demo_pnls, backtest_ci, min_n=30) -> dict:
    g1 = signal_agreement(live_sides, backtest_sides)
    g2 = cost_gate(realized, modeled)
    g3 = expectancy_gate(demo_pnls, backtest_ci, min_n)
    verdict = "pass" if g1 >= 0.95 and g2["median_ok"] and g3["status"] in ("pass","insufficient_n") else "investigate"
    return {"gate1_agreement": g1, "gate2_cost": g2, "gate3_expectancy": g3, "verdict": verdict}
```
- [ ] **Step 4 — run, expect PASS** · **Step 5 — commit:** `git commit -am "feat(phase4): demo/backtest reconciliation (three §13 gates)"`

---

### Task 2: drift monitor in `live/safety.py`

**Files:** Modify `mt5gold/live/safety.py`; Create `tests/test_drift.py`

**Interfaces:** `drift_throttle(recent_pnls, oos_ci, min_trades=20) -> bool` — returns True (throttle → FLAT / revert to rule-based) when there are ≥ `min_trades` recent trades AND their mean expectancy is below the lower bound of the OOS fold CI. Wire an optional `oos_ci` + rolling-pnl buffer into `LiveBot` so a drifting model stops trading before large losses accrue.

- [ ] **Step 1 — failing test:**
```python
from mt5gold.live.safety import drift_throttle
def test_drift_throttle_triggers_below_ci():
    assert drift_throttle([-1.0]*25, oos_ci=(0.2, 1.5), min_trades=20) is True
def test_drift_no_throttle_when_within_ci():
    assert drift_throttle([0.8]*25, oos_ci=(0.2, 1.5), min_trades=20) is False
def test_drift_insufficient_trades():
    assert drift_throttle([-5.0]*5, oos_ci=(0.2,1.5), min_trades=20) is False
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** (append to `safety.py`):
```python
import numpy as np
def drift_throttle(recent_pnls, oos_ci, min_trades=20) -> bool:
    if len(recent_pnls) < min_trades: return False
    lo, _ = oos_ci
    return float(np.mean(recent_pnls)) < lo
```
- [ ] **Step 4 — run, expect PASS** · **Step 5 — commit:** `git commit -am "feat(phase4): live edge-drift throttle"`

---

### Task 3: wire `MLStrategy` into `run_live` (fail-closed load)

**Files:** Modify `scripts/run_live.py`; Create `tests/test_run_live_ml.py`

**Interfaces:** `build_strategy_from_model(model_path, feature_version, config_hash, label_cfg, threshold) -> MLStrategy` — loads the model via `registry.load_model` (raises on version/config mismatch) and returns a wired `MLStrategy`. `run_live.py` gains `--model PATH` to run ML instead of rule-based; without it, runs rule-based (safe default).

- [ ] **Step 1 — failing test** (fail-closed on mismatch; success path wires MLStrategy):
```python
import pytest
from mt5gold.ml.registry import save_model
from mt5gold.ml.labeling import LabelConfig
import numpy as np
class _Stub:
    feature_list=["a"]
    def predict_proba(self,X): return np.array([0.7]*len(X))

def test_build_strategy_rejects_version_mismatch(tmp_path):
    from scripts.run_live import build_strategy_from_model  # import via exec in conftest if needed
    save_model(_Stub(), tmp_path/"m", feature_list=["a"], feature_version="v1",
               label_cfg=LabelConfig(), threshold=0.6, data_hash="d", shared_config_hash="c", metrics={})
    with pytest.raises(Exception):
        build_strategy_from_model(tmp_path/"m","v2","c",LabelConfig(),0.6)
```
> If importing `scripts/run_live.py` directly is awkward, move `build_strategy_from_model` into `mt5gold/live/wiring.py` and import from there (cleaner; do that).
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** `mt5gold/live/wiring.py::build_strategy_from_model` (calls `load_model`, constructs `MLStrategy`) and update `scripts/run_live.py` to use it under `--model`.
- [ ] **Step 4 — run, expect PASS** · **Step 5 — commit:** `git commit -am "feat(phase4): fail-closed ML model wiring into run_live"`

---

### Task 4: `scripts/reconcile.py` — Phase-4 forward-test report

**Files:** Create `mt5gold/live/report.py`, `scripts/reconcile.py`, `tests/test_report.py`

**Interfaces:** `build_report(journal_rows, demo_trades, backtest_replay, backtest_ci, modeled_costs) -> dict` — assembles the three gates (Task 1) + drift status into a single report dict; `write_report(report, path)`. `scripts/reconcile.py` reads the demo decision journal + MT5 closed-deal history (via `broker`) + a backtest replay of the same bars, and prints the verdict.

- [ ] **Step 1 — failing test:**
```python
from mt5gold.live.report import build_report
def test_report_assembles_gates():
    r=build_report(
        journal_rows=[{"side":"BUY"},{"side":"FLAT"}],
        demo_trades=[{"pnl":1.0}]*40,
        backtest_replay=["BUY","FLAT"],
        backtest_ci=(0.5,1.5),
        modeled_costs=[{"slippage":10}]*40)
    assert r["verdict"] in ("pass","investigate")
    assert "gate1_agreement" in r and "gate3_expectancy" in r
```
- [ ] **Step 2 — run, expect FAIL**
- [ ] **Step 3 — implement** `build_report` (maps inputs into `reconcile`, adds drift status), `write_report` (JSON), and `scripts/reconcile.py` (`main()` gluing journal + broker history + backtest replay).
- [ ] **Step 4 — run, expect PASS**; full suite `python -m pytest -v`.
- [ ] **Step 5 — commit:** `git commit -am "feat(phase4): forward-test reconciliation report + CLI"`

---

## Phase 4 Milestone & Gate
After running the ML bot on demo for a meaningful sample, `python scripts/reconcile.py` emits a report with: signal agreement %, realized-vs-modeled cost comparison, demo expectancy vs backtest CI, and drift status. **Final gate (spec §1.3 / §13):** demo tracks backtest within tolerance (Gates 1–2 pass, Gate 3 pass or insufficient-N) AND the Phase-3 locked-holdout criteria held → **"fewer mistakes" is proven with numbers**, decomposed as (B0→B1) bug-fix gain + (B1→ML) AI gain. Only then is a real-money discussion on the table (out of scope here).

## Self-Review
- **Spec coverage:** three quantified §13 gates ✓(T1, finding 37), drift throttle ✓(T2, finding 24), fail-closed ML load ✓(T3, findings 29/31), forward-test report + (B0→B1→ML) attribution ✓(T4, finding 35).
- **Placeholder scan:** T3/T4 Step 3 describe glue that composes already-coded units (registry, MLStrategy, reconcile); the algorithmic pieces are fully coded in their own steps. Live-dependent behavior (real order fills) is validated manually per DEV_SETUP, consistent with demo-only scope.
- **Type consistency:** reuses `MLStrategy`/`load_model`/`LabelConfig` (Phase 3), `LiveBot`/`journal` (Phase 2), `bootstrap` CI shape `(lo,hi)` (Phase 1).
- **Deferred to Future Work (spec §14):** real-money deployment, retrain automation/scheduling, deep-learning models — explicitly out of scope.
