# MT5 Gold AI Trading — Design Spec (แนวทาง B) · v2

- **วันที่:** 2026-07-03 (v2 ผ่านการตรวจ adversarial 47 findings, 2026-07-04)
- **สถานะ:** Draft (รอ user review)
- **Symbol:** XAUUSD (ทองคำ) บน MetaTrader 5
- **เป้าหมายระดับสูง:** พิสูจน์ด้วยตัวเลขว่า "พลาดน้อยลง" บน backtest + demo ก่อนตัดสินใจเรื่องเงินจริง — โดย **ออกแบบมาเพื่อกันการหลอกตัวเอง (self-deception) เป็นหลัก**
- **ผู้ใช้:** เขียน Python ได้ / ML ยังใหม่ → เอกสารนี้อธิบายแนวคิด ML ที่จำเป็นด้วย
- **เครื่อง:** Ryzen 7 4800H / RTX 3050 / RAM 24GB → โมเดลหลักเป็น tabular (LightGBM) บน CPU

> **หมายเหตุ v2:** ฉบับนี้เข้มขึ้นมากจากการตรวจ adversarial 5 มุมมอง (leakage, backtest realism, ML validation, architecture, completeness) จุดที่เพิ่มเข้ามาคือ: locked holdout + research log, cost model ครบ (swap/spread per-bar/gap), shared barrier resolver, feature_row + WARMUP_BARS, model registry, config hash, และ Go/No-Go kill gates

---

## 1. เป้าหมาย, ขอบเขต, และเกณฑ์วัดความสำเร็จ

### 1.1 Goals
1. สร้าง **รากฐานที่ซื่อสัตย์**: data pipeline สะอาด + backtest engine ที่จำลองต้นทุนจริง (spread/commission/slippage/**swap**)
2. **ซ่อมบั๊กวิกฤต**ของบอทเดิม
3. สร้าง **ML pipeline** ที่ใช้ AI เป็นตัวคัดกรองสัญญาณ
4. ทุกการอ้างว่า "ดีขึ้น" ต้องพิสูจน์ด้วย **out-of-sample metric ที่กัน overfitting + multiple-testing**

### 1.2 Non-Goals
- ไม่เทรดเงินจริง (demo เท่านั้นจนกว่าจะพิสูจน์ผ่าน) · ไม่ทำ RL/LSTM/Transformer เฟสแรก · ไม่ทำ multi-symbol · ไม่ทำ full requote/partial-fill simulation (log demo reject แทน)

### 1.3 เกณฑ์วัดความสำเร็จ (วัดบน **locked holdout** ที่แตะครั้งเดียว + หลังหักต้นทุนทุกชนิด)
- **Expectancy/ไม้:** 95% CI **lower bound > 0** (ไม่ใช่แค่ค่ากลาง > 0)
- **Profit Factor:** ค่ากลาง ≥ 1.2 **และ** CI lower bound ≥ 1.0
- **Max Drawdown:** ≤ 15% ของ equity
- **Deflated Sharpe Ratio (DSR) > 0** ที่ 95% เมื่อคิด N_trials ที่ทดลองไปทั้งหมด
- **ML ชนะ baseline B1 (cleaned rule-based)** แบบ **robustness bar** (ดู §2 หลักการ 7 — ไม่ใช่ "ชนะทุกหน้าต่าง")
- **ผล demo forward-test เกาะ backtest** ตาม tolerance ที่ quantify ใน §13
- **ต้อง pre-register ทุกเกณฑ์ตัวเลขนี้เป็นลายลักษณ์อักษรก่อนเปิด holdout**

### 1.4 Baseline Definition (frozen) — ตัวเลขที่โปรเจกต์ถูกตัดสิน
รันผ่าน engine + cost เดียวกัน บน OOS window เดียวกัน:
- **B0 = "Legacy reconstruction":** พฤติกรรมบอทเดิมรวม Smart-Guess enter-every-candle → ตอบว่า "เดิมเราทำ/เสียอะไรจริง"
- **B1 = "Cleaned rule-based":** `RuleBasedStrategy` ที่มี FLAT และตัด guess ทิ้ง → **edge floor ที่ ML ต้องเอาชนะ**
- Freeze B0/B1 เป็น JSON artifact (พร้อม dataset hash) ก่อนเริ่มงาน ML — timeframe/ช่วงข้อมูล/cost ต้องนิ่ง
- รายงานสุดท้ายต้องแยก: **(B0→B1) = ได้จากแก้บั๊ก**, **(B1→ML) = ได้จาก AI**

---

## 2. หลักการออกแบบ (Design Principles)

1. **Shared Core = "สิ่งที่เทสต์ = สิ่งที่เทรดจริง"**: feature, strategy, cost model, **และ barrier resolver** เป็นชุดเดียวที่ทั้ง backtest/ML/live เรียกใช้
2. **Causality เป็นกฎเหล็ก บังคับด้วยเทสต์ ไม่ใช่แค่ประกาศ**: feature แท่ง t ใช้ข้อมูลถึง close ของ t เท่านั้น; เข้าออเดอร์ที่ open ของ t+1
3. **FLAT คือคำตอบที่ถูกต้อง** (ตัด logic เดาทิ้ง)
4. **แต่ละไฟล์มีหน้าที่เดียว** เทสต์ได้อิสระ
5. **Provenance / Reproducibility แบบ fail-closed**: dataset, model, และ shared-config มี hash lineage; live ปฏิเสธเทรด (fail closed) ถ้า hash ไม่ตรง
6. **วัดผลเป็นเงิน + ช่วงความเชื่อมั่น** ไม่ใช่ % accuracy ของ classifier
7. **Pre-registration + robustness bar (กัน multiple-testing)**: freeze เกณฑ์ตัวเลขก่อนดูผล; เกณฑ์ผ่านคือ "expectancy หลังต้นทุนเป็นบวกใน **≥70% ของหน้าต่าง walk-forward** + dispersion ต่ำกว่าขอบเขต + ไม่มีหน้าต่างไหนต่ำกว่า floor" — ไม่ใช่ "ชนะทุกหน้าต่าง" (เปราะและชวนให้จูนจนเขียว); ทุก config ที่ลองนับเข้า **research log**

---

## 3. สถาปัตยกรรม & โครงสร้างโปรเจกต์

```
mt5gold/
├── data/
│   ├── fetch.py       # ดึงประวัติ (ผ่าน broker.py)
│   ├── clean.py       # gap, tz, resample, validate, เก็บ spread column
│   └── store.py       # parquet + manifest (รวม contract spec snapshot)
├── core/              # ◆ SHARED — backtest/ml/live ใช้ร่วม
│   ├── types.py       # Bar, Signal, Trade, Position, StrategyState (dataclasses)
│   ├── features.py    # feature_row (1 แถว) + build_features (bulk) — ค่าเท่ากัน
│   ├── barriers.py    # resolve_barrier_hit() — ใช้ทั้ง labeling + engine
│   ├── strategy.py    # Strategy Protocol + RuleBasedStrategy + MLStrategy
│   └── costs.py       # spread/commission/slippage/swap + sizing + min-stop-distance
├── backtest/
│   ├── engine.py      # event-driven loop (ML ก็เดินผ่าน engine นี้)
│   └── metrics.py     # WR, PF, Expectancy, MaxDD, Sharpe/DSR, bootstrap CI, equity curve
├── ml/
│   ├── labeling.py    # triple-barrier (เรียก barriers.py)
│   ├── dataset.py     # X, y + fold-local weights
│   ├── train.py       # LightGBM + calibration
│   ├── validate.py    # nested walk-forward + purge/embargo + locked holdout
│   ├── model.py       # โหลด/ทำนาย
│   └── registry.py    # model manifest (feature_version, label params, hashes)
├── live/
│   ├── broker.py      # ◆ โมดูลเดียวที่ import MetaTrader5 (mock/replay ได้)
│   ├── bot.py         # loop สด (rolling buffer ≥ WARMUP_BARS)
│   ├── execution.py   # order (type_filling, deviation จาก config)
│   ├── safety.py      # demo lock, spread guard, daily/DD limit, weekend-flat, drift monitor
│   └── journal.py     # decision log (JSON ต่อ 1 การตัดสินใจ)
├── config.py          # dataclasses แยก SHARED(pinned) / LIVE-only + secret ผ่าน env
└── research_log.jsonl # append-only: ทุก evaluation run (กัน data-snooping)

scripts/  · tests/  · docs/{superpowers/specs, diagrams(ไม่ commit)}
```

`trading_bot.py` เดิม: เก็บไว้จน migrate ครบ · **Tech stack:** Python 3.11 · MetaTrader5 · pandas · numpy · pyarrow · lightgbm · scikit-learn · pytest

---

## 4. Data Pipeline (`mt5gold/data/`)

### 4.1 `fetch.py`
- ดึง **M1 เป็น base** ผ่าน `broker.py` → `copy_rates_range` แบ่ง chunk; ดึงย้อนหลังมากสุดที่มี (2–5 ปี)
- **M1 ต้องมีเสมอ**คู่กับ timeframe ที่เทรด (engine ต้องใช้ไล่ intrabar — ดู §6)
- **OHLC ของ MT5 เป็น bid-based** → ต้อง reconstruct ฝั่ง ask จาก spread column (บันทึก convention นี้)

### 4.2 `clean.py` — กับดักเฉพาะทองคำ
1. **Broker time → UTC** (ตรวจ offset จริง เก็บใน manifest)
2. **Weekend/holiday gap:** ไม่สร้างแท่งผี (ใช้ session calendar)
3. **`tick_volume` = proxy** ไม่ใช่ volume จริง
4. **เก็บ `spread` column** (หน่วย points) + validate: reject bar ที่ spread หาย/≤0, flag bar ที่ spread กว้างผิดปกติ (ห้ามเฉลี่ยกลบ)
5. **Validation (fail ถ้าไม่ผ่าน):** timestamp เรียง/ไม่ซ้ำ · ระยะห่างตาม session · `high≥max(o,c)`, `low≤min(o,c)` · จับ bad tick

### 4.3 `store.py`
- **Parquet** แยก partition `symbol/timeframe/`, แยก `raw/` กับ `clean/`
- **Manifest (JSON):** เวลาดึง, ช่วง, broker, server, tz offset, จำนวนแท่ง, hash, **spread ต่ำสุด/ทั่วไป**, และ **contract spec snapshot** (contract_size, tick_size/value, volume_min/max/step, `trade_stops_level`, `trade_freeze_level`, `filling_mode`, `swap_long/short/mode`, `account_currency`) → backtest sizing/cost อ่านจาก snapshot นี้ (ไม่เรียก mt5 สดตอน replay)
- resample M1→M5/M15/H1 (OHLCV มาตรฐาน เคารพ session) · PostgreSQL เดิม = log เทรดสดเท่านั้น

---

## 5. Shared Core (`mt5gold/core/`)

### 5.1 `types.py`
```python
@dataclass(frozen=True)
class Signal: side: Literal["BUY","SELL","FLAT"]; confidence: float; sl_price: float|None; tp_price: float|None; meta: dict
@dataclass(frozen=True)
class Position: side: str; entry_price: float; sl: float; tp: float; lot: float; entry_time: datetime
@dataclass(frozen=True)
class StrategyState:  # ทั้ง backtest และ live ต้องสร้าง object นี้ด้วย semantics เดียวกัน
    open_position: Position|None; bars_held: int; warmup_ready: bool
    bar_index: int; bar_time: datetime; balance: float; equity: float
    trades_today: int; daily_pnl: float
```
`Bar`, `Trade` เป็น dataclass เช่นกัน

### 5.2 `features.py` — causal, สองรูปแบบต้องให้ค่าเท่ากัน
- **`feature_row(window, htf_window) -> pd.Series`** = คำนวณ 1 แถวจาก trailing window; **`build_features(df, df_htf) -> DataFrame`** = เวอร์ชัน bulk ที่ให้ค่าเท่ากันทุกแถว (มีเทสต์ยืนยัน)
- **`WARMUP_BARS`** = lookback ยาวสุด (max ของ EMA50, ATR-percentile window, swing N, HTF EMA) + margin (EWM ต้อง converge)
- Feature families:
  - **Technical:** EMA(9,21,50), RSI(14), ATR(14), MACD, Bollinger + slope/ระยะ
  - **Price action:** body/wick, ระยะถึง swing high/low N (**trailing เท่านั้น ห้าม centered/forward, ใช้ถึงแท่งปิด t**), ระยะถึง zone
  - **Volatility regime:** ATR percentile บน **fixed trailing window (เช่น 252)** คำนวณ causal — **ห้าม expanding/full-sample rank**
  - **Time/session:** ชั่วโมง UTC, one-hot Asian/London/NY, day-of-week
  - **Multi-timeframe (HTF):** เทรนด์ H1 attach ด้วย `merge_asof(...direction='backward')` โดย key H1 ที่ **close timestamp** → แท่ง M15 10:15 ต้องได้ค่า H1 ของ 09:00–10:00 **ห้าม**ได้ 10:00–11:00 (ยังไม่ปิด)

### 5.3 `strategy.py`
```python
class Strategy(Protocol):
    def generate_signal(self, features_row: pd.Series, state: StrategyState) -> Signal: ...
```
- `RuleBasedStrategy` (zone+EMA/RSI สะอาด, FLAT ได้) · `MLStrategy` (เข้าเมื่อ **calibrated P > threshold**)
- **MLStrategy ต้อง emit sl/tp/time-exit จาก k_tp/k_sl/N ชุดเดียวกับที่ใช้ label** (ผ่าน `LabelConfig` ร่วม) → P(TP ก่อน SL) จึงตรงกับผลจริง
- ทั้งสองเส้นทางสร้าง `StrategyState` เหมือนกัน (live จาก positions_get/account_info, backtest จาก simulated book); `warmup_ready=False` → บังคับ FLAT

### 5.4 `costs.py`
- `apply_costs(fill_price, side, spread_at_bar, commission_per_lot, slippage_model)` — **spread เป็น per-bar จริง** (floor ที่ broker min), slippage เป็น **โมเดลขึ้นกับ regime** (`base + f(ATR_pctile, spread)`, exit ฝั่ง SL/stop สูงกว่า entry)
- `financing_cost(side, lots, entry_time, exit_time, spec)` — **swap ข้ามคืน** (รวม triple-swap วันพุธ) อ่านจาก snapshot
- `position_size(balance, risk_pct, sl_distance_price, spec)` — สูตรถูกต้อง คูณ value_per_price_unit จาก snapshot; แปลง account currency ถ้าไม่ใช่ USD
- `enforce_min_stop_distance(entry, sl, tp, side, spec) -> (sl, tp, action)` — ถ้า SL/TP ชิดกว่า `max(stops_level, freeze_level)` → **skip (FLAT)** หรือ widen (เลือกนโยบายเดียว ใช้ทั้ง backtest+live+labeling)

### 5.5 `barriers.py` — resolver ร่วม
`resolve_barrier_hit(bar, m1_path, sl, tp, side) -> +1/-1/0` — ตรรกะ first-touch เดียวที่ **ทั้ง engine และ labeling import** (Design Principle 1)

---

## 6. Backtest Engine (`mt5gold/backtest/`) — หัวใจ

### 6.1 `engine.py` — event-driven (ML ก็เดินผ่านนี้)
- ตัดสินบนแท่งปิด → เข้าที่ open ของ t+1; **ML ก็เดิน bar-by-bar เรียก MLStrategy.generate_signal เหมือน rule-based** (ห้ามคำนวณ P&L แบบ vectorized ตรงจาก label — จะข้าม state: position เดียวต่อครั้ง, trailing, time-exit)
- **Data contract:** engine รับ decision-timeframe bars + **M1 series ที่ align เสมอ**
- **Fill & cost convention:**
  - OHLC = bid → BUY เข้า/ออกฝั่ง ask, SELL ฝั่ง bid (spread จาก column)
  - **Gap check ก่อน:** ถ้า open เลย SL/TP ไปแล้ว → fill ที่ open (ราคา gap) ไม่ใช่ที่ barrier → loss ใหญ่กว่า nominal
  - **Fill asymmetry:** SL/stop = market → บวก slippage (แย่กว่า level); TP = limit → fill ที่ level พอดี
  - **Reject-if-moved:** ถ้า fill เลย `deviation` จาก decision price → นับเป็น reject (เหมือน live)
- **Intrabar/first-touch:** ใช้ `barriers.resolve_barrier_hit` — default = ไล่ M1 path; pessimistic assume-SL-first เฉพาะเมื่อ M1 หายจริง (นับ `fallback_bar_rate`)
- **Trailing stop:** ต้องไล่ด้วย M1 (cadence/min-step ตรง live); **ห้ามไล่ด้วย high/low ของแท่ง (lookahead)**; Phase 1 baseline ใช้ SL/TP คงที่ไปก่อน (ระบุชัด)
- **Weekend policy (ร่วม backtest+live):** `FORCE_FLAT` (ปิดก่อนศุกร์ปิดตลาด) หรือ `HOLD_OVER_WEEKEND` (จันทร์เปิด resolve ที่ราคา gap)

### 6.2 `metrics.py`
จำนวนไม้ · WR · PF · Expectancy/ไม้ · Avg win/loss · MaxDD · Sharpe/Sortino · **DSR (คิด N_trials)** · **bootstrap CI (block/stationary) ของ Expectancy/PF/Sharpe** · ต้นทุนรวม (แยก spread/commission/slippage/swap) · **realized-spread ต่อ session** · `fallback_bar_rate` · equity curve · แยกผลตาม session/ทิศทาง

### 6.3 กฎกัน leakage (บังคับในโค้ด)
- ไม่มี distributional transform (percentile/rank/z-score/scaler) ที่ fit เกินแท่ง t หรือ fit บน train+test รวมกัน — fit บน train fold แล้ว apply ไปหน้า
- bid/ask side convention เดียวกันทั้ง features/labels/fills/live

---

## 7. Strategy + ซ่อมบอทสด (`mt5gold/live/`)

### 7.1 ตารางบั๊กวิกฤต → วิธีแก้
| บั๊กเดิม (ไฟล์:บรรทัด) | ผล | วิธีแก้ |
|---|---|---|
| `history_rsi_value` NameError (`:654`) | สัญญาณ zone/EMA crash | context ครบใน strategy |
| `ORDER BY ASC LIMIT 20` (`:140`) | ดึงแท่งเก่า | rolling window ถูกต้อง |
| บังคับเข้าทุกแท่ง | แพ้สถิติ | FLAT = ปกติ |
| วิเคราะห์แท่งไม่ปิด (`:199,249`) | สัญญาณเพี้ยน | ตัดสินบนแท่งปิด |
| กำไร cap $1 / loss เต็ม (`:547`) | expectancy ติดลบ | exit สอดคล้อง RR |
| sizing ไม่คูณ tick value (`:308`) | 29 lots | สูตรถูก + snapshot |
| **30s post-candle delay (`:586`)** | live เข้าไม่ตรง next-open → skew | เข้า next-open เหมือน backtest (shared ENTRY_TIMING) |
| hardcode password (`:33`), init ทุก loop (`:518`) | ปลอดภัย/สิ้นเปลือง | env + init ครั้งเดียว (ใน broker.py) |
| trailing ไม่มี min-stop floor (`:371`) | order ถูก reject | ผ่าน enforce_min_stop_distance |

### 7.2 `bot.py`
รักษา rolling buffer ≥ `WARMUP_BARS` (rehydrate ตอน start/reconnect ไม่งั้น FLAT) → `feature_row` (shared) → `generate_signal` → `safety` → `execution`

### 7.3 `safety.py`
demo-lock · spread guard (shared predicate กับ backtest) · daily loss/profit limit · **max DD → pause** · **weekend-flat** · **drift monitor** (เทียบ live/demo กับ OOS fold CI → ถ้าหลุด → throttle/กลับไป rule-based/flag retrain)

### 7.4 `execution.py` / Timeframe
- ตั้ง `type_filling` จาก `symbol_info.filling_mode` (ไม่ปล่อยว่าง = กัน reject 10030), `deviation` จาก config
- Timeframe เป็น config; default แนะนำ M15/H1 — แต่ **ต้อง fix ก่อนเริ่มงาน ML** (กันจูน TF ให้เข้าข้าง ML)

---

## 8. ML Pipeline (`mt5gold/ml/`)

### 8.1 Labeling — Triple-Barrier (`labeling.py`, เรียก `barriers.py`)
- 3 เส้น อิง ATR ที่ close ของ t: **TP=entry+k_tp·ATR**, **SL=entry−k_sl·ATR**, **เวลา=N แท่ง** — `k_tp,k_sl,N` เป็น **`LabelConfig` ร่วม** (เช่น N=24 บน M15 ≈ 6 ชม.)
- **Causal เหมือน engine:** entry = open ของ t+1 (ไม่ใช่ close ของ t); first-touch ไล่จาก t+1 ผ่าน `barriers.resolve_barrier_hit` (M1 path / pessimistic เหมือน §6.1); **ห้ามใช้ high/low ของแท่ง t ตัดสิน touch**
- **side/spread-aware:** long ประเมิน barrier บน bid, entry ที่ ask (ผ่าน costs.py) — label +1 = ชน TP ได้จริงหลังข้าม spread
- label geometry = tradable geometry เป๊ะ (label_SL==tradable_SL ฯลฯ); enforce min-stop เหมือน trade จริง

### 8.2 Dataset (`dataset.py`)
- `X`=feature row, `y`=label · **binary "TP-before-SL"**; timeout(0) → assign ตามผลจริง ณ vertical barrier (net cost) **ไม่ทิ้ง** (กัน survivorship); ถ้าจะเป็น "no-trade" → route ไป FLAT ไม่ใช่รวมเข้า SL
- **class weight / uniqueness weight คำนวณจาก train slice ของแต่ละ fold เท่านั้น** (ไม่ใช่ทั้งชุด) · ไม่ใช้ SMOTE เฟส 1

### 8.3 Model (`train.py`)
- **LightGBM** (`scale_pos_weight`) → P(TP ก่อน SL) · hyperparameter จูนบน **inner-validation เท่านั้น**
- **Feature importance = diagnostic เท่านั้น** ใช้ permutation/SHAP บน OOS (ไม่ใช่ default gain บน train); ใช้จับ leakage (feature เด่นผิดปกติ = สงสัยบั๊ก); เปลี่ยน feature = trial ใหม่ นับเข้า budget

### 8.4 Validation (`validate.py`) — จุดตายของมือใหม่
- **Nested walk-forward:** ในแต่ละ outer-train เจียก **inner-validation** (purge/embargo) เลือก threshold+hyperparameter ที่นั่น → **freeze ก่อนเห็น outer-test**; outer-test ให้คะแนน **ครั้งเดียว**
- **Purge + Embargo เชิงปริมาณ:** purge ทุก training sample ที่ label window `[t, t+span]` คาบเกี่ยว test; **embargo ≥ N** (อ่าน N จาก LabelConfig)
- **Locked holdout:** กันช่วงล่าสุด **6–12 เดือน** ไม่แตะระหว่างวิจัย → เปิดครั้งเดียวตอนจบ ใช้เป็นเกณฑ์ §1.3
- **Calibration:** fit isotonic/Platt บน inner-validation (หลัง purge/embargo) → รายงาน reliability + Brier; threshold เลือกบน **calibrated P** ด้วย objective **expectancy-net-of-cost** (ไม่ใช่ accuracy/F1), re-derive ต่อ fold
- **Baseline เทียบ 2 แบบ:** (1) frozen **B1** rule-based, (2) **matched permutation null** — entry timestamp/count เท่า ML สุ่มทิศ ≥1000 รอบ ต้องเกิน 95th percentile
- **Data-adequacy gate:** รายงาน effective N (คิด label overlap) + **min ≥100–200 closed trades/fold**; fold ต่ำกว่า = "low confidence" ไม่นับ pass/fail; tag regime ต่อ fold, flag train/test regime mismatch
- **Window:** rolling 18–24 เดือน (เทียบ rolling vs expanding เชิงประจักษ์)
- **Research log:** ทุก evaluation run append เข้า `research_log.jsonl` (params+metrics+counter) → N_trials auditable; holdout eval counter ต้อง = 1

### 8.5 Integration
`MLStrategy` เข้าเมื่อ calibrated `P > threshold` (per-fold) — AI = ตัวคัดกรองพลาดน้อยลง

### 8.6 ความคาดหวังตามจริง
edge เล็ก · ชนะ = กรองไม้แย่ · **point estimate ที่ CI คร่อม 0 = "ยังไม่พิสูจน์" ไม่ใช่ผ่าน**

### 8.7 Provenance (model artifact)
`registry.py` เก็บ manifest: model_id, **feature_list (มีลำดับ)**, **feature_version (hash ของ features.py — ไม่ใช่แค่ schema)**, dataset hash, k_tp/k_sl/N, threshold, fold bounds, lib versions, OOS metrics, **shared-config hash**

---

## 9. Error Handling & Edge Cases
- MT5 หลุด → reconnect + backoff (ใน `broker.py`) + **re-warm buffer** ก่อนเทรด
- ข้อมูลไม่พอ/`warmup_ready=False` → FLAT
- spread กว้าง (ข่าว) → ไม่เข้า (shared guard)
- broker reject (No money/requote/off-quote/filling) → log retcode + reason, ไม่ blind-retry
- DST/tz เปลี่ยน → ตรวจ offset ทุกครั้ง fetch
- **fail-closed:** feature_version / schema hash / shared-config hash ไม่ตรง artifact → ปฏิเสธเทรด
- required env secret หาย → fail-fast ตอน start (ไม่ log secret)

## 10. Testing Strategy
- **Unit:** costs (sizing/cost/swap/min-stop), barriers (first-touch), labeling (ตรง engine, side/spread-aware), metrics/CI
- **Per-feature causality (gate):** truncation-equality — `feature_row(data[:t+1])` == `build_features(full).loc[t]` ทุก feature; buffer < WARMUP_BARS ต้องถูกปฏิเสธ
- **Leakage tests:** (ก) inject future feature → ต้องจับได้; (ข) เปิด/ปิด purge-embargo (<N) → OOS PF ต้องพองผิดปกติ (พิสูจน์ guard ทำงาน); (ค) percentile ขยาย window เข้า test → จับได้; (ง) fold-local weight ไม่เปลี่ยนเมื่อตัด test
- **Backtest realism:** gap-through-SL → loss > 1R; stop มี slippage / limit ไม่มี; swap รวม triple วันพุธ; BUY vs SELL ต่างกันเท่า spread
- **Parity (สำคัญ):** (A) engine เทียบ (B) **live feature path จริง** (rolling buffer + per-bar) บน bar sequence เดียว → Signal ตรงกันหลัง warm-up; **offline live-replay harness** (mock broker.py) เป็น gate ของ Phase 2; cold-restart parity; StrategyState ตรงกัน; shared-config hash ตรงกัน
- **ML-engine parity:** per-bar prediction ผ่าน engine object == ผล (ไม่มี vectorized P&L แยก)

## 11. เฟส & Milestones + Go/No-Go Gates (pre-registered)
| เฟส | ทำ | Milestone | **Gate (fix ตัวเลขก่อนดูผล)** |
|---|---|---|---|
| 0 | scaffold + config + broker.py + fetch M1→parquet | dataset reproducible | manifest+hash ครบ |
| 1 | engine + cost(รวม swap) + B0 + B1 | ได้ **B0 และ B1** จริง | **ถ้า B1 OOS expectancy ≤0 หรือ PF<1.0 → STOP** (กลยุทธ์เดิมไม่มี edge; ห้ามเสียบ ML ทับ) |
| 2 | ซ่อม+refactor live ใช้ core ร่วม (demo) + replay harness | demo ตรง backtest | signal-agreement ≥95% + cost gate ผ่าน |
| 3 | features + label + LightGBM + nested walk-forward + calibration | ชนะ B1 (robustness bar) + permutation | **ถ้า expectancy หลังต้นทุนเป็นบวก < 70% ของหน้าต่าง (เช่น < 4 จาก 5) หรือแพ้ permutation null → REJECT ML, demo ด้วย rule-based** |
| 3.5 | เปิด **locked holdout ครั้งเดียว** | ผ่านเกณฑ์ §1.3 บน holdout | **ถ้า holdout ไม่ผ่าน → ถือว่า edge ลวง, ห้าม re-tune** |
| 4 | เสียบ ML → forward-test demo + reconciliation | demo เกาะ backtest | **ถ้า demo เบี่ยง > tolerance (§13) → หยุด, สอบ skew** |

## 12. Risks & Mitigations
- **Overfitting/multiple-testing:** nested walk-forward + purge/embargo≥N + locked holdout ครั้งเดียว + DSR + research log + robustness bar (ไม่ใช่ perfection bar)
- **Backtest เกินจริง:** cost ครบ (swap/spread per-bar/gap-through/SL-slippage) + pessimistic M1 fallback (นับ rate) + stress spread ×1.5/×2 + parity path B
- **Data snooping:** research log counter + holdout eval = 1
- **ข้อมูล/regime จำกัด:** effective N + min trades/fold + regime tagging
- **Non-stationarity:** rolling window + retrain cadence + drift monitor → throttle
- **Backtest history ≠ live tape:** spread เป็น lower-bound → stress test; ตีความ demo divergence ว่าอาจเป็น cost understatement
- **Train/live skew:** shared core + broker.py mock + replay harness + hash fail-closed
- **Scope creep:** ยึด non-goals

## 13. Demo ↔ Backtest Reconciliation (quantify tolerance)
- **Gate 1 (signal skew, N เล็ก):** replay bar ที่ demo เห็นผ่าน engine → decision agreement **≥95%**
- **Gate 2 (execution/cost):** ทุก fill เทียบ realized vs modeled slippage/spread → median realized ≤ modeled, p90 ภายในกรอบ; ถ้า realized > modeled เป็นระบบ → re-calibrate cost ก่อนอ้าง "พิสูจน์"
- **Gate 3 (expectancy, ต้องมี N):** เมื่อ demo ≥30 (ดี ≥100) ไม้ → expectancy/ไม้ ต้องอยู่ใน bootstrap CI ของ backtest; ต่ำกว่า N = "ข้อมูลไม่พอ" ไม่ตัดสิน
- Phase 2/4 ผ่านเมื่อ Gate 1,2 ผ่าน และ Gate 3 ผ่านหรือ "N ไม่พอ"

## 14. Future Work (ไม่ทำตอนนี้)
Deep learning (LSTM/Transformer บน RTX 3050) · RL (PPO) · multi-symbol · full requote/partial-fill sim · real-money (หลังพิสูจน์ครบ)
