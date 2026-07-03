# MT5 Gold AI Trading — Design Spec (แนวทาง B)

- **วันที่:** 2026-07-03
- **สถานะ:** Draft (รอ user review)
- **Symbol:** XAUUSD (ทองคำ) บน MetaTrader 5
- **เป้าหมายระดับสูง:** พิสูจน์ด้วยตัวเลขว่า "พลาดน้อยลง" บน backtest + demo ก่อนตัดสินใจเรื่องเงินจริง
- **ผู้ใช้:** เขียน Python ได้ / ML ยังใหม่ → เอกสารนี้อธิบายแนวคิด ML ที่จำเป็นด้วย
- **เครื่อง:** Ryzen 7 4800H / RTX 3050 / RAM 24GB → โมเดลหลักเป็น tabular (LightGBM) บน CPU

---

## 1. เป้าหมาย, ขอบเขต, และสิ่งที่ไม่ทำ

### 1.1 เป้าหมาย (Goals)
1. สร้าง **รากฐานที่ซื่อสัตย์**: data pipeline ที่สะอาด + backtest engine ที่จำลองต้นทุนจริง เพื่อไม่หลอกตัวเอง
2. **ซ่อมบั๊กวิกฤต**ของบอทเดิมที่เป็นต้นเหตุการขาดทุน
3. สร้าง **ML pipeline** ที่ใช้ AI เป็น "ตัวคัดกรองสัญญาณ" ให้เทรดแม่นขึ้น/พลาดน้อยลง
4. ทุกการอ้างว่า "ดีขึ้น" ต้อง **พิสูจน์ด้วย out-of-sample metric** ที่กัน overfitting

### 1.2 สิ่งที่ไม่ทำในเฟสนี้ (Non-Goals)
- ไม่เทรดเงินจริง (demo เท่านั้น จนกว่าจะพิสูจน์ผ่าน)
- ไม่ทำ Reinforcement Learning / LSTM / Transformer ในเฟสแรก (เก็บเป็น future work)
- ไม่ทำ multi-symbol / portfolio (โฟกัส XAUUSD ตัวเดียว)
- ไม่ทำ production trading infra (colocation, HA, ฯลฯ)

### 1.3 เกณฑ์วัดความสำเร็จ (Success Criteria — เป็นตัวเลข)
วัดบน **out-of-sample (walk-forward)** และหลังหักต้นทุนจริงเท่านั้น:
- **Expectancy ต่อไม้ > 0** (หลังหัก spread/commission/slippage)
- **Profit Factor ≥ 1.2**
- **Max Drawdown** อยู่ในกรอบที่กำหนด (เช่น ≤ 15% ของ equity)
- **ML strategy ชนะ rule-based baseline อย่างเสถียรในทุกหน้าต่าง walk-forward** (ไม่ใช่ชนะแค่บางช่วง)
- **ผล demo forward-test เกาะกับ backtest** ในกรอบที่ยอมรับได้ (หลักฐานว่าไม่ overfit และไม่มี train/live skew)

---

## 2. หลักการออกแบบ (Design Principles)

1. **Shared Core = "สิ่งที่เทสต์ = สิ่งที่เทรดจริง"**: โค้ด feature engineering, strategy, และ cost model เป็นชุดเดียว ที่ทั้ง backtest, ML research, และ live bot เรียกใช้ → ตัดปัญหา train/live skew ที่ทำให้ระบบดูดีตอนเทสต์แต่พังตอนรันจริง
2. **Causality เป็นกฎเหล็ก**: feature ของแท่ง t ใช้ข้อมูลถึง close ของ t เท่านั้น; เข้าออเดอร์ที่ open ของ t+1 — บังคับโดย backtest engine
3. **FLAT คือคำตอบที่ถูกต้อง**: ไม่บังคับเข้าตลาดเมื่อไม่มี edge (ตัด logic "เดาแล้วเข้าทุกแท่ง" ทิ้ง)
4. **แต่ละไฟล์มีหน้าที่เดียว**: เล็ก โฟกัส เทสต์ได้อิสระ
5. **Reproducibility**: dataset และ model artifact มี metadata ย้อนกลับได้ว่ามาจากข้อมูล/พารามิเตอร์ชุดไหน
6. **วัดผลเป็นเงิน ไม่ใช่ % ความแม่นของ classifier**: ML ที่แม่น 55% แต่ทำเงินได้ ดีกว่าแม่น 70% แต่ขาดทุน

---

## 3. สถาปัตยกรรม & โครงสร้างโปรเจกต์

```
mt5gold/
├── data/
│   ├── fetch.py       # ดึงประวัติจาก MT5 (copy_rates_range)
│   ├── clean.py       # gap, tz, resample, validate
│   └── store.py       # อ่าน/เขียน parquet + metadata
├── core/              # ◆ SHARED — ใช้ร่วมทั้ง backtest/ml/live
│   ├── types.py       # Bar, Signal, Trade, Position (dataclasses)
│   ├── features.py    # feature engineering (causal)
│   ├── strategy.py    # Strategy Protocol + RuleBasedStrategy + MLStrategy
│   └── costs.py       # spread/commission/slippage + position sizing
├── backtest/
│   ├── engine.py      # event-driven loop
│   └── metrics.py     # WR, PF, Expectancy, MaxDD, Sharpe/Sortino, equity curve
├── ml/
│   ├── labeling.py    # triple-barrier labeling
│   ├── dataset.py     # ประกอบ X, y จาก features + labels
│   ├── train.py       # LightGBM training
│   ├── validate.py    # walk-forward + purge + embargo
│   └── model.py       # โหลด/ทำนาย wrapper
├── live/
│   ├── bot.py         # loop สด (refactor จาก trading_bot.py)
│   ├── execution.py   # จัดการออเดอร์ MT5
│   └── safety.py      # demo lock, spread guard, daily/DD limits
└── config.py          # config รวม (dataclass) + secret ผ่าน env

scripts/               # entrypoints: fetch_data.py, run_backtest.py, train_model.py, run_live.py
tests/                 # unit + integration tests
docs/
  ├── superpowers/specs/   # spec นี้
  └── diagrams/            # ไดอะแกรม (ไม่ commit)
```

`trading_bot.py` เดิม: เก็บไว้จนกว่าจะ migrate ครบ แล้วค่อย deprecate

**Tech stack:** Python 3.11 · `MetaTrader5` · `pandas` · `numpy` · `pyarrow` · `lightgbm` · `scikit-learn` · `pytest`

---

## 4. Data Pipeline (`mt5gold/data/`)

### 4.1 `fetch.py`
- ดึง **M1 เป็น base timeframe** ผ่าน `mt5.copy_rates_range(symbol, TIMEFRAME_M1, start, end)` แบ่งดึงเป็นช่วง (chunk) เพื่อเลี่ยงข้อจำกัดจำนวนแท่งต่อครั้ง
- ดึงย้อนหลังให้มากที่สุดที่โบรกเกอร์มี (คาดหวัง 2–5 ปีของ M1)
- คืน DataFrame ดิบพร้อม `time, open, high, low, close, tick_volume, spread, real_volume`

### 4.2 `clean.py` — กับดักเฉพาะทองคำ (จุดที่ทำให้ผลลวงถ้าพลาด)
1. **Broker time → UTC**: เวลา MT5 มักเป็น UTC+2/+3 (ขึ้นกับ DST) — ตรวจ offset จริงจาก server แล้วแปลงเป็น UTC เก็บ offset ไว้ใน metadata
2. **Weekend/holiday gap**: ทองหยุด ศ. ~22:00 UTC ถึง อา. ~22:00 UTC — resample ต้อง **ไม่สร้างแท่งผี**ช่วงตลาดปิด (ใช้ session calendar ไม่ใช่ reindex ต่อเนื่อง)
3. **`tick_volume` ≠ volume จริง**: เป็นจำนวน tick ต้องระบุชัดใน feature ว่าเป็น proxy
4. **Validation (fail ถ้าไม่ผ่าน):** timestamp เรียงขึ้นและไม่ซ้ำ · ระยะห่างแท่งสม่ำเสมอตาม session · `high ≥ max(open,close)` และ `low ≤ min(open,close)` · ราคาไม่กระโดดเกิน threshold (จับ bad tick)

### 4.3 `store.py`
- **Parquet** แยก partition ตาม `symbol/timeframe/`
- แยก `raw/` (ดิบ ไม่แก้) กับ `clean/` (ผ่าน validation)
- เก็บ **manifest** (JSON): เวลาดึง, ช่วงข้อมูล, โบรกเกอร์, server, tz offset, จำนวนแท่ง, hash → reproducibility
- resample: M1 → M5/M15/H1 ด้วยกฎ OHLCV มาตรฐาน (`open`=first, `high`=max, `low`=min, `close`=last, volume=sum) เคารพ session boundary
- PostgreSQL เดิม: ใช้เฉพาะ log เทรดสด (แยกจาก data วิจัย)

---

## 5. Shared Core (`mt5gold/core/`)

### 5.1 `types.py`
```python
@dataclass(frozen=True)
class Signal:
    side: Literal["BUY", "SELL", "FLAT"]
    confidence: float          # 0..1 (rule-based = 1.0 เมื่อเข้าเงื่อนไข; ML = prob)
    sl_price: float | None
    tp_price: float | None
    meta: dict                 # เหตุผล/ค่าตัวชี้วัด ณ จุดตัดสิน
```
`Bar`, `Trade`, `Position` เป็น dataclass เช่นกัน

### 5.2 `features.py` — คำนวณแบบ causal
- **Technical:** EMA(9,21,50), RSI(14), ATR(14), MACD, Bollinger + slope/ระยะห่างเส้น
- **Price action:** สัดส่วน body/wick, ระยะถึง swing high/low N แท่ง, ระยะถึง demand/supply zone
- **Volatility regime:** ATR percentile (rolling)
- **Time/session:** ชั่วโมง (UTC), one-hot session Asian/London/NY, day-of-week
- **Multi-timeframe:** เทรนด์ H1 (EMA fast/slow) มาเป็น context ของ M15
- ฟังก์ชันหลัก: `build_features(df_base, df_htf) -> pd.DataFrame` (index = time, ทุกคอลัมน์ causal)

### 5.3 `strategy.py`
```python
class Strategy(Protocol):
    def generate_signal(self, features_row, state) -> Signal: ...

class RuleBasedStrategy:   # zone + EMA/RSI สะอาด, คืน FLAT ได้
class MLStrategy:          # ห่อ model.predict, เข้าเมื่อ prob > threshold
```

### 5.4 `costs.py`
- `apply_costs(fill_price, side, spread, commission_per_lot, slippage) -> effective_price`
- `position_size(balance, risk_pct, sl_distance_price, symbol_spec) -> lot`
  สูตรถูกต้อง: `lot = (balance*risk_pct) / (sl_distance_price * value_per_price_unit_per_lot)` แล้ว clamp ด้วย volume_min/max/step + margin check
- อ่าน contract spec จริงจาก `mt5.symbol_info` (ไม่ hardcode)

---

## 6. Backtest Engine (`mt5gold/backtest/`) — หัวใจ

### 6.1 `engine.py` — event-driven
- วน loop ทีละแท่งของ base timeframe
- **ตัดสินใจบนแท่งปิด → เข้าออเดอร์ที่ open ของแท่งถัดไป** (กัน lookahead)
- จำลอง lifecycle ออเดอร์: entry (หัก cost) → ถือ → SL/TP/trailing/time-exit → บันทึก `Trade`
- **Intrabar SL/TP resolution:** เมื่อทั้ง SL และ TP อยู่ในช่วง high–low ของแท่งเดียวกัน:
  - ถ้ามี **M1** ในแท่งนั้น → ไล่ path จริงว่าราคาแตะเส้นไหนก่อน
  - ถ้าไม่มี → **สมมติฐานมองร้าย (assume SL โดนก่อน)** เพื่อไม่ให้ผลเกินจริง
- รองรับทั้ง `RuleBasedStrategy` และ `MLStrategy` ผ่าน interface เดียว

### 6.2 `metrics.py`
- จำนวนไม้, Win rate, **Profit Factor**, **Expectancy ต่อไม้**, Avg win/loss, **Max Drawdown**, Sharpe/Sortino, exposure, **ต้นทุนรวมที่จ่าย**, equity curve (สำหรับ plot)
- แยกผลตาม session/ชั่วโมง/ทิศทาง เพื่อวิเคราะห์จุดอ่อน

### 6.3 กฎกัน leakage (บังคับในโค้ด)
- Feature computation และ backtest ใช้ timestamp เดียวกัน, ห้าม index อนาคต
- Label (ML) ต้องไม่รั่วเข้า feature
- Walk-forward: model ที่ใช้ทำนายแท่ง test ต้อง train จากข้อมูลก่อนหน้า test window เท่านั้น

---

## 7. Strategy + ซ่อมบอทสด (`mt5gold/live/`)

### 7.1 ตารางบั๊กวิกฤต → วิธีแก้
| บั๊กเดิม (ไฟล์:บรรทัด) | ผลกระทบ | วิธีแก้ |
|---|---|---|
| `history_rsi_value` NameError (`trading_bot.py:654`) | สัญญาณ zone/EMA crash → เทรดไม่ได้ เหลือแต่ "เดา" | context คำนวณครบใน strategy ที่เดียว |
| `ORDER BY ASC LIMIT 20` (`:140`) | ดึงแท่งเก่าสุด ไม่ใช่ล่าสุด | rolling window ในหน่วยความจำ / query ถูก |
| บังคับเข้าทุกแท่ง (README ข้อ 5) | เข้าตลาดไม่มี edge → แพ้สถิติ | FLAT = ปกติ |
| วิเคราะห์แท่งไม่ปิด (`iloc[-1]`) | สัญญาณไม่เสถียร | ตัดสินบนแท่งปิด |
| กำไร cap $1 แต่ loss เต็ม SL (`:547-560`) | expectancy ติดลบเชิงออกแบบ | exit policy สอดคล้อง RR เดียวกับ backtest |
| position sizing ไม่คูณ tick value (`:308`) | 29 lots → "No money" | สูตรถูกต้องใน `costs.py` |
| hardcode DB password (`:33-39`) | ความปลอดภัย | ผ่าน env |
| `mt5.initialize()` ทุก loop (`:518`) | สิ้นเปลือง/เสี่ยง | init ครั้งเดียว + reconnect guard |

### 7.2 `bot.py` (loop บาง)
`ดึงแท่งปิดล่าสุด → build_features (shared) → strategy.generate_signal → safety checks → execution` — เส้นทางเดียวกับ backtest

### 7.3 `safety.py`
demo-lock (shutdown ถ้าเจอ live account) · spread guard · daily loss/profit limit · **max drawdown → pause** · จำกัดจำนวนไม้/วัน · emergency close

### 7.4 Timeframe
ค่าเริ่มต้นแนะนำ **M15/H1** (TP ใหญ่กว่า spread หลายเท่า) แต่เป็น **config** — ให้ backtest ตัดสินว่า timeframe ไหน edge ดีสุด

---

## 8. ML Pipeline (`mt5gold/ml/`) — ส่วน AI

### 8.1 Labeling — Triple-Barrier Method (`labeling.py`)
แต่ละแท่งตั้ง 3 เส้น อิง ATR:
- **Upper barrier (TP)** = entry + k_tp × ATR
- **Lower barrier (SL)** = entry − k_sl × ATR
- **Vertical barrier (เวลา)** = ถือได้สูงสุด N แท่ง

Label = เส้นไหนโดนก่อน: **+1 (TP), −1 (SL), 0 (timeout)** → เป้าหมายที่ AI เรียนตรงกับผลเทรดจริง (ต่างจากการทาย "แท่งหน้าขึ้น/ลง" ที่ไม่สน SL/TP/cost)

### 8.2 Dataset (`dataset.py`)
- `X` = feature row ณ แท่งตัดสินใจ, `y` = triple-barrier label
- จัดการ class imbalance (class weight)
- (ทางเลือก) sample weight ตาม uniqueness/ความคาบเกี่ยวของ label

### 8.3 Model (`train.py`)
- **LightGBM classifier** (เริ่ม binary: TP-before-SL; หรือ 3-class)
- Output = **P(ชน TP ก่อน SL)**
- Feature importance เพื่อ interpretability
- Hyperparameter จูนภายใต้ walk-forward (ไม่จูนบน test)

### 8.4 Validation (`validate.py`) — จุดตายของมือใหม่ ML
- **Walk-forward เท่านั้น ห้าม random shuffle**
- **Purging + Embargo**: เว้นช่องระหว่าง train/test กัน label คาบเกี่ยวรั่ว (เพราะ triple-barrier label กินเวลาหลายแท่ง)
- วัด **out-of-sample** และวัด **เป็นเงินผ่าน backtest engine** (ไม่ใช่แค่ accuracy/AUC)
- เทียบกับ baseline: rule-based + random → ต้องชนะอย่างเสถียรทุกหน้าต่าง

### 8.5 Integration
- `MLStrategy` เข้าเทรดเมื่อ `P > threshold` (เช่น 0.6) — threshold จูนบน validation
- AI = "ตัวคัดกรองให้พลาดน้อยลง" ตรงเป้าหมายผู้ใช้

### 8.6 ความคาดหวังตามจริง
ML บนราคาทอง edge เล็ก ชัยชนะคือ "กรองไม้แย่ออก" ไม่ใช่ทายถูกทุกไม้ ทั้งระบบออกแบบเพื่อ **พิสูจน์ตัวเลข**ก่อนเสี่ยงเงิน

---

## 9. Error Handling & Edge Cases
- **MT5 หลุดการเชื่อมต่อ:** reconnect guard + backoff; live bot ไม่ crash ทั้งระบบ
- **ข้อมูลไม่พอ/ว่าง:** strategy คืน FLAT, backtest ข้ามแท่ง
- **Spread กว้างผิดปกติ (ข่าว):** ไม่เข้าไม้ (spread guard)
- **Broker rejection (No money / requote):** log + ไม่ retry แบบ blind
- **DST/tz เปลี่ยน:** ตรวจ offset ทุกครั้งที่ fetch
- **Model artifact เวอร์ชันไม่ตรง feature:** ตรวจ schema hash ก่อนใช้ ไม่งั้นปฏิเสธ

## 10. Testing Strategy
- **Unit:** features (causality/ค่าถูกต้อง), costs (sizing/หัก cost), labeling (triple-barrier ถูกต้อง), metrics
- **Integration:** backtest บนข้อมูลจำลองที่รู้ผลล่วงหน้า → ตรวจว่าเครื่องนับกำไร/DD ถูก
- **Leakage test:** จงใจใส่ feature ที่รั่วอนาคต → ต้องเห็นผลดีผิดปกติ (ยืนยันว่า test จับได้)
- **Live/backtest parity test:** รัน strategy เดียวกันบนข้อมูลชุดเดียว ผ่าน 2 เส้นทาง → ผลต้องตรง

## 11. เฟส & Milestones (แต่ละเฟส = plan แยก)
| เฟส | ทำ | Milestone |
|---|---|---|
| 0 | scaffold + config + fetch M1→parquet | dataset reproducible |
| 1 | backtest engine + cost + rule-based | **baseline จริงของกลยุทธ์เดิม** |
| 2 | ซ่อม+refactor live bot ใช้ core ร่วม (demo) | demo ตรงกับ backtest |
| 3 | features + triple-barrier + LightGBM + walk-forward | out-of-sample ชนะ baseline เสถียร |
| 4 | เสียบ ML → backtest → forward-test demo | demo เกาะ backtest = พิสูจน์ "พลาดน้อยลง" |

## 12. Risks & Mitigations
- **Overfitting:** walk-forward + purge/embargo + เทียบ baseline + วัด out-of-sample เท่านั้น
- **Backtest เกินจริง:** cost model จริง + intrabar pessimistic + parity test
- **ข้อมูลประวัติจำกัด:** ดึงมากที่สุด, ถ้าไม่พอ ปรับความคาดหวัง/timeframe
- **Broker-specific behavior:** เก็บ spec/tz ใน metadata, ทดสอบ demo จริง
- **Scope creep:** ยึด non-goals, ทำทีละเฟส

## 13. Future Work (ไม่ทำตอนนี้)
- Deep learning (LSTM/Transformer) บน RTX 3050
- Reinforcement learning (PPO)
- Multi-symbol / portfolio
- Real-money deployment (หลังพิสูจน์ผ่านครบ)
