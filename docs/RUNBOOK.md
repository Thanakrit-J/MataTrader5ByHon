# Runbook — วิธีรันโปรเจ็ค MT5 Gold

คู่มือ **ใช้งานจริง** ของ pipeline ที่ implement แล้ว (Phase 0-2)
สำหรับการติดตั้ง environment ครั้งแรก (Python 3.11, MT5 terminal, Docker) ดู [`DEV_SETUP.md`](DEV_SETUP.md) ก่อน
สถานะ thesis ปัจจุบัน: **ปิดแล้ว (Go/No-Go = STOP)** — ดู [`DECISION-phase1-go-no-go.md`](DECISION-phase1-go-no-go.md)

> ทุกคำสั่งรันจาก **root ของ repo** (`C:\work\MataTrader5ByHon`)
> ตัวอย่างใช้ `.\venv\Scripts\python.exe` (ทำงานไม่ว่าจะ activate venv หรือไม่)
> ถ้า activate venv แล้ว (เห็น `(venv)` หน้าบรรทัด) จะพิมพ์แค่ `python` ก็ได้

---

## ต้องเปิด MT5 terminal เมื่อไหร่?

| งาน | ต้องเปิด MT5 + login demo? |
|---|---|
| รัน tests | ❌ ไม่ต้อง |
| Phase 0 — fetch data | ✅ **ต้อง** (ดึงจาก terminal ผ่าน IPC) |
| Phase 1 — backtest / sweep | ❌ ไม่ต้อง (อ่านจาก Parquet) |
| Phase 2 — live bot | ✅ **ต้อง** (วางออเดอร์ demo) |

---

## 0. เตรียม venv (ถ้าเครื่องใหม่ / ยังไม่มี)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # ถ้าติด ExecutionPolicy: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements-dev.txt
```
`requirements-dev.txt` รวม base deps + `pyarrow`, `pytest` (และ `scikit-learn`/`lightgbm` เผื่ออนาคต)

---

## 1. รัน tests (ตรวจว่าโค้ดทั้งหมดเขียว — ไม่ต้องมี MT5)

```powershell
.\venv\Scripts\python.exe -m pytest -q
```
คาดหวัง: `63 passed`

---

## 2. Phase 0 — สร้าง dataset (ต้องเปิด MT5 demo)

**2.1 เช็คก่อนว่าต่อ MT5 ได้ + เป็น demo:**
```powershell
.\venv\Scripts\python.exe -c "import MetaTrader5 as mt5; ok=mt5.initialize(); a=mt5.account_info(); print('init',ok,'| demo?',(a.trade_mode==mt5.ACCOUNT_TRADE_MODE_DEMO) if a else None); mt5.shutdown()"
```
ต้องได้ `init True | demo? True`

**2.2 ดึงข้อมูล** — ⚠️ **แนะนำ `--timeframe M15`** ไม่ใช่ M1 เพราะโบรก demo นี้มี M1 แค่ ~3.5 เดือน แต่มี M15 ย้อนถึง 2022 (~4.25 ปี):
```powershell
.\venv\Scripts\python.exe scripts\fetch_data.py --symbol XAUUSD --timeframe M15 --start 2022-01-01
```
ผลลัพธ์: `data\clean\XAUUSD\M15\data.parquet` + `manifest.json`

**2.3 ตรวจผล:**
```powershell
Get-Content data\clean\XAUUSD\M15\manifest.json
```
ดูว่า `rows` เยอะพอ, `contract.contract_size` และ `contract.point` ไม่เป็น null

**flags:** `--root` (default `data`) · `--symbol` · `--timeframe` (M1/M5/M15/H1) · `--start` (ISO date) · `--end`

> **ชื่อ symbol:** ถ้าโบรกใช้ suffix (เช่น `XAUUSD.`, `GOLD`) ต้องใส่ `--symbol` ให้ตรง (ดูใน Market Watch) และต้อง Show ไว้ใน Market Watch ด้วย

---

## 3. Phase 1 — Backtest + Go/No-Go (ไม่ต้องมี MT5)

**3.1 รัน baseline B0/B1 + verdict:**
```powershell
.\venv\Scripts\python.exe scripts\run_backtest.py --timeframe M15
```
- อ่าน dataset ที่ timeframe เดียวกันตรงๆ (ไม่ resample)
- ถ้าเก็บเป็น M1 แล้วอยาก backtest M15: เพิ่ม `--base-timeframe M1` (จะ resample M1→M15 ให้)

**อ่าน output ยังไง:**
```
B1: n_trades=7199 expectancy=-0.035 CI=[-0.325,0.259] PF=0.992 PF_CI=[0.927,1.060] maxDD=-1988
GO/NO-GO: STOP - ...
```
- **PROCEED** ก็ต่อเมื่อ **ขอบล่างของ expectancy CI > 0 และ PF CI ≥ 1.0** (ไม่ใช่แค่ค่ากลาง — กัน noise)
- artifact ถูก freeze ที่ `artifacts\baseline_B0.json`, `baseline_B1.json`

**3.2 ประเมินกลยุทธ์หลาย config (variant sweep):**
```powershell
.\venv\Scripts\python.exe scripts\sweep_baseline.py --base-timeframe M15
```
- ลอง grid (timeframe × R:R × RSI) แบบมีวินัย, แบ่ง in-sample/out-of-sample, deflated Sharpe คุม multiple-testing
- พิมพ์ตารางว่าตัวไหนผ่าน gate ทั้ง in-sample และ holdout
- **flags:** `--base-timeframe` · `--train-frac` (default 0.7)

---

## 4. Phase 2 — Live bot บน demo (ต้องเปิด MT5 demo)

> ⚠️ จะ **วางออเดอร์จริงบนบัญชี demo** เป็น loop — มี safe-lock (ถ้าเป็น live จะปิดตัวทันที)

**4.1 รัน** (สังเกต: ต้องมี `python` นำหน้าและ path `scripts\` — พิมพ์ `run_live.py` เฉยๆ PowerShell ไม่รู้จัก):
```powershell
.\venv\Scripts\python.exe scripts\run_live.py --symbol XAUUSD --timeframe M5
```
เห็น `Live loop on XAUUSD M5 (DEMO). Ctrl+C to stop.` = ทำงานแล้ว
**flags:** `--symbol` · `--timeframe` · `--risk-pct` (default 0.01) · `--journal` (default `journal/live.jsonl`) · `--poll-seconds` (default 5)

**4.2 ดู decision journal** (เปิด PowerShell **หน้าต่างที่ 2**):
```powershell
cd C:\work\MataTrader5ByHon
Get-Content journal\live.jsonl -Wait -Tail 10
```
ทุกแท่งปิดจะมี 1 record (BUY/SELL/FLAT) — `-Wait` = tail สด

**4.3 หยุด:** กด `Ctrl+C` ในหน้าต่างที่ loop รันอยู่

---

## 5. Cheat sheet

```powershell
# tests
.\venv\Scripts\python.exe -m pytest -q

# Phase 0: fetch (ต้องเปิด MT5)
.\venv\Scripts\python.exe scripts\fetch_data.py --symbol XAUUSD --timeframe M15 --start 2022-01-01

# Phase 1: backtest + go/no-go
.\venv\Scripts\python.exe scripts\run_backtest.py --timeframe M15

# Phase 1: variant sweep
.\venv\Scripts\python.exe scripts\sweep_baseline.py --base-timeframe M15

# Phase 2: live demo (ต้องเปิด MT5)
.\venv\Scripts\python.exe scripts\run_live.py --symbol XAUUSD --timeframe M5
```

---

## 6. Troubleshooting (จากที่เจอจริง)

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `'run_live.py' is not recognized` | ต้องมี `python` นำหน้า + path: `.\venv\Scripts\python.exe scripts\run_live.py ...` |
| `init False` / ต่อ MT5 ไม่ได้ | เปิด terminal ค้าง + login demo + รัน Python เป็น user เดียวกับที่เปิด terminal |
| `LIVE account detected` แล้วปิดตัว | safe-lock ทำงานถูก — ต้อง login **demo** เท่านั้น |
| fetch ได้ข้อมูลน้อย/สั้น | โบรก demo นี้ M1 ตื้น (~3.5 เดือน) — ใช้ `--timeframe M15` (ลึกถึง 2022); เพิ่ม Max bars in chart + scroll กราฟย้อนหลัง |
| `symbol_info returned None` | ชื่อ symbol ไม่ตรง/ไม่ได้ Show ใน Market Watch — เช็คชื่อจริงแล้วใส่ `--symbol` |
| `copy_rates_range` คืน `Invalid params` ตอนช่วงยาว | เป็นเพดานจำนวนบาร์ต่อ call — pipeline chunk 30 วันให้แล้ว (ปัญหานี้เกิดเฉพาะตอน query ตรงๆ ช่วงยาวมาก) |
| ผลลัพธ์ (contract_size/spread) เพี้ยน | pipeline map `trade_contract_size` + impute spread==0 ให้แล้ว (Phase 0 fixes) |

---

## หมายเหตุ: ข้อจำกัดที่ยังค้าง
- `tz_offset` ตรวจไม่ได้จาก MT5 จริง → timestamp เป็นเวลาเซิร์ฟเวอร์โบรก ไม่ใช่ UTC จริง
- `CostConfig.commission_per_lot = 0` → ต้นทุนมองโลกในแง่ดี (ใส่ค่าคอมจริงถ้าจะประเมินกลยุทธ์ใหม่)
- ดู tech debt เต็มๆ ใน [`DECISION-phase1-go-no-go.md`](DECISION-phase1-go-no-go.md)
