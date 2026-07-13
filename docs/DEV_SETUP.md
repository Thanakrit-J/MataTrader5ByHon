# Dev Machine Setup Guide — MT5 Gold AI Trading

คู่มือติดตั้งสภาพแวดล้อมสำหรับ **เครื่อง dev จริง** (Windows) ที่จะ implement + รันบอท
สเปคอ้างอิงผู้ใช้: Ryzen 7 4800H / RTX 3050 / RAM 24GB — เพียงพอมากสำหรับ LightGBM บน CPU

> ⚠️ **สำคัญ:** ไลบรารี `MetaTrader5` เป็น **Windows-only** และต้องรันบน "เครื่องเดียวกับที่เปิดโปรแกรม MT5 terminal" เพราะ Python คุยกับ terminal ผ่าน IPC ในเครื่อง — รันบน Linux/Mac หรือคนละเครื่องกับ terminal ไม่ได้

> 📌 **สถานะโค้ดตอนนี้ (branch `MillionDollars`):** repo มีแค่ `trading_bot.py` (บอทเทรดสด) + `trade_history.py` (อ่าน log) — **รันได้เลยหลัง setup เสร็จ**
> ส่วน backtest / ML pipeline / unit test / `scripts/fetch_data.py` เป็น **deliverable ของ Phase 0-4** (ดูตาราง §7) ที่ยังไม่ได้ลงมือ — คู่มือนี้จะระบุชัดว่าขั้นไหน "ใช้ได้ตอนนี้" vs "มาทีหลัง"

---

## 0. Prerequisites (สิ่งที่ต้องมีก่อน)
- Windows 10/11
- โปรแกรม **MetaTrader 5 terminal** + **บัญชี Demo** → วิธีติดตั้งดู §1 (ห้ามใช้ live — บอทมี safe-lock)
- **Python 3.11** (python.org — ไม่ใช่ Microsoft Store) → §2
- **Git**
- **Docker Desktop** (สำหรับ PostgreSQL ที่ใช้ log เทรดสด — มี `docker-compose.yml` อยู่แล้ว)

---

## 1. ติดตั้ง MetaTrader 5 terminal

### ทางเลือกที่ 1 (แนะนำ): โหลดจากโบรกเกอร์ของคุณ
installer ของโบรกจะ **ผูกกับ trade server (demo/live) ของโบรกนั้นให้อัตโนมัติ** เปิดบัญชี demo ได้ง่ายกว่ามาก
1. เข้าเว็บโบรกที่จะใช้ (เช่น Exness / IC Markets / XM / Pepperstone ฯลฯ) → หน้า **Download MT5**
2. โหลด installer `.exe` → ดับเบิลคลิก → Next → ยอมรับ license → Finish
3. เปิดโปรแกรม → เซิร์ฟเวอร์ของโบรกจะถูกเลือกให้อยู่แล้ว

### ทางเลือกที่ 2: โหลดจาก MetaQuotes (ตัวกลาง)
- ลิงก์: https://www.metatrader5.com/en/download
- ข้อเสีย: ตอน login demo ต้องพิมพ์ชื่อ **trade server** ของโบรกเอง (ถ้าไม่รู้ชื่อ server จะหาบัญชียาก) → แนะนำทางเลือกที่ 1 มากกว่า

ตั้งค่า terminal ต่อหลังติดตั้งเสร็จ → ดู §5

---

## 2. ติดตั้ง Python 3.11
ดาวน์โหลดจาก https://www.python.org/downloads/release/python-3119/ (Windows installer 64-bit)

ตอนติดตั้ง **ติ๊ก "Add python.exe to PATH"** ให้เรียบร้อย

> หมายเหตุ: อย่าใช้ตัว `python.exe` จาก Microsoft Store (เป็น stub) — ต้องเป็นตัวจาก python.org

ตรวจสอบ (เปิด PowerShell ใหม่):
```powershell
python --version      # ต้องได้ Python 3.11.x
```
ถ้าขึ้นหน้าต่าง Microsoft Store ให้ปิด App execution alias:
`Settings > Apps > Advanced app settings > App execution aliases` → ปิด python.exe / python3.exe

---

## 3. Clone + สร้าง virtual environment
```powershell
git clone <repo-url> MataTrader5ByHon
cd MataTrader5ByHon
git checkout MillionDollars

python -m venv venv
.\venv\Scripts\Activate.ps1        # ถ้าติด ExecutionPolicy: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
```
`venv/` ถูก gitignore ไว้แล้ว

---

## 4. ติดตั้ง dependencies

### 4.1 deps สำหรับรันบอทสด (ใช้ได้ตอนนี้)
repo มีไฟล์ `requirements.txt` มาให้แล้ว — ติดตั้งตรงๆ ได้เลย:
```powershell
pip install -r requirements.txt
```
> 💡 `requirements.txt` ถูก save เป็น **UTF-16** (pip อ่านได้ปกติ) — แต่ถ้าจะแก้ไฟล์นี้เอง แนะนำ save ใหม่เป็น **UTF-8** เพื่อกัน tool อื่นอ่านเพี้ยน

ตรวจสอบว่าครบ:
```powershell
python -c "import MetaTrader5, numpy, pandas, psycopg2, pytz, ta; print('base deps OK')"
```

### 4.2 deps เพิ่มสำหรับ Phase 0-4 (backtest / ML / test) — ลงตอนถึงเฟสนั้น
Phase 0-4 ต้องใช้ไลบรารีเพิ่ม (pyarrow, scikit-learn, lightgbm, pytest) ที่ **ยังไม่อยู่ใน** `requirements.txt`
สร้างไฟล์ `requirements-dev.txt` (อ้าง base เดิมเพื่อไม่ให้เวอร์ชันชนกัน):
```
-r requirements.txt
pyarrow==16.1.0
scikit-learn==1.5.0
lightgbm==4.3.0
pytest==8.2.0
```
ติดตั้งเมื่อพร้อมเริ่ม Phase 0:
```powershell
pip install -r requirements-dev.txt
python -c "import pyarrow, lightgbm, sklearn, pytest; print('dev deps OK')"
```

---

## 5. ตั้งค่า MetaTrader 5 terminal
1. เปิด MT5 → **File > Open an Account** → เลือกโบรก → เปิดบัญชี **Demo** (ห้าม live — บอทมี safe-lock ที่บังคับ `ACCOUNT_TRADE_MODE_DEMO`)
2. `Tools > Options > Expert Advisors` → ติ๊ก **"Allow algorithmic trading"**
3. เปิด **Market Watch** แล้วคลิกขวา → Show All → หาให้เจอ **`XAUUSD`**
   - ชื่อ symbol อาจมี suffix แล้วแต่โบรก เช่น `XAUUSD.`, `GOLD` — ถ้าไม่ตรงต้องแก้ค่า `SYMBOL` ใน [`trading_bot.py`](../trading_bot.py) (บรรทัด 12) ให้ตรง
   - _(Phase 0 จะย้ายค่านี้ไปเป็น `DataConfig.symbol`)_
4. เปิดกราฟ XAUUSD timeframe M5 ทิ้งไว้สักครู่ให้ terminal ดึงประวัติมาก (History depth ตั้งได้ที่ `Tools > Options > Charts > Max bars in chart` → ตั้งสูงๆ เช่น unlimited)
   - _หมายเหตุ: บอทสดใช้ **M5** (`TIMEFRAME_M5`); Phase 0 จะดึง M1 มาทำ dataset_
5. เปิด terminal ค้างไว้ตอนรัน script Python ทุกครั้ง

---

## 6. ตั้งค่า PostgreSQL (log เทรดสด)
```powershell
docker-compose up -d          # ใช้ไฟล์เดิมในโปรเจกต์ (port 5433)
```
`docker-compose.yml` ตั้ง user/pass/db ให้ตรงกับที่ `trading_bot.py` ใช้อยู่แล้ว:
`bot_user` / `BotPassword123` / `mt5_trading` — **รันได้เลยไม่ต้องตั้งอะไรเพิ่ม**

> ⚠️ **หนี้ทางเทคนิคที่ต้องรู้:** ตอนนี้รหัสผ่าน DB ถูก **hardcode** อยู่ทั้งใน `DB_CONFIG` ของ [`trading_bot.py`](../trading_bot.py) (บรรทัด 33-39) และใน `docker-compose.yml`
> การย้ายไปอ่านจาก environment variable (`require_env`) เป็น **เป้าหมายของ Phase 2 (live-refactor)** — ยังไม่ใช่ของตอนนี้
> ถ้าจะเตรียม env var ไว้ล่วงหน้า (ยังไม่มีผลกับโค้ดปัจจุบัน):
> ```powershell
> setx MT5GOLD_DB_HOST "localhost"
> setx MT5GOLD_DB_PORT "5433"
> setx MT5GOLD_DB_NAME "mt5_trading"
> setx MT5GOLD_DB_USER "bot_user"
> setx MT5GOLD_DB_PASSWORD "BotPassword123"
> # ปิด/เปิด PowerShell ใหม่ให้ค่ามีผล
> ```

---

## 7. ตรวจสอบว่าพร้อม (smoke test)

### 7.1 เช็คที่ทำได้ "ตอนนี้เลย"
```powershell
# (a) DB ต่อได้ไหม (ต้อง docker-compose up -d ก่อน)
python -c "import psycopg2; c=psycopg2.connect(host='localhost',port='5433',dbname='mt5_trading',user='bot_user',password='BotPassword123'); print('DB OK'); c.close()"

# (b) ต่อ MT5 terminal ได้ไหม + เป็นบัญชี demo จริงไหม (ต้องเปิด terminal + login demo ก่อน)
python -c "import MetaTrader5 as mt5; ok=mt5.initialize(); a=mt5.account_info(); print('init',ok,'| login',a.login if a else None,'| demo?', (a.trade_mode==mt5.ACCOUNT_TRADE_MODE_DEMO) if a else None); mt5.shutdown()"

# (c) รันบอทสดจริง (จะเริ่มวนลูปเทรดบน demo — Ctrl+C เพื่อหยุด)
python trading_bot.py
```

### 7.2 เช็คที่จะใช้ได้ "หลังทำเฟสนั้นเสร็จ" (ยังรันไม่ได้ตอนนี้)
```powershell
# หลัง Phase 0/1 เพิ่ม unit test + FakeBroker แล้ว:
python -m pytest -v

# หลัง Phase 0 สร้าง scripts/fetch_data.py แล้ว:
python scripts/fetch_data.py --symbol XAUUSD --start 2020-01-01
# ควรได้ไฟล์ data/clean/XAUUSD/M1/data.parquet + manifest.json
```

---

## 8. ลำดับการทำงาน (execution order)
รันทีละเฟสตาม plan ใน `docs/superpowers/plans/` โดยใช้ skill `subagent-driven-development` หรือทำมือ:

| เฟส | plan | ผลลัพธ์ที่พิสูจน์ได้ |
|---|---|---|
| 0 | `2026-07-04-phase0-data-foundation.md` | dataset reproducible |
| 1 | `2026-07-04-phase1-backtest-and-baseline.md` | **baseline B0/B1 จริง** (มี edge ไหม?) |
| 2 | `2026-07-04-phase2-live-refactor-parity.md` | demo ตรงกับ backtest |
| 3 | `2026-07-04-phase3-ml-pipeline.md` | ML ชนะ B1 (out-of-sample) |
| 4 | `2026-07-04-phase4-integration-demo.md` | demo forward-test พิสูจน์ "พลาดน้อยลง" |

> **Go/No-Go:** ถ้า Phase 1 พบว่า B1 (กลยุทธ์เดิมที่แก้บั๊กแล้ว) ยัง expectancy ≤ 0 หลังหักต้นทุน → **STOP** ตาม spec §11 (อย่าเสียบ ML ทับกลยุทธ์ที่ไม่มี edge)

---

## 9. Troubleshooting
- `mt5.initialize()` คืน False → เปิด terminal ค้างไว้หรือยัง? login demo แล้วหรือยัง? รัน Python เป็น user เดียวกับที่เปิด terminal
- บอทขึ้น `Live account detected` แล้วปิดตัว → นี่คือ safe-lock ทำงานถูกต้อง; ต้อง login **บัญชี demo** เท่านั้น
- ดึงข้อมูลได้น้อย → เพิ่ม "Max bars in chart" + scroll กราฟย้อนหลังให้ terminal โหลดประวัติ
- `XAUUSD` ไม่เจอ → เช็คชื่อ symbol จริงใน Market Watch แล้วแก้ค่า `SYMBOL` ใน `trading_bot.py`
- `pip install -r requirements.txt` เพี้ยน/อ่านไม่ออก → ไฟล์เป็น UTF-16; อัปเดต pip ให้ล่าสุด หรือ save ไฟล์ใหม่เป็น UTF-8
- DB ต่อไม่ได้ → `docker-compose up -d` แล้วหรือยัง? port 5433 ชนกับอะไรอยู่ไหม (`docker ps`)
- lightgbm/pyarrow ลงไม่ได้ → อัปเดต pip + ติดตั้ง Microsoft Visual C++ Redistributable
