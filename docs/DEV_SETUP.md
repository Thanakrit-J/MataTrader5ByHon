# Dev Machine Setup Guide — MT5 Gold AI Trading

คู่มือติดตั้งสภาพแวดล้อมสำหรับ **เครื่อง dev จริง** (Windows) ที่จะ implement + รันบอท
สเปคอ้างอิงผู้ใช้: Ryzen 7 4800H / RTX 3050 / RAM 24GB — เพียงพอมากสำหรับ LightGBM บน CPU

> ⚠️ **สำคัญ:** ไลบรารี `MetaTrader5` เป็น **Windows-only** และต้องรันบน "เครื่องเดียวกับที่เปิดโปรแกรม MT5 terminal" เพราะ Python คุยกับ terminal ผ่าน IPC ในเครื่อง — รันบน Linux/Mac หรือคนละเครื่องกับ terminal ไม่ได้

---

## 0. Prerequisites
- Windows 10/11
- โปรแกรม **MetaTrader 5 terminal** (โหลดจากโบรกเกอร์) + **บัญชี Demo** (ห้ามใช้ live — บอทมี safe-lock)
- **Git**
- **Docker Desktop** (สำหรับ PostgreSQL ที่ใช้ log เทรดสด — มี `docker-compose.yml` อยู่แล้ว)

---

## 1. ติดตั้ง Python 3.11
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

## 2. Clone + สร้าง virtual environment
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

## 3. ติดตั้ง dependencies
สร้างไฟล์ `requirements-dev.txt` (ค่าที่ pin ไว้ — เปลี่ยนเลข patch ได้):
```
MetaTrader5==5.0.5735
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0
scikit-learn==1.5.0
lightgbm==4.3.0
pytz==2024.1
psycopg2-binary==2.9.12
pytest==8.2.0
```
ติดตั้ง:
```powershell
pip install -r requirements-dev.txt
```
ตรวจสอบว่าครบ:
```powershell
python -c "import MetaTrader5, pandas, numpy, pyarrow, lightgbm, sklearn, pytest; print('OK')"
```

---

## 4. ตั้งค่า MetaTrader 5 terminal
1. เปิด MT5 → Login บัญชี **Demo**
2. `Tools > Options > Expert Advisors` → ติ๊ก **"Allow algorithmic trading"**
3. เปิด **Market Watch** แล้วคลิกขวา → Show All → หาให้เจอ **`XAUUSD`** (ชื่อ symbol อาจมี suffix แล้วแต่โบรก เช่น `XAUUSD.`, `GOLD` — ต้องแก้ `DataConfig.symbol` ให้ตรง)
4. เปิดกราฟ XAUUSD timeframe M1 ทิ้งไว้สักครู่ให้ terminal ดึงประวัติมาก (History depth ตั้งได้ที่ `Tools > Options > Charts > Max bars in chart` → ตั้งสูงๆ เช่น unlimited)
5. เปิด terminal ค้างไว้ตอนรัน script Python ทุกครั้ง

---

## 5. ตั้งค่า PostgreSQL (log เทรดสด)
```powershell
docker-compose up -d          # ใช้ไฟล์เดิมในโปรเจกต์ (port 5433)
```

**ห้าม hardcode password** — ตั้งผ่าน environment variable แทน (โค้ดใหม่ใช้ `require_env`):
```powershell
# ตั้งถาวรให้ user ปัจจุบัน
setx MT5GOLD_DB_HOST "localhost"
setx MT5GOLD_DB_PORT "5433"
setx MT5GOLD_DB_NAME "mt5_trading"
setx MT5GOLD_DB_USER "bot_user"
setx MT5GOLD_DB_PASSWORD "BotPassword123"
# ปิด/เปิด PowerShell ใหม่ให้ค่ามีผล
```

---

## 6. ตรวจสอบว่าพร้อม (smoke test)
```powershell
# 6.1 รัน unit test ทั้งหมด (ไม่ต้องพึ่ง MT5 — ใช้ FakeBroker)
python -m pytest -v

# 6.2 ดึงข้อมูลจริงจาก MT5 (ต้องเปิด terminal + login demo ก่อน)
python scripts/fetch_data.py --symbol XAUUSD --start 2020-01-01
# ควรได้ไฟล์ data/clean/XAUUSD/M1/data.parquet + manifest.json
```

---

## 7. ลำดับการทำงาน (execution order)
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

## 8. Troubleshooting
- `mt5.initialize()` คืน False → เปิด terminal ค้างไว้หรือยัง? login demo แล้วหรือยัง? รัน Python เป็น user เดียวกับที่เปิด terminal
- ดึงข้อมูลได้น้อย → เพิ่ม "Max bars in chart" + scroll กราฟย้อนหลังให้ terminal โหลดประวัติ
- `XAUUSD` ไม่เจอ → เช็คชื่อ symbol จริงใน Market Watch แล้วแก้ `--symbol`
- lightgbm/pyarrow ลงไม่ได้ → อัปเดต pip + ติดตั้ง Microsoft Visual C++ Redistributable
