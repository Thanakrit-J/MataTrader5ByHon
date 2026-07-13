# Decision Record — Phase 1 Go/No-Go: **STOP** (thesis closed)

**Date:** 2026-07-14
**Decision:** หยุดที่ Phase 2 ยอมรับว่า pure technical-rule (EMA/RSI trend) บน XAUUSD **ไม่มี edge ที่ trade ได้** — **ไม่ดำเนินการ Phase 3 (ML) และ Phase 4**

---

## บริบท
spec §11 กำหนด Go/No-Go gate: ถ้ากลยุทธ์ baseline (B1) มี expectancy ≤ 0 หรือ PF < 1.0 หลังหักต้นทุน → **STOP อย่าสร้าง ML ทับกลยุทธ์ที่ไม่มี edge** เราสร้าง Phase 0-2 มาเพื่อหาคำตอบนี้อย่างซื่อสัตย์

## วิธีวัด (honest / leakage-safe)
- Backtest แบบ event-driven: ตัดสินบนแท่งปิด → เข้าไม้ที่ราคาเปิดแท่งถัดไป, ต้นทุนจริง (spread จาก ask, slippage บน stop, swap รวม triple-Wednesday, gap-through)
- Metrics พร้อม **bootstrap CI + deflated Sharpe**; gate = **ขอบล่าง CI ของ expectancy > 0 และ PF ≥ 1.0** (ไม่ใช่แค่ point estimate)
- Parity harness พิสูจน์ว่า live-path = backtest-path (no train/live skew)

## ผลลัพธ์

### 1) Baseline บน 4.25 ปี M15 (99,988 บาร์, 2022-04 → 2026-07)
| | expectancy | PF | PF CI | maxDD | n_trades |
|---|---|---|---|---|---|
| B1 (rule) | **−0.035** | **0.992** | [0.93, 1.06] | −1,989 | 7,199 |
| B0 (legacy) | −0.062 | 0.986 | [0.92, 1.05] | −2,752 | 9,513 |

### 2) Disciplined variant sweep (`scripts/sweep_baseline.py`)
กริดประกาศล่วงหน้า 12 variants (timeframe M15/H1 × R:R × RSI), แบ่ง in-sample/OOS, deflated Sharpe ใช้ n_trials = 12
- **ทั้ง 12 variants: expectancy ติดลบ in-sample, PF CI ขอบล่าง < 1.0 ทุกตัว → 0 ผ่าน gate**
- ไม่มีอะไรให้ยืนยัน out-of-sample

**ข้อสรุป:** ไม่มี edge — robust ข้าม timeframe/config, บนตัวอย่างใหญ่, และยังเป็น `commission=0` (มองโลกในแง่ดี จริงแย่กว่า)

## สินทรัพย์ที่ยังมีค่า (reusable)
- `mt5gold/` — data pipeline, backtest engine, live loop ที่ **cost-honest, leakage-safe, parity-proven** (63 tests) ใช้กับ instrument/thesis อื่นได้
- Go/No-Go process (CI-aware) — ประเมินกลยุทธ์ใหม่ได้ทันทีผ่าน `sweep_baseline.py`
- Live bot demo-locked + decision journal — พร้อมถ้ามีกลยุทธ์ที่มี edge ในอนาคต

## จะเปิด thesis ใหม่เมื่อไร
ทำได้ถ้าจะลอง **thesis ที่ต่างเชิงโครงสร้าง** (ไม่ใช่จูน trend rule เดิมซึ่งตายแล้ว):
mean-reversion, regime/session filter, breakout, หรือ instrument/timeframe อื่น
→ เขียน `Strategy` class ใหม่ แล้วรันผ่าน `sweep_baseline.py` เดิม ถ้าผ่าน gate ทั้ง in/out-of-sample ค่อยพิจารณา Phase 3

## ข้อจำกัด/หนี้เทคนิคที่ยังค้าง (ถ้ากลับมาทำต่อ)
- `tz_offset` ตรวจไม่ได้จาก MT5 จริง → เก็บเป็นเวลาเซิร์ฟเวอร์โบรก ไม่ใช่ UTC จริง
- `CostConfig.commission_per_lot = 0` → ต้นทุนมองโลกในแง่ดี (ควรใส่ค่าคอมจริงถ้าประเมินใหม่)
- โบรก demo ให้ M1 แค่ ~3.5 เดือน (M15 ~4.25 ปี, D1 ~11.5 ปี)
