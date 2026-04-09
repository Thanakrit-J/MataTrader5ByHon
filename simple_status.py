"""
Simple Bot Status Checker
"""

import MetaTrader5 as mt5
import pandas as pd
import ta
from datetime import datetime

MAGIC  = 20250101   # ← ต้องตรงกับ ai_trading_bot.py
SYMBOL = "XAUUSD"


def check_status():
    print("🤖 สถานะบอทเทรดทองคำ")
    print("=" * 40)

    if not mt5.initialize():
        print("❌ MT5 ไม่เชื่อมต่อ")
        return

    # --- Account ---
    account = mt5.account_info()
    if account:
        print(f"💰 Balance : ${account.balance:,.2f}")
        print(f"📈 Equity  : ${account.equity:,.2f}")
        color = "🟢" if account.profit >= 0 else "🔴"
        print(f"{color} Floating : ${account.profit:,.2f}")
    else:
        print("❌ ไม่มีข้อมูลบัญชี")

    # --- Open positions (กรอง magic ให้ถูก) ---
    positions    = mt5.positions_get()
    bot_positions = [p for p in positions if p.magic == MAGIC] if positions else []

    print(f"\n📊 Open Positions: {len(bot_positions)}")
    for pos in bot_positions:
        direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        color     = "🟢" if pos.profit >= 0 else "🔴"
        print(f"  {pos.symbol} {direction} {pos.volume} lots | {color} ${pos.profit:,.2f} | SL={pos.sl:.2f} TP={pos.tp:.2f}")

    # --- Market data ---
    print(f"\n📈 ข้อมูลตลาด {SYMBOL}:")
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 50)
    if rates is not None:
        df      = pd.DataFrame(rates)
        df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['rsi']   = ta.momentum.rsi(df['close'], window=14)
        df['atr']   = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=14
        )

        curr = df.iloc[-1]
        print(f"  ราคา  : ${curr['close']:,.2f}")
        print(f"  EMA10 : {curr['ema10']:,.2f}")
        print(f"  EMA20 : {curr['ema20']:,.2f}")
        print(f"  RSI   : {curr['rsi']:.1f}")
        print(f"  ATR   : {curr['atr']:.2f}")

        # สัญญาณตรงกับ scoring system ของบอทหลัก
        score = 0
        score += 2 if curr['ema10'] > curr['ema20'] else -2
        score += 1 if curr['rsi'] > 55 else (-1 if curr['rsi'] < 45 else 0)
        rsi_trend = curr['rsi'] - df['rsi'].iloc[-5]
        score += 1 if rsi_trend > 0 else -1

        print(f"\n  🧠 Score: {score:+d} (threshold ±4)")
        if score >= 4:
            print("  🟢 สัญญาณ: BUY")
        elif score <= -4:
            print("  🔴 สัญญาณ: SELL")
        else:
            print("  ⚪ สัญญาณ: รอ")
    else:
        print("  ❌ ไม่มีข้อมูลตลาด")

    # ← ลบ mt5.shutdown() ออก ให้ monitor.py จัดการเอง
    print(f"\n📅 อัปเดต: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    check_status()
    mt5.shutdown()   # shutdown เฉพาะตอนรันไฟล์นี้เดี่ยวๆ