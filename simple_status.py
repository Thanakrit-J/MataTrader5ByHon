"""
Simple Bot Status Checker
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def check_status():
    print("🤖 สถานะบอทเทรดทองคำ")
    print("=" * 30)

    # Connect to MT5
    if not mt5.initialize():
        print("❌ MT5 ไม่เชื่อมต่อ")
        return

    # Account info
    account = mt5.account_info()
    if account:
        print(f"💰 ยอดเงิน: ${account.balance:,.2f}")
        print(f"📈 Equity: ${account.equity:,.2f}")
        profit = account.profit
        color = "🟢" if profit >= 0 else "🔴"
        print(f"{color} กำไร/ขาดทุน: ${profit:,.2f}")
    else:
        print("❌ ไม่มีข้อมูลบัญชี")

    # Open positions
    positions = mt5.positions_get()
    open_positions = [p for p in positions if p.magic == 123456] if positions else []

    print(f"\n📊 ตำแหน่งเปิด: {len(open_positions)}")
    for pos in open_positions:
        profit = pos.profit
        color = "🟢" if profit >= 0 else "🔴"
        print(f"  {pos.symbol} {pos.type} {pos.volume} lots - {color} ${profit:,.2f}")

    # Market data
    print("\n📈 ข้อมูลตลาด XAUUSD:")
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 50)
    if rates is not None:
        df = pd.DataFrame(rates)
        latest = df.iloc[-1]
        print(f"  ราคาปัจจุบัน: ${latest['close']:,.2f}")

        # Simple indicators
        import ta
        df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        print(f"  EMA10: {df['ema10'].iloc[-1]:,.2f}")
        print(f"  EMA20: {df['ema20'].iloc[-1]:,.2f}")
        print(f"  RSI: {df['rsi'].iloc[-1]:.1f}")

        # Signal
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        crossover_up = prev['ema10'] <= prev['ema20'] and curr['ema10'] > curr['ema20']
        crossover_down = prev['ema10'] >= prev['ema20'] and curr['ema10'] < curr['ema20']

        if crossover_up and curr['rsi'] < 40:
            print("  🟢 สัญญาณ: BUY")
        elif crossover_down and curr['rsi'] > 60:
            print("  🔴 สัญญาณ: SELL")
        else:
            print("  ⚪ สัญญาณ: รอ")
    else:
        print("  ❌ ไม่มีข้อมูลตลาด")

    print(f"\n📅 อัปเดต: {datetime.now().strftime('%H:%M:%S')}")
    mt5.shutdown()

if __name__ == "__main__":
    check_status()