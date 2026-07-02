"""
Simple Bot Status Checker — EURUSD Edition
"""

import MetaTrader5 as mt5
import pandas as pd
import ta
from datetime import datetime

MAGIC  = 20250101   # ← ต้องตรงกับ eurusd_bot.py
SYMBOL = "EURUSD"

SCORE_THRESHOLD = 3  # ✅ แก้: 4 → 3 ให้ตรงกับบอทหลัก


def check_status():
    print("🤖 สถานะบอทเทรด EURUSD")
    print("=" * 40)

    if not mt5.initialize():
        print("❌ MT5 ไม่เชื่อมต่อ")
        return

    # --- Account ---
    account = mt5.account_info()
    if account:
        print(f"💰 Balance    : ${account.balance:,.2f}")
        print(f"📈 Equity     : ${account.equity:,.2f}")
        print(f"🔓 Free Margin: ${account.margin_free:,.2f}")
        color = "🟢" if account.profit >= 0 else "🔴"
        print(f"{color} Floating  : ${account.profit:,.2f}")
    else:
        print("❌ ไม่มีข้อมูลบัญชี")

    # --- Open positions ---
    positions     = mt5.positions_get()
    bot_positions = [p for p in positions if p.magic == MAGIC] if positions else []

    print(f"\n📊 Open Positions: {len(bot_positions)}")
    for pos in bot_positions:
        direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        color     = "🟢" if pos.profit >= 0 else "🔴"
        print(f"  {pos.symbol} {direction} {pos.volume} lots | {color} ${pos.profit:,.2f} | SL={pos.sl:.5f} TP={pos.tp:.5f}")

    # --- Market data ---
    print(f"\n📈 ข้อมูลตลาด {SYMBOL}:")
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 100)
    if rates is not None:
        df          = pd.DataFrame(rates)
        df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['rsi']   = ta.momentum.rsi(df['close'], window=14)
        df['atr']   = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=14
        )

        curr = df.iloc[-1]
        tick = mt5.symbol_info_tick(SYMBOL)
        info = mt5.symbol_info(SYMBOL)

        spread = (tick.ask - tick.bid) / info.point if tick and info else 0

        print(f"  ราคา   : {curr['close']:.5f}")
        print(f"  Spread : {spread:.1f} pts")
        print(f"  EMA10  : {curr['ema10']:.5f}")
        print(f"  EMA20  : {curr['ema20']:.5f}")
        print(f"  RSI    : {curr['rsi']:.1f}")
        print(f"  ATR    : {curr['atr']:.5f}")

        # --- ATR volatility check ---
        atr_mean = df['atr'].rolling(20).mean().iloc[-1]
        atr_ok   = curr['atr'] >= atr_mean * 0.8
        print(f"  ATR mean(20): {atr_mean:.5f} | {'✅ Volatile' if atr_ok else '💤 Low volatility'}")

        # --- Score (ตรงกับ eurusd_bot.py) ---
        score = 0
        score += 2 if curr['ema10'] > curr['ema20'] else -2

        if curr['rsi'] > 55:
            score += 1
        elif curr['rsi'] < 45:
            score -= 1

        rsi_prev  = df['rsi'].iloc[-5]
        score    += 1 if curr['rsi'] > rsi_prev else -1

        high = df['high'].tail(20).max()
        low  = df['low'].tail(20).min()
        rng  = high - low
        pos  = (curr['close'] - low) / rng if rng > 0 else 0.5
        if pos > 0.7:
            score += 1
        elif pos < 0.3:
            score -= 1

        trend_strength = abs(curr['ema10'] - curr['ema20']) / curr['close']
        score += 1 if trend_strength > 0.0003 else -1

        # --- HTF trend ---
        rates_h1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 100)
        if rates_h1 is not None:
            df_h1         = pd.DataFrame(rates_h1)
            df_h1['ema10'] = ta.trend.ema_indicator(df_h1['close'], window=10)
            df_h1['ema20'] = ta.trend.ema_indicator(df_h1['close'], window=20)
            htf = 'up' if df_h1['ema10'].iloc[-1] > df_h1['ema20'].iloc[-1] else 'down'
        else:
            htf = 'unknown'

        print(f"\n  🧠 Score : {score:+d} (threshold ±{SCORE_THRESHOLD})")
        print(f"  📡 HTF   : {htf}")
        print(f"  📏 Spread: {'✅ OK' if spread <= 15 else '⛔ Too wide'} ({spread:.1f}/15 pts)")

        # --- สัญญาณ ---
        if score >= SCORE_THRESHOLD and htf == 'up' and atr_ok and spread <= 15:
            print("  🟢 สัญญาณ: BUY (ผ่านทุก filter)")
        elif score <= -SCORE_THRESHOLD and htf == 'down' and atr_ok and spread <= 15:
            print("  🔴 สัญญาณ: SELL (ผ่านทุก filter)")
        elif abs(score) >= SCORE_THRESHOLD:
            print(f"  🟡 Score ผ่าน แต่ติด filter (HTF={htf}, ATR={'ok' if atr_ok else 'low'}, Spread={'ok' if spread<=15 else 'wide'})")
        else:
            print("  ⚪ สัญญาณ: รอ")

    else:
        print("  ❌ ไม่มีข้อมูลตลาด")

    print(f"\n📅 อัปเดต: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    check_status()
    mt5.shutdown()