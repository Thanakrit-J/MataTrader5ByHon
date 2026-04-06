"""
Test script to validate bot components without MT5 connection
"""

import pandas as pd
import numpy as np
import ta
import logging

def test_indicators():
    """Test indicator calculations"""
    print("🔄 กำลังสร้างข้อมูลทดสอบ...")
    # Create sample data (using gold-like price levels)
    np.random.seed(42)
    data = {
        'high': np.random.uniform(4680, 4700, 100),
        'low': np.random.uniform(4670, 4690, 100),
        'close': np.random.uniform(4675, 4695, 100),
    }
    df = pd.DataFrame(data)

    print("📊 กำลังคำนวณ EMA (10 และ 20)...")
    # Test EMA
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=20)

    print("📈 กำลังคำนวณ RSI...")
    # Test RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    print("📉 กำลังคำนวณ ATR...")
    # Test ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    print("✅ คำนวณ Indicators เสร็จสิ้น")
    print(f"📊 ขนาดข้อมูล: {df.shape}")
    print(f"📈 RSI ล่าสุด: {df['rsi'].iloc[-1]:.2f}")
    print(f"📉 ATR ล่าสุด: {df['atr'].iloc[-1]:.5f}")

    return df

def test_signal_generation(df):
    """Test signal generation logic"""
    print("🔍 กำลังตรวจสอบสัญญาณการซื้อขาย...")

    # Simple crossover detection
    prev_fast = df['ema_fast'].iloc[-2]
    prev_slow = df['ema_slow'].iloc[-2]
    curr_fast = df['ema_fast'].iloc[-1]
    curr_slow = df['ema_slow'].iloc[-1]

    crossover_up = prev_fast <= prev_slow and curr_fast > curr_slow
    crossover_down = prev_fast >= prev_slow and curr_fast < curr_slow

    rsi = df['rsi'].iloc[-1]

    if crossover_up and rsi < 40:
        signal = 'BUY'
    elif crossover_down and rsi > 60:
        signal = 'SELL'
    else:
        signal = None

    print(f"📊 สัญญาณที่สร้าง: {signal}")
    return signal

if __name__ == "__main__":
    print("🚀 เริ่มทดสอบส่วนประกอบของ Trading Bot...")
    print("=" * 50)

    try:
        print("📈 กำลังทดสอบ Indicators...")
        df = test_indicators()
        print()

        print("🔄 กำลังทดสอบการสร้างสัญญาณ...")
        signal = test_signal_generation(df)
        print()

        print("✅ การทดสอบทั้งหมดผ่านแล้ว!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ การทดสอบล้มเหลว: {e}")
        logging.error(f"Test error: {e}")