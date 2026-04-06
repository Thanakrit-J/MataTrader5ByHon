"""
Trading Bot Monitoring Dashboard

A simple, human-readable dashboard to monitor the trading bot status.
Shows current balance, positions, recent trades, and market conditions.
"""

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json

def connect_mt5():
    """Connect to MT5"""
    if not mt5.initialize():
        print("❌ ไม่สามารถเชื่อมต่อ MT5")
        return False
    return True

def get_account_info():
    """Get account information"""
    account = mt5.account_info()
    if account is None:
        return None

    return {
        'login': account.login,
        'balance': account.balance,
        'equity': account.equity,
        'margin': account.margin,
        'margin_free': account.margin_free,
        'profit': account.profit
    }

def get_open_positions():
    """Get all open positions"""
    positions = mt5.positions_get()
    if positions is None:
        return []

    bot_positions = []
    for pos in positions:
        if pos.magic == 123456:  # Our bot's magic number
            bot_positions.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'sl': pos.sl,
                'tp': pos.tp
            })

    return bot_positions

def get_recent_trades(hours=24):
    """Get recent closed trades"""
    # Get deals from last 24 hours
    from_time = datetime.now() - timedelta(hours=hours)
    to_time = datetime.now()

    deals = mt5.history_deals_get(from_time, to_time)
    if deals is None:
        return []

    trades = []
    for deal in deals:
        if deal.magic == 123456 and deal.entry == mt5.DEAL_ENTRY_OUT:  # Closing deals
            trades.append({
                'time': datetime.fromtimestamp(deal.time),
                'symbol': deal.symbol,
                'profit': deal.profit,
                'volume': deal.volume
            })

    # Sort by time, most recent first
    trades.sort(key=lambda x: x['time'], reverse=True)
    return trades[:10]  # Last 10 trades

def get_market_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15):
    """Get current market data"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Calculate indicators
    import ta
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=20)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    return df

def display_dashboard():
    """Display the monitoring dashboard"""
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen

    print("🤖 TRADING BOT MONITORING DASHBOARD")
    print("=" * 50)
    print(f"📅 เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Account Information
    print("💰 ข้อมูลบัญชี")
    print("-" * 30)

    if not connect_mt5():
        print("❌ ไม่สามารถเชื่อมต่อ MT5")
        return

    account = get_account_info()
    if account:
        print(f"📋 บัญชี: {account['login']}")
        print(f"💵 ยอดเงิน: ${account['balance']:,.2f}")
        print(f"📈 Equity: ${account['equity']:,.2f}")
        print(f"💼 Margin Used: ${account['margin']:,.2f}")
        print(f"🆓 Margin Free: ${account['margin_free']:,.2f}")
        profit_color = "🟢" if account['profit'] >= 0 else "🔴"
        print(f"{profit_color} กำไร/ขาดทุน: ${account['profit']:,.2f}")
    else:
        print("❌ ไม่สามารถดึงข้อมูลบัญชี")
    print()

    # Open Positions
    print("📊 ตำแหน่งที่เปิดอยู่")
    print("-" * 30)

    positions = get_open_positions()
    if positions:
        for pos in positions:
            profit_color = "🟢" if pos['profit'] >= 0 else "🔴"
            print(f"🎯 {pos['symbol']} {pos['type']} {pos['volume']} lots")
            print(f"   📍 ราคาเปิด: ${pos['price_open']:,.2f}")
            print(f"   📈 ราคาปัจจุบัน: ${pos['price_current']:,.2f}")
            print(f"   {profit_color} P/L: ${pos['profit']:,.2f}")
            print(f"   🛑 Stop Loss: ${pos['sl']:,.2f}")
            print(f"   🎯 Take Profit: ${pos['tp']:,.2f}")
            print()
    else:
        print("📭 ไม่มีตำแหน่งที่เปิดอยู่")
    print()

    # Market Data
    print("📊 ข้อมูลตลาด (XAUUSD - ทองคำ)")
    print("-" * 30)

    df = get_market_data()
    if df is not None and len(df) > 0:
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        print(f"💵 ราคาปิดล่าสุด: ${latest['close']:,.2f}")
        print(f"📈 EMA10: {latest['ema_fast']:,.2f}")
        print(f"📉 EMA20: {latest['ema_slow']:,.2f}")
        print(f"📊 RSI: {latest['rsi']:.2f}")

        # Signal check
        crossover_up = prev['ema_fast'] <= prev['ema_slow'] and latest['ema_fast'] > latest['ema_slow']
        crossover_down = prev['ema_fast'] >= prev['ema_slow'] and latest['ema_fast'] < latest['ema_slow']

        if crossover_up and latest['rsi'] < 40:
            print("🟢 สัญญาณ: BUY (พร้อมเทรด)")
        elif crossover_down and latest['rsi'] > 60:
            print("🔴 สัญญาณ: SELL (พร้อมเทรด)")
        else:
            print("⚪ สัญญาณ: รอ (ยังไม่พร้อมเทรด)")
    else:
        print("❌ ไม่สามารถดึงข้อมูลตลาด")
    print()

    # Recent Trades
    print("📋 การเทรดล่าสุด (24 ชั่วโมง)")
    print("-" * 30)

    trades = get_recent_trades()
    if trades:
        for trade in trades:
            profit_color = "🟢" if trade['profit'] >= 0 else "🔴"
            print(f"{profit_color} {trade['time'].strftime('%H:%M')} | {trade['symbol']} | ${trade['profit']:,.2f}")
    else:
        print("📭 ไม่มีการเทรดล่าสุด")
    print()

    # Risk Metrics
    print("⚠️ ตัวชี้วัดความเสี่ยง")
    print("-" * 30)

    if account:
        balance = account['balance']
        equity = account['equity']

        # Calculate drawdown
        if balance > 0:
            drawdown = ((balance - equity) / balance) * 100
            risk_color = "🟢" if drawdown < 5 else "🟡" if drawdown < 10 else "🔴"
            print(f"{risk_color} Drawdown: {drawdown:.2f}%")

        # Margin usage
        if account['margin'] > 0:
            margin_usage = (account['margin'] / equity) * 100
            margin_color = "🟢" if margin_usage < 50 else "🟡" if margin_usage < 80 else "🔴"
            print(f"{margin_color} Margin Usage: {margin_usage:.1f}%")

    print()
    print("🔄 กำลังอัปเดต... (กด Ctrl+C เพื่อออก)")
    print("=" * 50)

    mt5.shutdown()

def main():
    """Main monitoring function - run once"""
    try:
        display_dashboard()
        print("\n💡 คำแนะนำ:")
        print("- รัน 'python monitor_dashboard.py' เพื่อดูสถานะปัจจุบัน")
        print("- Dashboard จะแสดงข้อมูลแบบ Real-time")
        print("- กด Ctrl+C เพื่อออกจากการ monitoring ต่อเนื่อง")

    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()