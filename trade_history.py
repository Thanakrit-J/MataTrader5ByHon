"""
Trade History Viewer for MT5 Trading Bot

This script reads the trading bot log and displays a summary of all trades
including duration and profit/loss information.
"""

import re
from datetime import datetime
import os

def parse_trade_log(log_file='trading_bot.log'):
    """Parse the trading log for trade information"""
    if not os.path.exists(log_file):
        print(f"❌ ไม่พบไฟล์ log: {log_file}")
        return []

    trades = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for trade closed messages
            trade_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - TRADE CLOSED: (\d+) \| ([A-Z]+) (BUY|SELL) \| Duration: (.+?) \| P/L: ([\+\-\$]\d+\.\d+)', line)
            if trade_match:
                timestamp_str, ticket, symbol, direction, duration, pl = trade_match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                trades.append({
                    'timestamp': timestamp,
                    'ticket': ticket,
                    'symbol': symbol,
                    'direction': direction,
                    'duration': duration,
                    'profit_loss': pl
                })

            # Also look for emergency closes
            emergency_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - EMERGENCY CLOSE: (\d+) \| ([A-Z]+) (BUY|SELL) \| Duration: (.+?) \| P/L: ([\+\-\$]\d+\.\d+)', line)
            if emergency_match:
                timestamp_str, ticket, symbol, direction, duration, pl = emergency_match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                trades.append({
                    'timestamp': timestamp,
                    'ticket': ticket,
                    'symbol': symbol,
                    'direction': direction,
                    'duration': duration,
                    'profit_loss': pl,
                    'emergency': True
                })

    return trades

def display_trade_summary(trades):
    """Display a summary of all trades"""
    if not trades:
        print("📊 ไม่มีข้อมูลการเทรด")
        return

    print("📊 สรุปการเทรด")
    print("=" * 60)

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if not t['profit_loss'].startswith('-') and not t['profit_loss'].startswith('$'))
    losing_trades = total_trades - winning_trades

    total_pl = 0
    for trade in trades:
        pl_str = trade['profit_loss']
        if pl_str.startswith('$'):
            pl_value = float(pl_str[1:])
        elif pl_str.startswith('+$'):
            pl_value = float(pl_str[2:])
        else:
            pl_value = float(pl_str[1:])  # Handle -$ format
        total_pl += pl_value

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"จำนวนการเทรดทั้งหมด: {total_trades}")
    print(f"การเทรดที่ชนะ: {winning_trades}")
    print(f"การเทรดที่แพ้: {losing_trades}")
    print(f"อัตราการชนะ: {win_rate:.1f}%")
    print(f"กำไร/ขาดทุนรวม: ${total_pl:.2f}")
    print()

    print("รายละเอียดการเทรด:")
    print("-" * 60)
    for i, trade in enumerate(trades, 1):
        emergency = " (ปิดด่วน)" if trade.get('emergency') else ""
        print(f"{i:2d}. {trade['timestamp']} | {trade['symbol']} {trade['direction']} | {trade['duration']} | {trade['profit_loss']}{emergency}")

def main():
    print("🔍 กำลังอ่านประวัติการเทรด...")
    trades = parse_trade_log()

    display_trade_summary(trades)

    print("\n💡 หมายเหตุ:")
    print("- Bot จะบันทึกการเทรดทุกครั้งที่มีการปิด position")
    print("- ระยะเวลาการเทรดแสดงเป็น ชั่วโมง:นาที:วินาที")
    print("- กำไร/ขาดทุนแสดงเป็นดอลลาร์")

if __name__ == "__main__":
    main()