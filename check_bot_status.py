"""
Bot Status Checker

This script checks if the trading bot is running and shows its current status.
"""

import psutil
import os
import time
from datetime import datetime, timedelta

def check_bot_process():
    """Check if trading_bot.py is running"""
    print("🔍 กำลังตรวจสอบ Process...")

    bot_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe':
                cmdline = proc.info['cmdline']
                if cmdline and len(cmdline) > 1 and 'trading_bot.py' in cmdline[1]:
                    bot_running = True
                    print(f"✅ พบ Bot Process: PID {proc.info['pid']}")
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not bot_running:
        print("❌ ไม่พบ Bot Process ที่กำลังทำงาน")
        return False

    return True

def check_recent_log_activity():
    """Check if log file has recent activity"""
    print("\n📝 กำลังตรวจสอบ Log File...")

    log_file = 'trading_bot.log'
    if not os.path.exists(log_file):
        print("❌ ไม่พบไฟล์ log")
        return False

    # Check file modification time
    mod_time = os.path.getmtime(log_file)
    mod_datetime = datetime.fromtimestamp(mod_time)
    time_diff = datetime.now() - mod_datetime

    print(f"📅 Log แก้ไขล่าสุด: {mod_datetime}")
    print(f"⏰ ผ่านมาแล้ว: {str(time_diff).split('.')[0]}")

    # Check if log has recent entries (within last 5 minutes)
    if time_diff < timedelta(minutes=5):
        print("✅ Log มีการอัปเดตล่าสุด")
        return True
    else:
        print("⚠️  Log ไม่มีการอัปเดตมานาน")
        return False

def check_mt5_connection():
    """Check if MT5 is running"""
    print("\n💹 กำลังตรวจสอบ MT5...")

    mt5_running = False
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'terminal' in proc.info['name'].lower() and 'mt5' in proc.info['name'].lower():
                mt5_running = True
                print(f"✅ พบ MT5 Process: {proc.info['name']}")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not mt5_running:
        print("❌ ไม่พบ MT5 Terminal ที่กำลังทำงาน")

    return mt5_running

def show_log_summary():
    """Show recent log entries"""
    print("\n📋 Log ล่าสุด 5 บรรทัด:")

    log_file = 'trading_bot.log'
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-5:]
                for line in lines:
                    print(f"  {line.strip()}")
        except Exception as e:
            print(f"❌ ไม่สามารถอ่าน log: {e}")
    else:
        print("  ไม่พบไฟล์ log")

def main():
    print("🤖 Trading Bot Status Checker")
    print("=" * 40)

    bot_ok = check_bot_process()
    log_ok = check_recent_log_activity()
    mt5_ok = check_mt5_connection()

    print("\n" + "=" * 40)
    print("📊 สรุปสถานะ:")

    if bot_ok and log_ok:
        print("✅ Bot กำลังทำงานปกติ")
    elif bot_ok and not log_ok:
        print("⚠️  Bot Process ทำงาน แต่ไม่มีการอัปเดต log")
    elif not bot_ok and log_ok:
        print("⚠️  ไม่พบ Process แต่ log ยังอัปเดต")
    else:
        print("❌ Bot ไม่ทำงาน")

    if mt5_ok:
        print("✅ MT5 Terminal ทำงาน")
    else:
        print("❌ MT5 Terminal ไม่ทำงาน")

    show_log_summary()

    print("\n💡 เคล็ดลับ:")
    print("- หาก Bot ไม่ทำงาน: ใช้คำสั่ง 'python trading_bot.py'")
    print("- หาก MT5 ไม่ทำงาน: เปิด MT5 Terminal และเข้าสู่บัญชี demo")
    print("- ตรวจสอบ log เพิ่มเติม: 'Get-Content trading_bot.log -Tail 10'")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("❌ ต้องติดตั้ง psutil ก่อน: pip install psutil")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")