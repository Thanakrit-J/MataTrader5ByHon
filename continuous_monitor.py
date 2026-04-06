"""
Continuous Bot Monitor
Run this to continuously monitor the bot status
"""

import os
import time
from simple_status import check_status

def main():
    try:
        while True:
            check_status()
            print("\n" + "="*50)
            print("🔄 กำลังอัปเดตใน 30 วินาที... (กด Ctrl+C เพื่อหยุด)")
            time.sleep(30)
            os.system('cls' if os.name == 'nt' else 'clear')

    except KeyboardInterrupt:
        print("\n👋 หยุดการ monitoring")

if __name__ == "__main__":
    main()