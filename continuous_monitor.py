import os
import time
import MetaTrader5 as mt5
from datetime import datetime
from simple_status import check_status


REFRESH_INTERVAL = 30


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    print("🚀 เริ่ม Bot Monitoring...\n")

    while True:
        try:
            clear_screen()  # ← ล้างจอก่อน print

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"⏰ เวลา: {now}")
            print("-" * 50)

            check_status()

            print("-" * 50)
            print(f"🔄 อัปเดตอีกครั้งใน {REFRESH_INTERVAL} วินาที (Ctrl+C เพื่อหยุด)")

            time.sleep(REFRESH_INTERVAL)

        except KeyboardInterrupt:  # ← ต้องอยู่ก่อน Exception
            print("\n👋 หยุดการ monitoring")
            mt5.shutdown()
            break

        except Exception as e:
            print(f"❌ Error: {e}")
            print("⏳ ลองใหม่ใน 10 วินาที...")
            time.sleep(10)


if __name__ == "__main__":
    main()