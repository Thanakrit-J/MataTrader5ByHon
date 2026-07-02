import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta, time, pytz
import psycopg2
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional

# ===== CONFIG =====
SYMBOL, TIMEFRAME = "XAUUSD", mt5.TIMEFRAME_M15
LOCAL_TIMEZONE = pytz.timezone("Asia/Bangkok")

# ปรับค่าสอดคล้องกับพอร์ตทองคำ ทุน $30
RISK_PER_TRADE         = 0.03
TRAILING_STOP_POINTS   = 350
TAKE_PROFIT_MULTIPLIER = 2.0
SL_ATR_MULTIPLIER      = 1.5

ORDER_PROFIT_MIN    = 0.5
ORDER_PROFIT_MAX    = 2.0
DAILY_PROFIT_TARGET = 3.0
DAILY_LOSS_LIMIT    = -3.0

FAST_EMA, SLOW_EMA, RSI_PERIOD, ATR_PERIOD = 10, 20, 14, 14
SCORE_THRESHOLD, MAX_SPREAD_POINTS, MAGIC = 3, 40, 20250101
TRADING_HOUR_START, TRADING_HOUR_END, RECONNECT_WAIT, MAX_RECONNECT = 7, 16, 5, 10
MIN_FREE_MARGIN_BUFFER, ANALYSIS_HISTORY_DAYS, DASHBOARD_EVERY_N_LOOPS = 1.3, 30, 10

# การเชื่อมต่อฐานข้อมูล PostgreSQL บน Docker
DB_CONFIG = {
    "host": "localhost",
    "database": "mt5_trading",
    "user": "bot_user",
    "password": "BotPassword123",
    "port": "5432"
}

class TradeAnalyzer:
    def __init__(self, magic: int, symbol: str, days_back: int = 30):
        self.magic, self.symbol, self.days_back = magic, symbol, days_back

    def fetch_closed_deals(self) -> pd.DataFrame:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        deals = mt5.history_deals_get(now - timedelta(days=self.days_back), now)
        if not deals: return pd.DataFrame()
        df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df[(df['entry'] == mt5.DEAL_ENTRY_OUT) & (df['magic'] == self.magic) & (df['symbol'] == self.symbol)].copy()
        if df.empty: return df
        df['direction'] = df['type'].map({mt5.DEAL_TYPE_BUY: 'BUY', mt5.DEAL_TYPE_SELL: 'SELL'}).fillna('OTHER')
        df['win'], df['hour_ict'] = df['profit'] > 0, (df['time'].dt.hour + 7) % 24
        return df.reset_index(drop=True)

    def compute_stats(self, df: pd.DataFrame) -> dict:
        if df.empty: return {}
        total, wins = len(df), int(df['win'].sum())
        losses, avg_win, avg_loss = total - wins, df[df['win']]['profit'].mean() or 0, df[~df['win']]['profit'].mean() or 0
        g_win, g_loss = df[df['win']]['profit'].sum(), df[~df['win']]['profit'].sum()
        h_stats = df.groupby('hour_ict').agg(trades=('profit', 'count'), profit=('profit', 'sum'), win_rate=('win', 'mean')).sort_values('profit', ascending=False)
        return {
            'total': total, 'wins': wins, 'losses': losses, 'win_rate': wins / total * 100,
            'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': abs(g_win / g_loss) if g_loss != 0 else float('inf'),
            'actual_rr': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'), 'net_profit': df['profit'].sum(),
            'best_hours': h_stats[h_stats['profit'] > 0].index.tolist()[:3], 'worst_hours': h_stats[h_stats['profit'] < 0].index.tolist(),
            'hour_stats': h_stats, 'dir_stats': {d: {'count': len(df[df['direction'] == d]), 'win_rate': df[df['direction'] == d]['win'].mean() * 100 if len(df[df['direction'] == d]) > 0 else 0, 'net': df[df['direction'] == d]['profit'].sum()} for d in ['BUY', 'SELL']}
        }

    def print_report(self, label: str = ""):
        stats = self.compute_stats(self.fetch_closed_deals())
        sep = "═" * 56
        print(f"\n{sep}\n  📊 ANALYSIS REPORT {label}\n{sep}")
        if not stats: print("  ยังไม่มีประวัติเทรดเพียงพอในช่วง 30 วัน")
        else:
            print(f"  เทรดทั้งหมด: {stats['total']} | Net: ${stats['net_profit']:+.2f} | WR: {stats['win_rate']:.1f}% | PF: {stats['profit_factor']:.2f}")
            print(f"  RR จริง: 1:{stats['actual_rr']:.2f} | AvgWin: ${stats['avg_win']:+.2f} | AvgLoss: ${stats['avg_loss']:+.2f}")
            for d, s in stats['dir_stats'].items(): print(f"  {d}: {s['count']} เทรด | WR {s['win_rate']:.0f}% | Net ${s['net']:+.2f}")
            print(f"  ⭐ ทอง (ICT): {stats['best_hours']} | ⚠️ เลี่ยง (ICT): {stats['worst_hours'][:3]}")
        print(sep)

    def get_next_trade_advice(self) -> str:
        stats = self.compute_stats(self.fetch_closed_deals())
        if not stats or stats['total'] < 5: return "  📋 ข้อมูลยังน้อย — เทรดตาม config ปกติ"
        h_ict = (datetime.now(timezone.utc).hour + 7) % 24
        lines = []
        if h_ict in stats['hour_stats'].index:
            if stats['hour_stats'].loc[h_ict, 'profit'] < 0: lines.append(f"  ⚠️ ชั่วโมงนี้ ({h_ict}:xx ICT) มักขาดทุน")
            elif stats['hour_stats'].loc[h_ict, 'win_rate'] >= 0.6: lines.append(f"  ⭐ ชั่วโมงนี้ ({h_ict}:xx ICT) ประวัติดี")
        if stats['actual_rr'] < TAKE_PROFIT_MULTIPLIER * 0.8: lines.append("  💡 RR ต่ำ — รอสัญญาณชัดกว่านี้")
        return "\n".join(lines) if lines else "  ✅ ประวัติดี — เทรดตาม config ได้เลย"

class AIBot:
    def __init__(self):
        self._connect()
        self._check_account_type()
        self._init_db()
        self.analyzer = TradeAnalyzer(MAGIC, SYMBOL, ANALYSIS_HISTORY_DAYS)
        self.daily_trades, self.today, self.processed_deals, self.trade_history = 0, self._utc_now().date(), set(), []
        self.recent_losses, self.daily_profit_total, self.daily_loss_total = deque(maxlen=10), 0.0, 0.0
        self.daily_profit_halt, self.daily_loss_halt, self.last_candle_time, self._loop_count = False, False, None, 0
        self.current_context = {}

    def _connect(self):
        global MAX_RECONNECT, RECONNECT_WAIT
        for i in range(1, MAX_RECONNECT + 1):
            if mt5.initialize(): return print("✅ MT5 connected")
            print(f"⚠️ MT5 init failed ({i}/{MAX_RECONNECT})"); time.sleep(RECONNECT_WAIT)
        raise RuntimeError("❌ Cannot connect to MT5")

    def _check_account_type(self):
        acc = mt5.account_info()
        if not acc: raise RuntimeError("❌ ไม่สามารถดึงข้อมูลบัญชีได้")
        if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            print("\n" + "═"*60 + "\n🛑 LIVE ACCOUNT DETECTED! บอทนี้ทำงานบนบัญชี DEMO เท่านั้นเพื่อความปลอดภัย\n" + "═"*60 + "\n")
            mt5.shutdown(); exit()
        print(f"🔒 Demo Verified (พอร์ต: {acc.login} | Server: {acc.server})")

    def _init_db(self):
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS closed_trades (
                            ticket BIGINT PRIMARY KEY,
                            symbol VARCHAR(20) NOT NULL,
                            direction VARCHAR(10) NOT NULL,
                            profit NUMERIC(10, 2) NOT NULL,
                            rsi_entry NUMERIC(5, 2),
                            atr_entry NUMERIC(10, 5),
                            score_entry INT,
                            htf_trend VARCHAR(15),
                            closed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE INDEX IF NOT EXISTS idx_closed_trades_time ON closed_trades (closed_at DESC);
                    """)
                    conn.commit()
            print("💾 Database Brain-Table ready")
        except Exception as e:
            print(f"⚠️ Database init failed: {e}")

    def _save_trade_to_db(self, ticket: int, symbol: str, direction: str, profit: float):
        try:
            ctx = self.current_context.get(ticket, {"rsi": None, "atr": None, "score": None, "htf": "unknown"})
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO closed_trades (ticket, symbol, direction, profit, rsi_entry, atr_entry, score_entry, htf_trend, closed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticket) DO NOTHING;
                    """, (ticket, symbol, direction, profit, ctx["rsi"], ctx["atr"], ctx["score"], ctx["htf"], datetime.now(timezone.utc)))
                    conn.commit()
            if ticket in self.current_context: del self.current_context[ticket]
            print(f"💾 Saved Ticket {ticket} with Market Context to DB successfully")
        except Exception as e:
            print(f"⚠️ Failed to save trade to DB: {e}")

    # 🧠 ฟังก์ชันใหม่: ดึงสถิติจากฐานข้อมูลมาคำนวณและปรับตัวกรองความปลอดภัยในการเทรดต่อ
    def _get_db_logic_filter(self, current_rsi: float, direction: str) -> bool:
        try:
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    # 1. เช็กประวัติการเข้าช่วงค่า RSI ใกล้เคียงกัน (+- 5) ว่าในอดีตถ้าเข้าทิศทางนี้แล้วพังบ่อยไหม
                    cur.execute("""
                        SELECT COUNT(*), SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END)
                        FROM closed_trades 
                        WHERE direction = %s AND rsi_entry BETWEEN %s AND %s;
                    """, (direction, current_rsi - 5, current_rsi + 5))
                    res = cur.fetchone()
                    
                    if res and res[0] >= 3: # ถ้ามีประวัติเทรดในโซนนี้อย่างน้อย 3 ครั้ง
                        win_rate = (res[1] / res[0]) * 100
                        if win_rate < 40.0:
                            print(f"   🧠 DB Filter Active: โซน RSI นี้ ({current_rsi:.1f}) ฝั่ง {direction} มี Win Rate ต่ำมากในอดีต ({win_rate:.1f}%) -> 🛑 สั่งระงับไม้")
                            return False
                    
                    # 2. เช็กผลรวมกำไรสะสมภาพรวมใน Database เพื่อปรับความเข้มงวดของ Score (Dynamic Threshold)
                    cur.execute("SELECT SUM(profit) FROM closed_trades;")
                    net_profit = cur.fetchone()[0] or 0.0
                    if net_profit < 0:
                        global SCORE_THRESHOLD
                        SCORE_THRESHOLD = 4 # ถ้ารวมแล้วระบบพอร์ตติดลบ ให้เพิ่มเกณฑ์ความแม่นยำจาก 3 แต้มเป็น 4 แต้ม
                        print(f"   🧠 DB Filter Active: ภาพรวมระบบติดลบ (${net_profit:.2f}) -> ยกระดับความปลอดภัย SCORE_THRESHOLD = 4")
                    else:
                        SCORE_THRESHOLD = 3 # ถ้ากลับมากำไร ให้ใช้เกณฑ์มาตรฐานเดิม
                        
            return True
        except Exception as e:
            print(f"   ⚠️ DB Filter Error: {e} (ข้ามไปใช้ระบบปกติ)")
            return True

    def _ensure_connected(self): return self._connect() if mt5.terminal_info() is None else True
    def _utc_now(self): return datetime.now(tz=pytz.utc)
    def _is_trading_hours(self): return TRADING_HOUR_START <= self._utc_now().hour < TRADING_HOUR_END

    def get_data(self, timeframe, bars: int = 100) -> pd.DataFrame:
        self._ensure_connected()
        rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)
        if rates is None or len(rates) == 0: raise ValueError(f"No data for {SYMBOL}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema_fast'], df['ema_slow'] = ta.trend.ema_indicator(df['close'], FAST_EMA), ta.trend.ema_indicator(df['close'], SLOW_EMA)
        df['rsi'] = ta.momentum.rsi(df['close'], RSI_PERIOD)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], ATR_PERIOD)
        return df

    def higher_trend(self) -> str:
        df = self.add_indicators(self.get_data(mt5.TIMEFRAME_H1))
        f, s = df['ema_fast'].iloc[-1], df['ema_slow'].iloc[-1]
        return 'neutral' if abs(f - s) / s < 0.0001 else ('up' if f > s else 'down')

    def calculate_score(self, df: pd.DataFrame) -> int:
        p, f, s, r, r_prev = df['close'].iloc[-1], df['ema_fast'].iloc[-1], df['ema_slow'].iloc[-1], df['rsi'].iloc[-1], df['rsi'].iloc[-5]
        rng = df['high'].tail(20).max() - df['low'].tail(20).min()
        pos = (p - df['low'].tail(20).min()) / rng if rng > 0 else 0.5
        return (2 if f > s else -2) + (1 if r > 55 else (-1 if r < 45 else 0)) + (1 if r > r_prev else -1) + (1 if pos > 0.7 else (-1 if pos < 0.3 else 0)) + (1 if abs(f - s) / p > 0.0003 else -1)

    def open_positions(self): self._ensure_connected(); pos = mt5.positions_get(symbol=SYMBOL); return [p for p in pos if p.magic == MAGIC] if pos else []
    def has_open_position(self): return len(self.open_positions()) > 0

    def update_trailing_stops(self):
        for p in self.open_positions():
            tick, pt = mt5.symbol_info_tick(SYMBOL), mt5.symbol_info(SYMBOL).point
            trail = TRAILING_STOP_POINTS * pt
            new_sl = tick.bid - trail if p.type == mt5.ORDER_TYPE_BUY else tick.ask + trail
            if p.sl == 0 or (p.type == mt5.ORDER_TYPE_BUY and new_sl > p.sl + pt) or (p.type == mt5.ORDER_TYPE_SELL and new_sl < p.sl - pt):
                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "sl": new_sl})

    def track_closed_trades(self):
        self._ensure_connected()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        deals = mt5.history_deals_get(now - timedelta(hours=2), now) or []
        new_closed = 0
        for d in deals:
            if d.entry != mt5.DEAL_ENTRY_OUT or d.ticket in self.processed_deals or d.magic != MAGIC: continue
            self.processed_deals.add(d.ticket)
            new_closed += 1
            
            direction = "BUY" if d.type == mt5.ORDER_TYPE_BUY else "SELL"
            self._save_trade_to_db(d.ticket, SYMBOL, direction, d.profit)
            
            if d.profit >= 0:
                self.daily_profit_total += d.profit
                if self.daily_profit_total >= DAILY_PROFIT_TARGET: self.daily_profit_halt = True
            else:
                self.daily_loss_total += d.profit
                self.recent_losses.append({"type": direction})
                if self.daily_loss_total <= DAILY_LOSS_LIMIT: self.daily_loss_halt = True
        if new_closed > 0:
            self.analyzer.print_report(f"(หลังปิด {new_closed} trade)")
            print(f"  💬 แนะนำสำหรับ trade ถัดไป:\n{self.analyzer.get_next_trade_advice()}\n")

    def lot_size(self, sl_points: float, signal: str, price: float) -> float:
        acc, sinfo = mt5.account_info(), mt5.symbol_info(SYMBOL)
        lot = (acc.balance * RISK_PER_TRADE) / (sl_points or 1.0)
        lot = round(round(max(sinfo.volume_min, min(lot, sinfo.volume_max)) / sinfo.volume_step) * sinfo.volume_step, 2)
        otype = mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL
        while lot >= sinfo.volume_min:
            margin = mt5.order_calc_margin(otype, SYMBOL, lot, price)
            if margin and margin * MIN_FREE_MARGIN_BUFFER <= acc.margin_free: break
            lot = round(lot - sinfo.volume_step, 2)
        return max(sinfo.volume_min, lot)

    def close_all_positions(self, reason: str):
        for p in self.open_positions():
            cp = mt5.symbol_info_tick(SYMBOL).bid if p.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).ask
            mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": p.volume, "type": mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY, "position": p.ticket, "price": cp, "deviation": 10, "magic": MAGIC, "comment": reason})

    def check_profit_target(self):
        fl = mt5.account_info().equity - mt5.account_info().balance
        if fl >= ORDER_PROFIT_MAX: self.close_all_positions("ORDER_MAX_PROFIT")
        elif fl >= ORDER_PROFIT_MIN: self.close_all_positions("ORDER_MIN_PROFIT")

    def analyze(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < 50 or not self._is_trading_hours(): return None
        tick, sinfo = mt5.symbol_info_tick(SYMBOL), mt5.symbol_info(SYMBOL)
        if (tick.ask - tick.bid) / sinfo.point > MAX_SPREAD_POINTS or df['atr'].iloc[-1] < df['atr'].rolling(20).mean().iloc[-1] * 0.8: return None
        
        rsi_now = float(df['rsi'].iloc[-1])
        score, ht = self.calculate_score(df), self.higher_trend()
        
        # เรียกใช้ตัวกรองสถิติจาก Database เพื่อปรับเงื่อนไข Score Threshold ล่าสุด
        # บอทจะเช็กว่าภาพรวมติดลบไหม ถ้าติดลบจะบังคับปรับให้เงื่อนไขผ่านยากขึ้นชั่วคราว
        self._get_db_logic_filter(rsi_now, 'BUY' if score > 0 else 'SELL') 
        
        decision = 'BUY' if score >= SCORE_THRESHOLD else ('SELL' if score <= -SCORE_THRESHOLD else None)
        if not decision or (decision == 'BUY' and ht == 'down') or (decision == 'SELL' and ht == 'up'): return None
        if sum(1 for r in list(self.recent_losses)[-5:] if r["type"] == decision) >= 3: return None
        
        # 🧠 ยิง SQL ไปตรวจสอบประวัติในโซน RSI ปัจจุบันว่าอดีตเคยพังไหม ถ้าเคยพังบ่อยให้สละสิทธิ์การเข้าเทรดรอบนี้
        if not self._get_db_logic_filter(rsi_now, decision): return None
        
        self.temp_market_context = {
            "rsi": rsi_now,
            "atr": float(df['atr'].iloc[-1]),
            "score": int(score),
            "htf": str(ht)
        }
        return decision

    def place_trade(self, signal: str, df: pd.DataFrame):
        sinfo, tick = mt5.symbol_info(SYMBOL), mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if signal == 'BUY' else tick.bid
        sl_dist = df['atr'].iloc[-1] * SL_ATR_MULTIPLIER
        sl_price = price - sl_dist if signal == 'BUY' else price + sl_dist
        tp_price = price + (sl_dist * TAKE_PROFIT_MULTIPLIER) if signal == 'BUY' else price - (sl_dist * TAKE_PROFIT_MULTIPLIER)
        lot = self.lot_size(sl_dist / sinfo.point, signal, price)
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL, SYMBOL, lot, price)
        if not margin or margin * MIN_FREE_MARGIN_BUFFER > mt5.account_info().margin_free: return
        
        r = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lot, "type": mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL, "price": price, "sl": round(sl_price, sinfo.digits), "tp": round(tp_price, sinfo.digits), "deviation": 10, "magic": MAGIC, "comment": "AI_BOT_V2"})
        
        if r.retcode == mt5.TRADE_RETCODE_DONE: 
            self.daily_trades += 1
            if hasattr(self, 'temp_market_context'):
                self.current_context[r.order] = self.temp_market_context

    def run(self):
        self.analyzer.print_report("(เริ่มต้น bot)")
        while True:
            self._loop_count += 1
            try:
                self._ensure_connected()
                self._check_account_type()
                if self._utc_now().date() != self.today:
                    self.today, self.daily_trades, self.daily_profit_total, self.daily_loss_total, self.daily_profit_halt, self.daily_loss_halt, self.last_candle_time, self._loop_count = self._utc_now().date(), 0, 0.0, 0.0, False, False, None, 0
                    self.analyzer.print_report("(เริ่มต้นวันใหม่)")
                self.update_trailing_stops()
                self.track_closed_trades()
                if self.daily_loss_halt or self.daily_profit_halt: time.sleep(60); continue
                if self.has_open_position(): self.check_profit_target(); time.sleep(30); continue
                if not self._is_trading_hours(): time.sleep(60); continue
                df = self.add_indicators(self.get_data(TIMEFRAME))
                if df['time'].iloc[-1] == self.last_candle_time: time.sleep(30); continue
                self.last_candle_time = df['time'].iloc[-1]
                signal = self.analyze(df)
                if signal: self.place_trade(signal, df)
                if self._loop_count % DASHBOARD_EVERY_N_LOOPS == 0: self.analyzer.print_report(f"(periodic — loop #{self._loop_count})")
            except Exception as e:
                print(f"❌ ERROR: {e}"); time.sleep(RECONNECT_WAIT); self._connect()
            time.sleep(60)

if __name__ == "__main__":
    AIBot().run()