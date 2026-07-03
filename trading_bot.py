import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time, pytz
import psycopg2
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple

# ==========================================
# ⚙️ GLOBAL CONFIGURATION
# ==========================================
SYMBOL, TIMEFRAME = "XAUUSD", mt5.TIMEFRAME_M5
LOCAL_TIMEZONE = pytz.timezone("Asia/Bangkok")

RISK_PER_TRADE         = 0.01        # ลดจาก 0.03 เหลือ 1%
MIN_VOLUME             = 50          # ต้องมี Volume >= 50 ก่อนเข้าเทรด (ลดจาก 100)
TRAILING_STOP_POINTS   = 120     
TAKE_PROFIT_MULTIPLIER = 3.0         # เพิ่มจาก 2.5 → 1:3 Risk/Reward     
SL_ATR_MULTIPLIER      = 1.5

ORDER_PROFIT_MIN    = 0.5
ORDER_PROFIT_MAX    = 2.0
DAILY_PROFIT_TARGET = 100.0
DAILY_LOSS_LIMIT    = -100.0

MAX_SPREAD_POINTS, MAGIC = 45, 20260702
TRADING_HOUR_START, TRADING_HOUR_END = 7, 20  # เวลา UTC: 07.00 - 20.00 (เทรดถึงตี 3 เวลาไทย)
RECONNECT_WAIT, MAX_RECONNECT = 5, 10
MIN_FREE_MARGIN_BUFFER = 1.3
ANALYSIS_HISTORY_DAYS  = 30
DASHBOARD_EVERY_N_LOOPS = 20  

DB_CONFIG = {
    "host": "localhost",
    "database": "mt5_trading",
    "user": "bot_user",
    "password": "BotPassword123",
    "port": "5433"  # <--- เปลี่ยนเป็นเลข 5433 ให้ตรงกับ Docker
}

# ==========================================
# 💾 1. DATABASE MANAGEMENT MODULE
# ==========================================
class DatabaseManager:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.db_ready = True
        self.last_error = None
        self.initialize_tables()

    def _get_connection(self):
        return psycopg2.connect(**self.config)

    def initialize_tables(self) -> None:
        try:
            with self._get_connection() as conn:
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
                        CREATE TABLE IF NOT EXISTS candles (
                            symbol VARCHAR(20) NOT NULL,
                            timeframe INT NOT NULL,
                            candle_time TIMESTAMP WITH TIME ZONE NOT NULL,
                            open_price NUMERIC(12, 5) NOT NULL,
                            high_price NUMERIC(12, 5) NOT NULL,
                            low_price NUMERIC(12, 5) NOT NULL,
                            close_price NUMERIC(12, 5) NOT NULL,
                            volume NUMERIC(12, 2),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (symbol, timeframe, candle_time)
                        );
                        CREATE INDEX IF NOT EXISTS idx_closed_trades_time ON closed_trades (closed_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_candles_time ON candles (symbol, timeframe, candle_time DESC);
                    """)
                    conn.commit()
            print("💾 [Database] Storage setup and verified.")
        except Exception as e:
            self.db_ready = False
            self.last_error = str(e)
            print(f"⚠️ [Database] Init failure: {e}")

    def save_trade(self, ticket: int, symbol: str, direction: str, profit: float, context: Dict[str, Any]) -> None:
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO closed_trades (ticket, symbol, direction, profit, rsi_entry, atr_entry, score_entry, htf_trend, closed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticket) DO NOTHING;
                    """, (ticket, symbol, direction, profit, context.get("rsi", 50.0), context.get("atr", 0.0), context.get("score", 5), context.get("htf", "PA_Trade"), datetime.now(timezone.utc)))
                    conn.commit()
            print(f"💾 [Database] Successfully logged Ticket {ticket} data.")
        except Exception as e:
            self.db_ready = False
            self.last_error = str(e)
            print(f"⚠️ [Database] Failed to save log for Ticket {ticket}: {e}")

    def save_candle(self, symbol: str, timeframe: int, candle: Dict[str, Any]) -> None:
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO candles (symbol, timeframe, candle_time, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timeframe, candle_time) DO NOTHING;
                    """, (
                        symbol,
                        timeframe,
                        candle["time"],
                        candle.get("open", 0.0),
                        candle.get("high", 0.0),
                        candle.get("low", 0.0),
                        candle.get("close", 0.0),
                        candle.get("tick_volume", 0.0),
                    ))
                    conn.commit()
        except Exception as e:
            self.db_ready = False
            self.last_error = str(e)
            print(f"⚠️ [Database] Failed to save candle for {symbol}: {e}")

    def load_recent_candles(self, symbol: str, timeframe: int, limit: int = 60) -> pd.DataFrame:
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT candle_time, open_price, high_price, low_price, close_price, volume
                        FROM candles
                        WHERE symbol = %s AND timeframe = %s
                        ORDER BY candle_time ASC
                        LIMIT %s;
                    """, (symbol, timeframe, limit))
                    rows = cur.fetchall()
            if not rows:
                return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df
        except Exception as e:
            self.db_ready = False
            self.last_error = str(e)
            print(f"⚠️ [Database] Failed to load candles for {symbol}: {e}")
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


# ==========================================
# 📈 2. MARKET & PRICE ACTION ANALYZER MODULE
# ==========================================
class MarketAnalyzer:
    def __init__(self, symbol: str, timeframe: int, max_spread: int):
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_spread = max_spread

    def fetch_market_data(self, bars_count: int = 60) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars_count)
        if rates is None or len(rates) == 0:
            raise ValueError(f"❌ [Analyzer] Failed to copy market data for {self.symbol}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df['atr'] = df['high'] - df['low']
        return df

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def analyze_candle_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], str]:
        if len(df) < 20:
            return None, "not enough candles"

        # 🔊 ตรวจสอบ Volume ก่อน (ต้องมี Volume >= MIN_VOLUME)
        last_volume = float(df['tick_volume'].iloc[-1]) if 'tick_volume' in df.columns else 0
        if last_volume < MIN_VOLUME:
            return None, f"volume too low ({last_volume} < {MIN_VOLUME})"

        closes = pd.to_numeric(df['close'], errors='coerce').astype(float)
        ema_fast = self._ema(closes, 9)
        ema_slow = self._ema(closes, 21)
        rsi = self._rsi(closes, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        last_close = float(last['close'])
        last_open = float(last['open'])
        prev_high = float(prev['high'])
        prev_low = float(prev['low'])
        prev2_high = float(prev2['high'])
        prev2_low = float(prev2['low'])
        last_body = abs(last_close - last_open)
        prev_body = abs(float(prev['close']) - float(prev['open']))
        prev2_body = abs(float(prev2['close']) - float(prev2['open']))

        ema_fast_val = float(ema_fast.iloc[-1])
        ema_slow_val = float(ema_slow.iloc[-1])
        rsi_val = float(rsi.iloc[-1])

        if last_body <= 0:
            return None, "latest candle body is too small"

        bullish_break = last_close > last_open and last_close > prev_high and last_close > prev2_high
        bearish_break = last_close < last_open and last_close < prev_low and last_close < prev2_low

        if bullish_break and last_body >= max(prev_body, prev2_body) * 0.8 and last_close > ema_fast_val and ema_fast_val > ema_slow_val and rsi_val > 50:
            return 'BUY', f"bullish breakout, close={last_close:.2f}, EMA9/21={ema_fast_val:.2f}/{ema_slow_val:.2f}, RSI={rsi_val:.1f}"

        if bearish_break and last_body >= max(prev_body, prev2_body) * 0.8 and last_close < ema_fast_val and ema_fast_val < ema_slow_val and rsi_val < 50:
            return 'SELL', f"bearish breakout, close={last_close:.2f}, EMA9/21={ema_fast_val:.2f}/{ema_slow_val:.2f}, RSI={rsi_val:.1f}"

        return None, f"no valid signal, close={last_close:.2f}, EMA9/21={ema_fast_val:.2f}/{ema_slow_val:.2f}, RSI={rsi_val:.1f}"

    def is_spread_valid(self) -> bool:
        tick = mt5.symbol_info_tick(self.symbol)
        sinfo = mt5.symbol_info(self.symbol)
        if not tick or not sinfo:
            return False
        current_spread = (tick.ask - tick.bid) / sinfo.point
        return current_spread <= self.max_spread

    def analyze_zones_and_signals(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < 40:
            return None

        # คำนวณหาแนวรับ-แนวต้าน (Demand / Supply Zones) ย้อนหลัง 30 แท่ง
        past_df = df.iloc[:-1]
        demand_zone = float(past_df['low'].tail(30).min())
        supply_zone = float(past_df['high'].tail(30).max())

        # ดึงราคาแท่งปัจจุบัน และแท่งที่พึ่งจบไป
        c_open, c_close = df['open'].iloc[-1], df['close'].iloc[-1]
        c_high, c_low = df['high'].iloc[-1], df['low'].iloc[-1]
        p_open, p_close = df['open'].iloc[-2], df['close'].iloc[-2]
        p_high, p_low = df['high'].iloc[-2], df['low'].iloc[-2]

        buffer = 1.5

        # 🟢 Check Bullish Engulfing inside Demand Zone
        if c_low <= demand_zone + buffer:
            if c_close > c_open and p_close < p_open and c_close > p_open and c_open < p_close:
                print(f"🔥 [Signal] Demand Zone Tested ({demand_zone}) + Bullish Engulfing! -> BUY")
                return 'BUY'

        # 🟢 Fallback: simple bullish reversal near demand zone
        if c_low <= demand_zone + buffer:
            if c_close > c_open and c_close > p_close and c_low < p_low:
                print(f"🔥 [Signal] Fallback bullish near demand zone -> BUY")
                return 'BUY'

        # 🔴 Check Bearish Engulfing inside Supply Zone
        if c_high >= supply_zone - buffer:
            if c_close < c_open and p_close > p_open and c_close < p_open and c_open > p_close:
                print(f"🔥 [Signal] Supply Zone Tested ({supply_zone}) + Bearish Engulfing! -> SELL")
                return 'SELL'

        # 🔴 Fallback: simple bearish reversal near supply zone
        if c_high >= supply_zone - buffer:
            if c_close < c_open and c_close < p_close and c_high > p_high:
                print(f"🔥 [Signal] Fallback bearish near supply zone -> SELL")
                return 'SELL'

        return None


# ==========================================
# 🏹 3. TRADE EXECUTION & RISK MANAGEMENT
# ==========================================
class TradeExecutionManager:
    def __init__(self, symbol: str, magic: int, risk_percent: float):
        self.symbol = symbol
        self.magic = magic
        self.risk_percent = risk_percent

    def get_active_positions(self) -> list:
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return []
        return [p for p in positions if p.magic == self.magic]

    def has_open_position(self) -> bool:
        return len(self.get_active_positions()) > 0

    def calculate_lot_size(self, sl_points: float, entry_price: float) -> float:
        acc = mt5.account_info()
        sinfo = mt5.symbol_info(self.symbol)
        if not acc or not sinfo:
            return 0.01

        risk_amount = acc.balance * self.risk_percent
        lot = risk_amount / (sl_points if sl_points > 0 else 1.0)
        lot = round(round(max(sinfo.volume_min, min(lot, sinfo.volume_max)) / sinfo.volume_step) * sinfo.volume_step, 2)
        
        # ตรวจสอบหลักประกัน (Margin Allocation Buffer)
        otype = mt5.ORDER_TYPE_BUY
        margin = mt5.order_calc_margin(otype, self.symbol, lot, entry_price)
        while margin and margin * MIN_FREE_MARGIN_BUFFER > acc.margin_free and lot >= sinfo.volume_min:
            lot = round(lot - sinfo.volume_step, 2)
            margin = mt5.order_calc_margin(otype, self.symbol, lot, entry_price)

        return max(sinfo.volume_min, lot)

    def execute_market_order(self, direction: str, candle_height: float, rsi_value: float = 50.0, atr_value: float = 0.0) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
        sinfo = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        if not sinfo or not tick:
            return None, None, None, None

        is_buy = direction == 'BUY'
        price = tick.ask if is_buy else tick.bid
        # 🧮 SL สมดุล: 1.3x candle height หรือ 0.8x ATR → Risk/Reward 1:3
        sl_dist = max(candle_height * 1.3, atr_value * 0.8 if atr_value > 0 else 2.0)
        
        sl_price = price - sl_dist if is_buy else price + sl_dist
        tp_price = price + (sl_dist * TAKE_PROFIT_MULTIPLIER) if is_buy else price - (sl_dist * TAKE_PROFIT_MULTIPLIER)
        
        sl_points = sl_dist / sinfo.point
        lot = self.calculate_lot_size(sl_points, price)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": round(sl_price, sinfo.digits),
            "tp": round(tp_price, sinfo.digits),
            "deviation": 10,
            "magic": self.magic,
            "comment": "PA_OOP_V2"
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ [Execution] Order filled successfully! Ticket: {result.order} ({direction} - Lot: {lot}) RSI={rsi_value:.1f} ATR={atr_value:.5f}")
            return result.order, price, rsi_value, atr_value
        else:
            print(f"❌ [Execution] Order rejected. Code: {result.retcode if result else 'Unknown'}")
            return None, None, None, None

    def manage_trailing_stop(self) -> None:
        positions = self.get_active_positions()
        sinfo = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        if not sinfo or not tick or not positions:
            return

        pt = sinfo.point
        trail_distance = TRAILING_STOP_POINTS * pt

        for p in positions:
            if p.type == mt5.ORDER_TYPE_BUY:
                new_sl = tick.bid - trail_distance
                if p.sl == 0 or new_sl > p.sl + pt:
                    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "sl": round(new_sl, sinfo.digits), "tp": p.tp})
            elif p.type == mt5.ORDER_TYPE_SELL:
                new_sl = tick.ask + trail_distance
                if p.sl == 0 or new_sl < p.sl - pt:
                    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "sl": round(new_sl, sinfo.digits), "tp": p.tp})

    def close_all_orders(self, reason: str) -> None:
        positions = self.get_active_positions()
        if not positions:
            print(f"🧹 [Execution] No positions to close for reason: {reason}")
            return

        for p in positions:
            tick = mt5.symbol_info_tick(self.symbol)
            price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
            inverse_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": p.volume,
                "type": inverse_type,
                "position": p.ticket,
                "price": price,
                "deviation": 10,
                "magic": self.magic,
                "comment": reason
            })
            print(f"🔒 [Execution] Closing position {p.ticket} for reason: {reason} -> {'ok' if result and result.retcode == mt5.TRADE_RETCODE_DONE else result.retcode if result else 'unknown'}")
        print(f"🎒 [Execution] Exit cycle complete: {reason}")


# ==========================================
# 🤖 4. MAIN CONTROLLER CORE ENGINE
# ==========================================
class PriceActionTradingBot:
    def __init__(self):
        self._initialize_mt5_connection()
        self.db = DatabaseManager(DB_CONFIG)
        self.analyzer = MarketAnalyzer(SYMBOL, TIMEFRAME, MAX_SPREAD_POINTS)
        self.executor = TradeExecutionManager(SYMBOL, MAGIC, RISK_PER_TRADE)
        
        self.today = self._get_utc_time().date()
        self.daily_profit, self.daily_loss = 0.0, 0.0
        self.processed_tickets = set()
        self.active_contexts = {}
        self.last_candle_time = None
        self.entry_ready_at = None
        self.loop_count = 0
        self.max_floating_pnl = 0.0

    def _initialize_mt5_connection(self) -> None:
        for i in range(1, MAX_RECONNECT + 1):
            if mt5.initialize():
                acc = mt5.account_info()
                if acc and acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO:
                    print(f"🔒 [System] Connected to MT5 Demo Server (พอร์ต: {acc.login})")
                    return
                else:
                    print("🛑 [System] Critical Error: Live account detected. Shutdown safe-lock active.")
                    mt5.shutdown(); exit()
            print(f"⚠️ [System] Connection failed. Retrying ({i}/{MAX_RECONNECT})..."); time.sleep(RECONNECT_WAIT)
        raise RuntimeError("❌ Cannot connect to MetaTrader 5 Terminal")

    def _get_utc_time(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def _is_market_open(self) -> bool:
        return TRADING_HOUR_START <= self._get_utc_time().hour < TRADING_HOUR_END

    def _is_trade_reversing(self, side: str, position) -> bool:
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick or not position:
            return False

        current_price = float(tick.bid if side == 'BUY' else tick.ask)
        entry_price = float(position.price_open)

        if side == 'BUY':
            if current_price <= entry_price:
                return True
        else:
            if current_price >= entry_price:
                return True

        # ตรวจสอบทิศทางแท่งเทียนล่าสุดเพื่อจับ reversal เริ่มต้น
        try:
            df = self.analyzer.fetch_market_data(bars_count=3)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            last_close = float(last['close'])
            last_open = float(last['open'])
            prev_close = float(prev['close'])

            if side == 'BUY' and last_close < last_open and last_close < prev_close:
                return True
            if side == 'SELL' and last_close > last_open and last_close > prev_close:
                return True
        except Exception:
            pass

        return False

    def check_and_reset_daily_limits(self) -> None:
        current_date = self._get_utc_time().date()
        if current_date != self.today:
            print(f"🌅 [System] New trading day started: {current_date}")
            self.today = current_date
            self.daily_profit, self.daily_loss = 0.0, 0.0
            self.loop_count = 0

    def audit_closed_positions(self) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        deals = mt5.history_deals_get(now - timedelta(hours=24), now) or []
        
        for d in deals:
            # สังเกตว่า or d.magic != MAGIC ถูกปิดไว้อยู่
            if d.entry != mt5.DEAL_ENTRY_OUT  or d.ticket in self.processed_tickets:
                continue
                
            self.processed_tickets.add(d.ticket)
            direction = "BUY" if d.type == mt5.ORDER_TYPE_BUY else "SELL"
            ctx = self.active_contexts.get(d.position_id, {"rsi": 50.0, "atr": 0.0, "score": 5, "htf": "PA_Engine"})
            
            self.db.save_trade(d.ticket, SYMBOL, direction, d.profit, ctx)
            
            if d.profit >= 0:
                self.daily_profit += d.profit
            else:
                self.daily_loss += d.profit
                
            if d.position_id in self.active_contexts:
                del self.active_contexts[d.position_id]

    def print_dashboard(self) -> None:
        sep = "═" * 60
        print(f"\n{sep}\n 📊 PRICE ACTION OOP LIVE DASHBOARD\n{sep}")
        print(f" ⏱️ เวลาสากล (UTC): {self._get_utc_time().strftime('%H:%M:%S')} | ไทม์เฟรม: M5")
        print(f" 💰 กำไรวันนี้: ${self.daily_profit:.2f} | ขาดทุนวันนี้: ${self.daily_loss:.2f}")
        print(f" 🔒 ขีดจำกัดเป้าหมาย: Target +${DAILY_PROFIT_TARGET:.2f} / Risk Max ${DAILY_LOSS_LIMIT:.2f}")
        print(sep)

    def start_engine(self) -> None:
        print("🚀 Price Action Trading Robot OOP Engine is running...")
        while True:
            self.loop_count += 1
            try:
                mt5.initialize() 
                self.check_and_reset_daily_limits()
                self.executor.manage_trailing_stop()
                self.audit_closed_positions()

                if self.daily_loss <= DAILY_LOSS_LIMIT or self.daily_profit >= DAILY_PROFIT_TARGET:
                    print("⏸️ [Trade] Daily limit reached, skipping new entries")
                    time.sleep(5); continue

                # 🧠 ระบบคำนวณเวลาถอยหลัง 10 วินาทีสุดท้าย และการปิดออเดอร์แบบ Scalping
                now = self._get_utc_time()
                minutes_to_next = 5 - (now.minute % 5)
                next_candle_time = (now + timedelta(minutes=minutes_to_next)).replace(second=0, microsecond=0)
                seconds_to_next = (next_candle_time - now).total_seconds()

                if not self.executor.has_open_position():
                    self.max_floating_pnl = 0.0

                if self.executor.has_open_position():
                    acc = mt5.account_info()
                    if acc:
                        floating_pnl = acc.equity - acc.balance
                        self.max_floating_pnl = max(self.max_floating_pnl, floating_pnl)
                        position = self.executor.get_active_positions()[0] if self.executor.get_active_positions() else None
                        position_side = None
                        if position:
                            position_side = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'

                        # 🟢 รอจนกำไรถึง $1.00 แล้วปิด
                        if floating_pnl >= 1.00:
                            self.executor.close_all_orders("TARGET_PROFIT_1USD")
                            print(f"💰 [Trade] ปิดกำไรที่เป้า $1.00: ${floating_pnl:.2f}")

                        # 🔴 ถ้ากำไรเคยสูงกว่าแล้วลดลงกว่า 0.20 USD จาก peak แต่ยังไม่ถึงเป้า ให้ปิดทันที
                        elif floating_pnl > 0 and self.max_floating_pnl >= 0.50 and \
                             self.max_floating_pnl - floating_pnl >= 0.20 and floating_pnl < 1.00:
                            self.executor.close_all_orders("EARLY_PROFIT_DRAWDOWN")
                            print(f"✂️ [Trade] กำไรเริ่มถอยจาก peak {self.max_floating_pnl:.2f} -> {floating_pnl:.2f}, ปิดก่อนลงต่อ")

                        # 🔴 ถ้าเวลากำลังจะต่อ candle ใหม่และยังขาดทุน ให้ปิด
                        elif seconds_to_next <= 10 and floating_pnl < 0:
                            self.executor.close_all_orders("LOSS_CUT_10SEC_BEFORE_CANDLE")
                            print(f"✂️ [Trade] หมดเวลายื้อ! ยอมตัดขาดทุนก่อนขึ้นแท่งใหม่: ${floating_pnl:.2f}")

                    time.sleep(1); continue 

                if not self._is_market_open():
                    print("⏸️ [Trade] Outside trading hours")
                    time.sleep(5); continue

                if not self.analyzer.is_spread_valid():
                    print("⏸️ [Trade] Spread too wide for entry")
                    time.sleep(5); continue

                # 🟢 ดึงราคากราฟสแกนสด
                df = self.analyzer.fetch_market_data(bars_count=60)
                current_candle_time = df['time'].iloc[-1]
                latest_candle = df.iloc[-1].to_dict()
                self.db.save_candle(SYMBOL, TIMEFRAME, latest_candle)
                history_df = self.db.load_recent_candles(SYMBOL, TIMEFRAME, limit=20)
                
                if history_df.empty or not self.db.db_ready:
                    history_df = df.copy()
                    print("🧠 [Trade] Database unavailable or empty, using live MT5 candle data for fallback")

                # ถ้าแท่งเทียนเปลี่ยนแท่งใหม่
                if current_candle_time != self.last_candle_time:
                    self.last_candle_time = current_candle_time
                    self.entry_ready_at = self._get_utc_time() + timedelta(seconds=30)
                    print(f"🆕 [Trade] แท่งเทียนใหม่มาแล้ว! บอทจะเริ่มสแกนสัญญาณในอีก 30 วินาที...")

                # 🧠 คัดกรองสัญญาณเข้าเทรดเฉพาะสัญญาณคุณภาพ (เพิ่ม Smart Guess จาก DB)
                if self.entry_ready_at and not self.executor.has_open_position() and self._get_utc_time() >= self.entry_ready_at:
                    chosen_signal = None
                    reason = ""
                    
                    # 1) ประเมินสัญญาณจากโซนอุปสงค์อุปทาน (Supply/Demand)
                    zone_signal = self.analyzer.analyze_zones_and_signals(df)
                    if zone_signal:
                        chosen_signal = zone_signal
                        reason = "PA zone signal"
                    else:
                        # 2) ประเมินสัญญาณจากเส้นค่าเฉลี่ยและ RSI (EMA/RSI Breakout)
                        sig, sig_reason = self.analyzer.analyze_candle_signal(history_df)
                        if sig:
                            chosen_signal = sig
                            reason = f"EMA/RSI signal: {sig_reason}"
                        else:
                            # 3) 🧠 Smart Guess จากข้อมูล Database 20 แท่งย้อนหลัง
                            try:
                                if not history_df.empty and len(history_df) >= 2:
                                    # ดึงราคาปิดทั้งหมดมาคำนวณหาค่าเฉลี่ย (เหมือนเส้น SMA 20)
                                    avg_close = history_df['close'].astype(float).mean()
                                    last_close = float(history_df['close'].iloc[-1])
                                    prev_close = float(history_df['close'].iloc[-2])
                                    
                                    # 🧮 คำนวณ RSI จากข้อมูล history_df
                                    history_closes = pd.to_numeric(history_df['close'], errors='coerce').astype(float)
                                    history_rsi = self.analyzer._rsi(history_closes, 14)
                                    history_rsi_value = float(history_rsi.iloc[-1]) if not history_rsi.empty else 50.0
                                    
                                    # เงื่อนไข BUY: ราคาปัจจุบันต้องยืน "เหนือ" ค่าเฉลี่ย 20 แท่ง AND แท่งล่าสุดมีแรงซื้อ AND RSI < 70 (ไม่ Overbought)
                                    if last_close > avg_close and last_close > prev_close and history_rsi_value < 70:
                                        chosen_signal = 'BUY'
                                        reason = f"DB Smart Guess: ยืนเหนือ SMA ({last_close:.2f} > {avg_close:.2f}) + RSI {history_rsi_value:.1f} < 70"
                                        
                                    # เงื่อนไข SELL: ราคาปัจจุบันต้องอยู่ "ใต้" ค่าเฉลี่ย 20 แท่ง AND แท่งล่าสุดมีแรงขาย AND RSI > 30 (ไม่ Oversold)
                                    elif last_close < avg_close and last_close < prev_close and history_rsi_value > 30:
                                        chosen_signal = 'SELL'
                                        reason = f"DB Smart Guess: หลุดใต้ SMA ({last_close:.2f} < {avg_close:.2f}) + RSI {history_rsi_value:.1f} > 30"
                                        
                                    else:
                                        # ถ้าราคาพันเจลากับเส้นค่าเฉลี่ย หรือ RSI อยู่ Overbought/Oversold บอทจะไม่เทรด
                                        chosen_signal = None
                                        reason = f"DB Data: กราฟแกว่งตัว หรือ RSI {history_rsi_value:.1f} ไม่ตรงเงื่อนไข (BUY<70, SELL>30)"
                                else:
                                    chosen_signal = None
                                    reason = "DB Data: ข้อมูลไม่เพียงพอสำหรับทำ Smart Guess"
                            except Exception as e:
                                chosen_signal = None
                                reason = f"DB Guess Error: {e}"

                    if chosen_signal:
                        candle_height = float(df['atr'].iloc[-1])
                        # 🧮 ใช้ history_rsi_value (จาก history_df) ให้ Consistent
                        # และคำนวณ ATR จริงๆ สำหรับบันทึก
                        closes = pd.to_numeric(df['close'], errors='coerce').astype(float)
                        
                        # ATR ที่ถูกต้อง = ค่าเฉลี่ยของ High-Low ย้อนหลัง 14 แท่ง
                        highs = pd.to_numeric(df['high'], errors='coerce').astype(float)
                        lows = pd.to_numeric(df['low'], errors='coerce').astype(float)
                        tr = np.maximum(highs - lows, np.maximum(abs(highs - closes.shift()), abs(lows - closes.shift())))
                        atr = tr.rolling(window=14).mean()
                        atr_value = float(atr.iloc[-1]) if not atr.empty else candle_height
                        
                        # ✅ ส่ง history_rsi_value (ค่าที่ตรงกับการตัดสินใจเข้าเทรด)
                        result = self.executor.execute_market_order(chosen_signal, candle_height, rsi_value=history_rsi_value, atr_value=atr_value)
                        ticket, entry_price, saved_rsi, saved_atr = result if result[0] is not None else (result[0], result[1], history_rsi_value, atr_value)
                        if ticket:
                            self.active_contexts[ticket] = {"rsi": saved_rsi if saved_rsi else history_rsi_value, "atr": saved_atr if saved_atr else atr_value, "score": 5, "htf": "PA_OOP_PA"}
                            print(f"✅ [Trade] Entry triggered: {chosen_signal} because {reason}")
                        else:
                            print(f"❌ [Trade] Failed to place order for {chosen_signal} (reason: {reason})")
                    else:
                        print(f"⏸️ [Trade] No high-quality entry signal found. Waiting... (Reason: {reason})")

                    # Reset entry window for next candle
                    self.entry_ready_at = None

                if self.loop_count % DASHBOARD_EVERY_N_LOOPS == 0:
                    self.print_dashboard()

            except Exception as e:
                print(f"❌ [Core Error] Crash detected: {e}"); time.sleep(RECONNECT_WAIT)
            
            # หน่วงรอบสั้นๆ 5 วินาที เพื่อเช็กแท่งราคาต่อเนื่องตลอดเวลา
            time.sleep(5)

# ==========================================
# 🏁 APPLICATION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    bot = PriceActionTradingBot()
    bot.start_engine()