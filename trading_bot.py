import MetaTrader5 as mt5
import pandas as pd
import ta
import time
import pytz
from datetime import datetime, timedelta
from collections import deque

# ===== CONFIG =====
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
SERVER_TIMEZONE = pytz.timezone("Etc/UTC")
LOCAL_TIMEZONE  = pytz.timezone("Asia/Bangkok")

RISK_PER_TRADE         = 0.005
MAX_TRADES_PER_DAY     = 3
TRAILING_STOP_POINTS   = 300
TAKE_PROFIT_MULTIPLIER = 2.5
SL_ATR_MULTIPLIER      = 1.5

FAST_EMA   = 10
SLOW_EMA   = 20
RSI_PERIOD = 14
ATR_PERIOD = 14

SCORE_THRESHOLD   = 4
MAX_SPREAD_POINTS = 50        # ✅ แก้: 30 → 50 (รองรับ spread ช่วง pre-London)

MAGIC              = 20250101
TRADING_HOUR_START = 13       # ✅ แก้: 8 → 13 UTC (London open — spread แคบจริง)
TRADING_HOUR_END   = 22

RECONNECT_WAIT = 5
MAX_RECONNECT  = 10


# ===================================================================
#  BOT
# ===================================================================
class AIBot:

    def __init__(self):
        self._connect()

        self.daily_trades     = 0
        self.today            = self._utc_now().date()
        self.processed_deals: set  = set()
        self.trade_history: list   = []
        self.recent_losses: deque  = deque(maxlen=10)

    # ------------------------------------------------------------------
    # CONNECTION
    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        for attempt in range(1, MAX_RECONNECT + 1):
            if mt5.initialize():
                print("✅ MT5 connected")
                return True
            print(f"⚠️  MT5 init failed (attempt {attempt}/{MAX_RECONNECT})")
            time.sleep(RECONNECT_WAIT)
        raise RuntimeError("❌ Cannot connect to MT5 after max retries")

    def _ensure_connected(self) -> bool:
        if mt5.terminal_info() is None:
            print("🔄 MT5 disconnected — reconnecting …")
            return self._connect()
        return True

    # ------------------------------------------------------------------
    # TIME HELPERS
    # ------------------------------------------------------------------
    def _utc_now(self) -> datetime:
        return datetime.now(tz=pytz.utc)

    def _is_trading_hours(self) -> bool:
        h = self._utc_now().hour
        return TRADING_HOUR_START <= h < TRADING_HOUR_END

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------
    def get_data(self, timeframe, bars: int = 100) -> pd.DataFrame:
        self._ensure_connected()
        rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {SYMBOL}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], FAST_EMA)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], SLOW_EMA)
        df['rsi']      = ta.momentum.rsi(df['close'], RSI_PERIOD)
        df['atr']      = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], ATR_PERIOD
        )
        return df

    # ------------------------------------------------------------------
    # HIGHER TIMEFRAME TREND
    # ------------------------------------------------------------------
    def higher_trend(self) -> str:
        df = self.get_data(mt5.TIMEFRAME_H1)
        df = self.add_indicators(df)
        return 'up' if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] else 'down'

    # ------------------------------------------------------------------
    # SCORE  (symmetric — max +6 / min -6)
    # ------------------------------------------------------------------
    def calculate_score(self, df: pd.DataFrame) -> int:
        score = 0

        price    = df['close'].iloc[-1]
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        rsi      = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-5]

        high = df['high'].tail(20).max()
        low  = df['low'].tail(20).min()
        rng  = high - low
        pos  = (price - low) / rng if rng > 0 else 0.5

        trend_strength = abs(ema_fast - ema_slow) / price

        # 1. EMA cross (±2)
        score += 2 if ema_fast > ema_slow else -2

        # 2. RSI level (±1)
        if rsi > 55:
            score += 1
        elif rsi < 45:
            score -= 1

        # 3. RSI momentum (±1)
        score += 1 if rsi > rsi_prev else -1

        # 4. Price position in range (±1)
        if pos > 0.7:
            score += 1
        elif pos < 0.3:
            score -= 1

        # 5. Trend strength (±1)
        score += 1 if trend_strength > 0.001 else -1

        return score

    # ------------------------------------------------------------------
    # POSITION HELPERS
    # ------------------------------------------------------------------
    def open_positions(self) -> list:
        self._ensure_connected()
        pos = mt5.positions_get(symbol=SYMBOL)
        return list(pos) if pos else []

    def has_open_position(self) -> bool:
        return len(self.open_positions()) > 0

    # ------------------------------------------------------------------
    # TRAILING STOP
    # ------------------------------------------------------------------
    def update_trailing_stops(self):
        for pos in self.open_positions():
            tick        = mt5.symbol_info_tick(SYMBOL)
            symbol_info = mt5.symbol_info(SYMBOL)
            point       = symbol_info.point
            trail_dist  = TRAILING_STOP_POINTS * point

            if pos.type == mt5.ORDER_TYPE_BUY:
                new_sl = tick.bid - trail_dist
                if new_sl > pos.sl + point:
                    self._modify_sl(pos.ticket, new_sl)

            elif pos.type == mt5.ORDER_TYPE_SELL:
                new_sl = tick.ask + trail_dist
                if new_sl < pos.sl - point:
                    self._modify_sl(pos.ticket, new_sl)

    def _modify_sl(self, ticket: int, new_sl: float):
        result = mt5.order_send({
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       new_sl,
        })
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"   🔧 Trail SL updated → {new_sl:.2f}")
        else:
            print(f"   ⚠️  Trail SL failed: {result.comment}")

    # ------------------------------------------------------------------
    # TRACK CLOSED TRADES
    # ------------------------------------------------------------------
    def track_closed_trades(self):
        self._ensure_connected()
        deals = mt5.history_deals_get(
            self._utc_now().replace(tzinfo=None) - timedelta(hours=2),
            self._utc_now().replace(tzinfo=None)
        )
        if deals is None:
            return

        for d in deals:
            if d.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if d.ticket in self.processed_deals:
                continue
            if d.magic != MAGIC:
                continue

            self.processed_deals.add(d.ticket)
            record = {
                "ticket": d.ticket,
                "profit": d.profit,
                "type":   "BUY" if d.type == mt5.ORDER_TYPE_BUY else "SELL",
                "time":   self._utc_now(),
            }
            self.trade_history.append(record)

            if d.profit < 0:
                print(f"   📉 Loss recorded (ticket={d.ticket}, P/L={d.profit:.2f})")
                self.recent_losses.append(record)

    # ------------------------------------------------------------------
    # LEARNING FILTER
    # ------------------------------------------------------------------
    def loss_filter(self, direction: str) -> bool:
        recent = list(self.recent_losses)[-5:]
        if len(recent) < 3:
            return False
        same_dir_losses = sum(1 for r in recent if r["type"] == direction)
        if same_dir_losses >= 3:
            print(f"   🧠 Learning filter: {same_dir_losses}/5 recent losses are {direction} → skip")
            return True
        return False

    # ------------------------------------------------------------------
    # LOT SIZE
    # ------------------------------------------------------------------
    def lot_size(self, sl_points: float) -> float:
        self._ensure_connected()
        acc         = mt5.account_info()
        symbol_info = mt5.symbol_info(SYMBOL)

        risk_money        = acc.balance * RISK_PER_TRADE
        pip_value_per_lot = symbol_info.trade_tick_value / symbol_info.trade_tick_size
        sl_money_per_lot  = sl_points * pip_value_per_lot

        if sl_money_per_lot == 0:
            return symbol_info.volume_min

        lot  = risk_money / sl_money_per_lot
        lot  = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
        step = symbol_info.volume_step
        lot  = round(round(lot / step) * step, 2)
        return lot

    # ------------------------------------------------------------------
    # ANALYZE
    # ✅ แก้: ย้าย trading hours check ขึ้นมาเป็นอันแรกใน analyze()
    #         เพื่อไม่ให้ log "Spread too wide" รัวช่วงนอกเวลา
    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame) -> str | None:
        if len(df) < 50:
            return None

        df = self.add_indicators(df)

        # ✅ 1. Trading hours — เช็คก่อนทุกอย่าง
        if not self._is_trading_hours():
            return None

        # 2. Spread filter
        tick        = mt5.symbol_info_tick(SYMBOL)
        symbol_info = mt5.symbol_info(SYMBOL)
        spread      = (tick.ask - tick.bid) / symbol_info.point
        if spread > MAX_SPREAD_POINTS:
            print(f"   ⛔ Spread too wide: {spread:.1f} pts")
            return None

        # 3. Volatility / sideways filter
        atr_now  = df['atr'].iloc[-1]
        atr_mean = df['atr'].rolling(20).mean().iloc[-1]
        if atr_now < atr_mean:
            print(f"   💤 Low volatility (ATR {atr_now:.2f} < mean {atr_mean:.2f})")
            return None

        score = self.calculate_score(df)
        print(f"   🧠 Score: {score:+d}")

        if score >= SCORE_THRESHOLD:
            decision = 'BUY'
        elif score <= -SCORE_THRESHOLD:
            decision = 'SELL'
        else:
            return None

        # 4. Higher TF confirmation
        ht = self.higher_trend()
        if decision == 'BUY' and ht != 'up':
            print("   ❌ HTF not aligned (need up, got down)")
            return None
        if decision == 'SELL' and ht != 'down':
            print("   ❌ HTF not aligned (need down, got up)")
            return None

        # 5. Learning filter
        if self.loss_filter(decision):
            return None

        return decision

    # ------------------------------------------------------------------
    # PLACE TRADE
    # ------------------------------------------------------------------
    def place_trade(self, signal: str, df: pd.DataFrame):
        self._ensure_connected()
        symbol_info = mt5.symbol_info(SYMBOL)
        tick        = mt5.symbol_info_tick(SYMBOL)
        point       = symbol_info.point

        price   = tick.ask if signal == 'BUY' else tick.bid
        atr     = df['atr'].iloc[-1]
        sl_dist = atr * SL_ATR_MULTIPLIER
        sl_pts  = sl_dist / point
        tp_dist = sl_dist * TAKE_PROFIT_MULTIPLIER

        sl_price = price - sl_dist if signal == 'BUY' else price + sl_dist
        tp_price = price + tp_dist if signal == 'BUY' else price - tp_dist

        lot = self.lot_size(sl_pts)

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    SYMBOL,
            "volume":    lot,
            "type":      mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price":     price,
            "sl":        round(sl_price, symbol_info.digits),
            "tp":        round(tp_price, symbol_info.digits),
            "deviation": 10,
            "magic":     MAGIC,
            "comment":   "AI_SCORING_BOT",
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"   ✅ {signal} | lot={lot} | SL={sl_price:.2f} | TP={tp_price:.2f}")
            self.daily_trades += 1
        else:
            print(f"   ❌ Trade failed: [{result.retcode}] {result.comment}")

    # ------------------------------------------------------------------
    # DAILY RESET
    # ------------------------------------------------------------------
    def _daily_reset(self):
        now_date = self._utc_now().date()
        if now_date != self.today:
            self.today        = now_date
            self.daily_trades = 0
            print("📅 New day — trade counter reset")

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    # def run(self):
    #     print("🚀 AI BOT STARTED")

    #     while True:
    #         now = self._utc_now()
    #         print(f"\n⏰ {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    #         try:
    #             self._ensure_connected()
    #             self._daily_reset()
    #             self.update_trailing_stops()
    #             self.track_closed_trades()

    #             print(f"   📊 Daily trades: {self.daily_trades}/{MAX_TRADES_PER_DAY}")

    #             if self.daily_trades >= MAX_TRADES_PER_DAY:
    #                 print("   ⛔ Max daily trades reached")
    #                 time.sleep(60)
    #                 continue

    #             if not self._is_trading_hours():
    #                 print("   🌙 Outside trading hours")
    #                 time.sleep(60)
    #                 continue

    #             if self.has_open_position():
    #                 print("   ⏳ Position already open — waiting")
    #                 time.sleep(30)
    #                 continue

    #             df     = self.get_data(TIMEFRAME)
    #             signal = self.analyze(df)
    #             print(f"   🔍 Signal: {signal}")

    #             if signal:
    #                 self.place_trade(signal, df)

    #         except Exception as e:
    #             print(f"   ❌ ERROR: {e}")
    #             time.sleep(RECONNECT_WAIT)
    #             self._connect()

    #         time.sleep(60)

    def run(self):
        print("🚀 AI BOT STARTED")

        while True:
            local_now = self._utc_now().astimezone(LOCAL_TIMEZONE)   # ✅ แปลงแค่ตอนแสดงผล
            print(f"\n⏰ {local_now.strftime('%Y-%m-%d %H:%M:%S ICT')}")

            try:
                self._ensure_connected()
                self._daily_reset()
                self.update_trailing_stops()
                self.track_closed_trades()

                print(f"   📊 Daily trades: {self.daily_trades}/{MAX_TRADES_PER_DAY}")

                if self.daily_trades >= MAX_TRADES_PER_DAY:
                    print("   ⛔ Max daily trades reached")
                    time.sleep(60)
                    continue

                if not self._is_trading_hours():
                    print("   🌙 Outside trading hours")
                    time.sleep(60)
                    continue

                if self.has_open_position():
                    print("   ⏳ Position already open — waiting")
                    time.sleep(30)
                    continue

                df     = self.get_data(TIMEFRAME)
                df     = self.add_indicators(df)          # ✅ add indicators ก่อนส่งต่อ
                signal = self.analyze(df)
                print(f"   🔍 Signal: {signal}")

                if signal:
                    self.place_trade(signal, df)

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                time.sleep(RECONNECT_WAIT)
                self._connect()

            time.sleep(60)


# ===== ENTRY POINT =====
if __name__ == "__main__":
    bot = AIBot()
    bot.run()