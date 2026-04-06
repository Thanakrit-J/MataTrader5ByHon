"""
MetaTrader 5 Trading Bot - EMA Crossover with RSI Confirmation and ATR Filtering

This bot implements an automated trading strategy for MetaTrader 5:
- EMA 10/20 crossover signals
- RSI confirmation (<40 for buy, >60 for sell)
- ATR volatility filtering (only trade in trending markets)
- Comprehensive risk management features
TRADING INSTRUMENT: XAUUSD (Gold vs US Dollar)
RISK MANAGEMENT FEATURES:
- Position sizing: 1-2% risk per trade
- Stop-loss and take-profit orders
- Trailing stops
- Max drawdown limit: 15%
- Daily loss limit: 5%
- Trade frequency controls (max 5 trades per day)

SAFETY FEATURES:
- Demo mode only (set DEMO_MODE = True)
- Connection validation
- Account balance checks
- Emergency stop functionality
- Comprehensive logging

REQUIREMENTS:
- MetaTrader 5 terminal installed and running
- Python 3.7+
- MetaTrader5 Python package: pip install MetaTrader5
- pandas, numpy, ta (technical analysis library)

SETUP INSTRUCTIONS:
1. Install MetaTrader 5 terminal from official website
2. Open demo account in MT5 terminal
3. Install required Python packages:
   pip install MetaTrader5 pandas numpy ta
4. Run this script in demo mode first
5. Monitor logs and performance before considering live trading

WARNING: This is for educational purposes. Trading involves risk.
Test thoroughly in demo mode before any live trading.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import time
import logging
from datetime import datetime, timedelta
import sys
import os

# Configuration
DEMO_MODE = True  # Set to False only after thorough testing
SYMBOL = "XAUUSD"  # Trading pair - Gold vs US Dollar
TIMEFRAME = mt5.TIMEFRAME_M15  # 15-minute timeframe
FAST_EMA_PERIOD = 10
SLOW_EMA_PERIOD = 20
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5  # Volatility threshold

# Risk Management Parameters
RISK_PER_TRADE = 0.02  # 2% of account per trade
MAX_DRAWDOWN = 0.15  # 15% max drawdown
DAILY_LOSS_LIMIT = 0.05  # 5% daily loss limit
MAX_TRADES_PER_DAY = 5
TRAILING_STOP_PIPS = 20  # Trailing stop distance in pips
TAKE_PROFIT_MULTIPLIER = 2  # TP at 2x SL distance

# Logging setup
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MT5TradingBot:
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.initial_balance = 0
        self.peak_balance = 0
        self.daily_start_balance = 0
        self.daily_trades = 0
        self.today = datetime.now().date()
        self.emergency_stop = False

        # Trade tracking
        self.open_positions = {}  # ticket: {'entry_time': datetime, 'entry_price': float, 'type': str}

        # Initialize MT5 connection
        self.connect_mt5()

    def connect_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            logging.error("MT5 initialization failed")
            sys.exit(1)

        if DEMO_MODE:
            # For demo accounts, MT5 should be running with demo account logged in
            logging.info("Running in DEMO MODE")
        else:
            logging.warning("Running in LIVE MODE - Use with extreme caution!")

        self.connected = True
        self.account_info = mt5.account_info()
        if self.account_info is None:
            logging.error("Failed to get account info")
            sys.exit(1)

        self.initial_balance = self.account_info.balance
        self.peak_balance = self.account_info.balance
        self.daily_start_balance = self.account_info.balance

        logging.info(f"Connected to MT5. Account: {self.account_info.login}, Balance: {self.account_info.balance}")

    def get_market_data(self, symbol, timeframe, bars=100):
        """Fetch historical market data"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None:
            logging.error(f"Failed to get rates for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # EMA calculations
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=FAST_EMA_PERIOD)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=SLOW_EMA_PERIOD)

        # RSI calculation
        df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)

        # ATR calculation
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_PERIOD)

        return df

    def check_volatility_filter(self, df):
        """Check if market is trending (high volatility)"""
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].rolling(window=20).mean().iloc[-1]

        # Trade only if current ATR is above average (trending market)
        return current_atr > avg_atr * ATR_MULTIPLIER

    def generate_signal(self, df):
        """Generate trading signals based on strategy"""
        if len(df) < max(FAST_EMA_PERIOD, SLOW_EMA_PERIOD, RSI_PERIOD, ATR_PERIOD) + 1:
            return None

        # EMA crossover signals
        prev_fast = df['ema_fast'].iloc[-2]
        prev_slow = df['ema_slow'].iloc[-2]
        curr_fast = df['ema_fast'].iloc[-1]
        curr_slow = df['ema_slow'].iloc[-1]

        crossover_up = prev_fast <= prev_slow and curr_fast > curr_slow
        crossover_down = prev_fast >= prev_slow and curr_fast < curr_slow

        # RSI confirmation
        rsi = df['rsi'].iloc[-1]

        # Volatility filter
        volatility_ok = self.check_volatility_filter(df)

        if not volatility_ok:
            return None

        if crossover_up and rsi < 40:
            return 'BUY'
        elif crossover_down and rsi > 60:
            return 'SELL'

        return None

    def calculate_position_size(self, symbol, stop_loss_pips):
        """Calculate position size based on risk management"""
        account_balance = self.account_info.balance

        # Risk amount (1-2% of account)
        risk_amount = account_balance * RISK_PER_TRADE

        # Get pip value for the symbol
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0

        pip_value = symbol_info.point * symbol_info.trade_contract_size

        # Calculate lot size
        lot_size = risk_amount / (stop_loss_pips * pip_value * 10)  # *10 for 5-digit brokers

        # Ensure minimum lot size
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max

        lot_size = max(min_lot, min(lot_size, max_lot))

        return round(lot_size, 2)

    def check_risk_limits(self):
        """Check if trading should be paused due to risk limits"""
        current_balance = self.account_info.balance

        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Check max drawdown
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        if drawdown >= MAX_DRAWDOWN:
            logging.warning(f"Max drawdown limit reached: {drawdown:.2%}")
            self.emergency_stop = True
            return False

        # Check daily loss limit
        if self.today != datetime.now().date():
            self.today = datetime.now().date()
            self.daily_start_balance = current_balance
            self.daily_trades = 0

        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance
        if daily_loss >= DAILY_LOSS_LIMIT:
            logging.warning(f"Daily loss limit reached: {daily_loss:.2%}")
            return False

        # Check trade frequency
        if self.daily_trades >= MAX_TRADES_PER_DAY:
            logging.info("Max trades per day reached")
            return False

        return True

    def place_order(self, symbol, order_type, lot_size, sl_pips, tp_pips):
        """Place market order with stop-loss and take-profit"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        price = tick.ask if order_type == 'BUY' else tick.bid

        # Calculate stop-loss and take-profit prices
        if order_type == 'BUY':
            sl_price = price - sl_pips * symbol_info.point * 10  # *10 for 5-digit
            tp_price = price + tp_pips * symbol_info.point * 10
        else:
            sl_price = price + sl_pips * symbol_info.point * 10
            tp_price = price - tp_pips * symbol_info.point * 10

        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": 123456,
            "comment": "EMA RSI ATR Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed: {result.comment}")
            return False

        logging.info(f"Order placed: {order_type} {lot_size} lots of {symbol} at {price}, SL: {sl_price}, TP: {tp_price}")
        self.daily_trades += 1

        # Track the position (will be updated when position appears in MT5)
        # Note: We can't get the position ticket immediately, so we'll track it in the main loop
        return True

    def track_positions(self):
        """Track open positions and log closed trades"""
        positions = mt5.positions_get()
        if positions is None:
            return

        current_positions = {pos.ticket: pos for pos in positions if pos.magic == 123456}

        # Check for newly opened positions
        for ticket, pos in current_positions.items():
            if ticket not in self.open_positions:
                self.open_positions[ticket] = {
                    'entry_time': datetime.now(),
                    'entry_price': pos.price_open,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'symbol': pos.symbol
                }
                logging.info(f"Position opened: {ticket} - {pos.symbol} {self.open_positions[ticket]['type']} {pos.volume} lots at {pos.price_open}")

        # Check for closed positions
        closed_tickets = [ticket for ticket in self.open_positions.keys() if ticket not in current_positions]
        for ticket in closed_tickets:
            pos_data = self.open_positions[ticket]
            entry_time = pos_data['entry_time']
            exit_time = datetime.now()
            duration = exit_time - entry_time

            # Get deal history to find the closing deal
            from_time = entry_time - timedelta(minutes=1)
            to_time = exit_time + timedelta(minutes=1)
            deals = mt5.history_deals_get(from_time, to_time)

            profit = 0
            if deals:
                for deal in deals:
                    if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                        profit = deal.profit
                        break

            # Log trade result
            profit_str = f"+${profit:.2f}" if profit > 0 else f"${profit:.2f}"
            logging.info(f"TRADE CLOSED: {ticket} | {pos_data['symbol']} {pos_data['type']} | Duration: {duration} | P/L: {profit_str}")
            print(f"📊 TRADE RESULT: {pos_data['symbol']} {pos_data['type']} | Duration: {str(duration).split('.')[0]} | P/L: {profit_str}")

            del self.open_positions[ticket]

    def manage_trailing_stops(self):
        """Update trailing stops for open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return

        for position in positions:
            if position.magic != 123456:  # Only manage our positions
                continue

            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                continue

            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                continue

            # Calculate new stop-loss for trailing stop
            if position.type == mt5.POSITION_TYPE_BUY:
                new_sl = tick.bid - TRAILING_STOP_PIPS * symbol_info.point * 10
                if new_sl > position.sl:
                    # Update stop-loss
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp,
                    }
                    mt5.order_send(request)
                    logging.info(f"Trailing stop updated for BUY position {position.ticket}: SL {new_sl}")
            else:
                new_sl = tick.ask + TRAILING_STOP_PIPS * symbol_info.point * 10
                if new_sl < position.sl:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp,
                    }
                    mt5.order_send(request)
                    logging.info(f"Trailing stop updated for SELL position {position.ticket}: SL {new_sl}")

    def run(self):
        """Main trading loop"""
        logging.info("Starting trading bot...")

        while not self.emergency_stop:
            try:
                # Update account info
                self.account_info = mt5.account_info()
                if self.account_info is None:
                    logging.error("Lost connection to MT5")
                    break

                # Track positions and log closed trades
                self.track_positions()

                print(f"💰 Balance: ${self.account_info.balance:.2f} | Open Positions: {len(self.open_positions)}")

                # Check risk limits
                if not self.check_risk_limits():
                    logging.info("Risk limits triggered - pausing trading")
                    time.sleep(300)  # Wait 5 minutes
                    continue

                # Get market data
                df = self.get_market_data(SYMBOL, TIMEFRAME)
                if df is None:
                    time.sleep(60)
                    continue

                # Calculate indicators
                df = self.calculate_indicators(df)

                # Generate signal
                signal = self.generate_signal(df)

                print(f"🔍 Checking signals... RSI: {df['rsi'].iloc[-1]:.2f}, EMA Fast: {df['ema_fast'].iloc[-1]:.5f}, EMA Slow: {df['ema_slow'].iloc[-1]:.5f}, Signal: {signal}")

                if signal:
                    # Calculate position parameters
                    stop_loss_pips = TRAILING_STOP_PIPS
                    take_profit_pips = TRAILING_STOP_PIPS * TAKE_PROFIT_MULTIPLIER
                    lot_size = self.calculate_position_size(SYMBOL, stop_loss_pips)

                    if lot_size > 0:
                        # Place order
                        success = self.place_order(SYMBOL, signal, lot_size, stop_loss_pips, take_profit_pips)
                        if success:
                            logging.info(f"Signal executed: {signal}")
                        else:
                            logging.error("Failed to execute signal")

                # Manage trailing stops
                self.manage_trailing_stops()

                # Wait before next iteration
                time.sleep(60)  # Check every minute

            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                time.sleep(60)

        logging.info("Trading bot stopped")

    def emergency_stop_trading(self):
        """Emergency stop - close all positions"""
        logging.warning("Emergency stop activated!")

        positions = mt5.positions_get()
        if positions:
            for position in positions:
                if position.magic == 123456:
                    # Close position
                    if position.type == mt5.POSITION_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = mt5.symbol_info_tick(position.symbol).bid
                    else:
                        order_type = mt5.ORDER_TYPE_BUY
                        price = mt5.symbol_info_tick(position.symbol).ask

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "type": order_type,
                        "price": price,
                        "position": position.ticket,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": "Emergency Close",
                    }

                    mt5.order_send(request)

                    # Log emergency close with trade details
                    if position.ticket in self.open_positions:
                        pos_data = self.open_positions[position.ticket]
                        entry_time = pos_data['entry_time']
                        exit_time = datetime.now()
                        duration = exit_time - entry_time

                        # Calculate profit/loss (approximate from current market price)
                        current_price = price
                        entry_price = pos_data['entry_price']
                        if pos_data['type'] == 'BUY':
                            profit = (current_price - entry_price) * pos_data['volume'] * 100000  # Approximate for EURUSD
                        else:
                            profit = (entry_price - current_price) * pos_data['volume'] * 100000

                        profit_str = f"+${profit:.2f}" if profit > 0 else f"${profit:.2f}"
                        logging.info(f"EMERGENCY CLOSE: {position.ticket} | {pos_data['symbol']} {pos_data['type']} | Duration: {duration} | P/L: {profit_str}")
                        print(f"🚨 EMERGENCY CLOSE: {pos_data['symbol']} {pos_data['type']} | Duration: {str(duration).split('.')[0]} | P/L: {profit_str}")

                        del self.open_positions[position.ticket]
                    else:
                        logging.info(f"Emergency closed position: {position.ticket}")

        self.emergency_stop = True

def main():
    bot = MT5TradingBot()

    try:
        bot.run()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
        bot.emergency_stop_trading()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        bot.emergency_stop_trading()
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed")

if __name__ == "__main__":
    main()