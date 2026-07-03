# MetaTrader 5 Candle-Based Trading Bot

This repository contains a Python trading bot for MetaTrader 5 that trades `XAUUSD` on the `M5` timeframe. It is built for demo use only and combines live MT5 candle data with PostgreSQL candle history for signal evaluation, logging, and a fallback live-data mode.

## Overview

The bot is designed to trade on every new 5-minute candle using a layered decision process:

1. Detect a new M5 candle.
2. Wait `30 seconds` after the candle starts.
3. Prefer price-action zone signals first.
4. Fallback to EMA/RSI breakout signals.
5. If no signal exists, guess direction from the last two closes and still trade.
6. Manage open positions with stop-loss, take-profit, trailing stop, and early exit rules.

It also remains operational when PostgreSQL is unavailable by using live MT5 candle history directly.

## Main Files

- `trading_bot.py`
  - `DatabaseManager`: initializes PostgreSQL tables, saves candles, logs closed trades, and handles DB fallback.
  - `MarketAnalyzer`: fetches live MT5 candles, computes EMA9/21 and RSI14, and generates zone and candle breakout signals.
  - `TradeExecutionManager`: calculates lot size, sends market orders, adjusts trailing stops, and closes positions.
  - `PriceActionTradingBot`: executes the main loop, manages trading hours, daily limits, candle timing, and exit logic.
- `check_bot_status.py`: helper for verifying the bot environment and MT5 connectivity.
- `trade_history.py`: helper for reviewing recent trade history from PostgreSQL.
- `requirements.txt`: Python dependencies.
- `docker-compose.yml`: PostgreSQL service setup.

## Entry Logic

The bot enters a new trade after a fresh candle is detected and `30 seconds` have passed.

Signal priority:

1. `MarketAnalyzer.analyze_zones_and_signals()`
   - Detects demand/supply zone tests and reversal patterns.
2. `MarketAnalyzer.analyze_candle_signal()`
   - Uses EMA9/21 and RSI14 to confirm breakout momentum.
3. Fallback guess
   - Chooses `BUY` when the last close is above the previous close; otherwise `SELL`.

## Exit Logic

Open trade exit decisions include:

- Close at `1.00 USD` profit.
- Close early if profit peaked and then fell by `0.20 USD` before reaching the target.
- Close on strong reversal conditions.
- Close losing trades `10 seconds` before the next candle.

## Risk Management

- Demo-only enforcement: the bot shuts down if a live MT5 account is detected.
- Spread check: only enters trades when current spread is within `MAX_SPREAD_POINTS`.
- Daily limits:
  - Profit target: `4.0 USD`
  - Loss limit: `-3.0 USD`
- Position sizing uses `RISK_PER_TRADE` and available margin.

## Database Behavior

PostgreSQL is used for:

- `candles`: candle history for signal evaluation.
- `closed_trades`: logging results and context for closed trades.

If the database is unavailable, the bot will still run using live MT5 candle data and print a fallback warning.

## Configuration

Important settings in `trading_bot.py`:

- `SYMBOL = "XAUUSD"`
- `TIMEFRAME = mt5.TIMEFRAME_M5`
- `RISK_PER_TRADE = 0.03`
- `DAILY_PROFIT_TARGET = 4.0`
- `DAILY_LOSS_LIMIT = -3.0`
- `MAX_SPREAD_POINTS = 45`
- `MAGIC = 20260702`
- `TRADING_HOUR_START = 7`
- `TRADING_HOUR_END = 20`

## How to Run

1. Start PostgreSQL:
   ```powershell
   docker-compose up -d
   ```
2. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Open MetaTrader 5 and log into a demo account.
4. Run the bot:
   ```powershell
   python trading_bot.py
   ```

## Notes for Developers

- Main trading orchestration lives in `PriceActionTradingBot.start_engine()`.
- `execute_market_order()` returns both ticket and entry price.
- `self.max_floating_pnl` tracks the highest profit on an open trade.
- The bot resets peak profit tracking whenever no position is open.
- `DatabaseManager.save_candle()` uses `ON CONFLICT DO NOTHING` to avoid duplicate candle records.

## Safety Notes

- Use demo accounts only.
- Review logs and monitor the bot closely before using any live strategy.
- The bot is intended for experimentation and learning.

## Disclaimer

This project is for educational and experimental use only. Trading involves financial risk. Use it with caution and at your own responsibility.
