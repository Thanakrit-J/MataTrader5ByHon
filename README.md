# MetaTrader 5 Trading Bot

This is an automated trading bot for MetaTrader 5 that implements an EMA crossover strategy with RSI confirmation and ATR volatility filtering.

## Features

- **Strategy**: EMA 10/20 crossover with RSI confirmation (<40 buy, >60 sell)
- **Volatility Filter**: ATR-based filtering for trending markets only
- **Risk Management**:
  - Position sizing: 1-2% risk per trade
  - Stop-loss and take-profit orders
  - Trailing stops
  - Max drawdown limit: 15%
  - Daily loss limit: 5%
  - Trade frequency controls (max 5 trades per day)

## Safety Features

- Demo mode only by default
- Connection validation
- Account balance checks
- Emergency stop functionality
- Comprehensive logging

## Requirements

- MetaTrader 5 terminal installed and running
- Python 3.7+
- Demo account in MT5

## Installation

1. **Install MetaTrader 5 Terminal**:
   - Download from official MetaTrader 5 website
   - Install and open the terminal
   - Create a demo account

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Bot**:
   - Open `trading_bot.py`
   - Ensure `DEMO_MODE = True` (default)
   - Adjust parameters if needed (symbol, risk settings, etc.)

## Usage

1. **Start MetaTrader 5 Terminal**:
   - Open MT5 terminal
   - Login to your demo account
   - Ensure the terminal is running

2. **Run the Bot**:
   ```bash
   python trading_bot.py
   ```

3. **Monitor Logs**:
   - Check `trading_bot.log` for activity
   - Monitor MT5 terminal for orders

## Configuration Parameters

- `SYMBOL`: Trading pair (default: "EURUSD")
- `TIMEFRAME`: Chart timeframe (default: M15)
- `RISK_PER_TRADE`: Risk per trade (default: 2%)
- `MAX_DRAWDOWN`: Max drawdown limit (default: 15%)
- `DAILY_LOSS_LIMIT`: Daily loss limit (default: 5%)
- `MAX_TRADES_PER_DAY`: Maximum trades per day (default: 5)

## Risk Management

The bot includes multiple layers of risk management:

1. **Position Sizing**: Automatically calculates lot size based on account balance and stop-loss distance
2. **Stop-Loss**: Every position has a stop-loss order
3. **Take-Profit**: Profit targets to lock in gains
4. **Trailing Stops**: Dynamic stop-loss that follows profitable trades
5. **Drawdown Control**: Pauses trading if account drawdown exceeds 15%
6. **Daily Loss Limit**: Stops trading if daily losses exceed 5%
7. **Trade Frequency**: Limits to 5 trades per day to prevent overtrading

## Testing

- Always test in demo mode first
- Monitor performance and logs
- Adjust parameters based on backtesting results
- Never switch to live trading without thorough testing

## Emergency Stop

- Press `Ctrl+C` to stop the bot gracefully
- The bot will close all open positions on emergency stop
- Check logs for any issues

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

## License

MIT License# MataTrader5ByHon
