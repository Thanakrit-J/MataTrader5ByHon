---
description: "Use when: backtesting MT5 trading strategies, validating bot performance on historical data, testing EMA crossover with RSI signals"
name: "MT5 Backtest Prompt"
argument-hint: "Strategy parameters and date range..."
agent: "MT5 Trading Bot Agent"
---
Backtest the EMA crossover trading strategy with RSI confirmation on MetaTrader 5 historical data.

## Strategy Parameters
- EMA periods: 10 and 20
- RSI period: 14
- Buy signal: EMA10 crosses above EMA20 and RSI < 40
- Sell signal: EMA10 crosses below EMA20 and RSI > 60
- ATR period: 14 for volatility filtering (trade only if ATR > threshold)

## Backtest Requirements
- Use specified date range and symbol (e.g., EURUSD, H1 timeframe)
- Calculate key metrics: total return, win rate, max drawdown, Sharpe ratio
- Include transaction costs (spread + commission)
- Generate performance charts and trade log
- Validate risk management: stop-loss, take-profit, position sizing

## Output Format
Provide complete Python backtesting code using MetaTrader5 library, performance summary, and analysis of results.