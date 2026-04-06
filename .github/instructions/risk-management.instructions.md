---
description: "Use when: implementing trading bots, adding safety features, managing financial risks in automated trading, setting stop-loss and position sizing"
name: "Risk Management Instructions"
applyTo: "**/*.py"
---
# Risk Management Guidelines for Trading Bots

## Core Principles
- **Never risk more than 1-2% of account per trade**
- **Always use stop-loss orders on every position**
- **Implement maximum drawdown limits (e.g., 15% before pausing)**
- **Include daily loss limits to prevent emotional decisions**

## Required Features
- **Position Sizing**: Calculate lot size based on risk percentage and stop-loss distance
- **Stop-Loss**: Fixed or trailing stops to limit losses
- **Take-Profit**: Target profit levels to lock in gains
- **Drawdown Control**: Monitor account equity and pause trading if drawdown exceeds threshold
- **Trade Frequency Limits**: Prevent overtrading (max trades per day/hour)
- **Volatility Filtering**: Avoid trading in low volatility or high-risk periods

## Safety Checks
- Validate account balance before placing orders
- Check connection status to MT5 terminal
- Log all trades with timestamps and reasons
- Implement emergency stop functionality
- Test in demo mode before live trading

## Code Examples
```python
# Position sizing calculation
risk_amount = account_balance * 0.01  # 1% risk
stop_loss_pips = 50
lot_size = risk_amount / (stop_loss_pips * pip_value)

# Drawdown check
if (account_balance - peak_balance) / peak_balance < -0.15:
    pause_trading()
```