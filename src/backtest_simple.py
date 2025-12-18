"""
Simple backtesting simulation for XAUUSD AI Bot
Simulates trades based on model predictions and confidence
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json

print("=" * 70)
print("SIMPLE BACKTEST - XAUUSD AI BOT")
print("=" * 70)
print()

# Configuration
CONFIDENCE_THRESHOLD = 0.55  # 55% confidence
INITIAL_CAPITAL = 50.0       # $50 starting capital
RISK_PERCENT = 0.05          # 5% risk per trade
SPREAD_PIPS = 2.5            # Average spread
COMMISSION = 0.0             # No commission (prop firm)

print("⚙️ Backtest Configuration:")
print(f"   Initial Capital: ${INITIAL_CAPITAL}")
print(f"   Risk per Trade: {RISK_PERCENT*100:.1f}%")
print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"   Spread: {SPREAD_PIPS} pips")
print()

# Load data
print("📥 Loading test data...")
df = pd.read_csv('data/processed/xauusd_labeled.csv')
df['time'] = pd.to_datetime(df['time'])

# Load model and features
with open('python_training/models/feature_list.json', 'r') as f:
    feature_cols = json.load(f)

model = lgb.Booster(model_file='python_training/models/lightgbm_xauusd_v1.txt')

# Use test set (last 20%)
split_idx = int(0.8 * len(df))
df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

print(f"   Test period: {df_test['time'].min().date()} → {df_test['time'].max().date()}")
print(f"   Test bars: {len(df_test):,}")
print()

# Get predictions
print("🔮 Running model predictions...")
X_test = df_test[feature_cols].values
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
max_proba = np.max(y_pred_proba, axis=1)

df_test['pred_class'] = y_pred
df_test['pred_confidence'] = max_proba

print(f"   ✓ {len(df_test):,} predictions generated")
print()

# Filter by confidence
df_trades = df_test[df_test['pred_confidence'] >= CONFIDENCE_THRESHOLD].copy()

print(f"📊 High-Confidence Signals:")
print(f"   Total: {len(df_trades):,} ({len(df_trades)/len(df_test)*100:.1f}% of test set)")
print()

# Simulate trades
print("💰 Simulating trades...")

trades = []
equity = INITIAL_CAPITAL
equity_curve = [INITIAL_CAPITAL]
peak_equity = INITIAL_CAPITAL

for idx, row in df_trades.iterrows():
    signal = row['pred_class']  # 0=SHORT, 1=HOLD, 2=LONG
    confidence = row['pred_confidence']
    forward_return = row['forward_return_15m']
    
    # Skip HOLD signals
    if signal == 1:
        equity_curve.append(equity)
        continue
    
    # Simulate trade outcome
    # Use forward_return as proxy for trade result
    
    if signal == 2:  # LONG
        # Deduct spread
        net_return = forward_return - (SPREAD_PIPS / 20000)  # Approx spread impact
        trade_result = 'WIN' if net_return > 0 else 'LOSS'
        
    elif signal == 0:  # SHORT
        # Inverse return for short
        net_return = -forward_return - (SPREAD_PIPS / 20000)
        trade_result = 'WIN' if net_return > 0 else 'LOSS'
    
    # Fixed lot sizing: $2.50 per trade
    pnl = 2.50 * (1 if trade_result == 'WIN' else -1)
    
    # Update equity
    equity += pnl
    equity_curve.append(equity)
    
    # Track peak for drawdown
    if equity > peak_equity:
        peak_equity = equity
    
    # Record trade
    trades.append({
        'time': row['time'],
        'signal': 'LONG' if signal == 2 else 'SHORT',
        'confidence': confidence,
        'pnl': pnl,
        'result': trade_result,
        'equity': equity
    })

trades_df = pd.DataFrame(trades)

print(f"   ✓ {len(trades_df):,} trades executed")
print()

# Calculate metrics
print("=" * 70)
print("📈 BACKTEST RESULTS")
print("=" * 70)
print()

# Basic stats
total_trades = len(trades_df)
winning_trades = (trades_df['pnl'] > 0).sum()
losing_trades = (trades_df['pnl'] <= 0).sum()
win_rate = winning_trades / total_trades if total_trades > 0 else 0

gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
net_profit = trades_df['pnl'].sum()
profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

# Drawdown
equity_series = pd.Series(equity_curve)
running_max = equity_series.cummax()
drawdown = (equity_series - running_max) / running_max
max_drawdown = drawdown.min()

print(f"💼 Trading Performance:")
print(f"   Total Trades: {total_trades}")
print(f"   Winning Trades: {winning_trades} ({win_rate*100:.1f}%)")
print(f"   Losing Trades: {losing_trades} ({(1-win_rate)*100:.1f}%)")
print()

print(f"💰 Financial Results:")
print(f"   Starting Capital: ${INITIAL_CAPITAL:.2f}")
print(f"   Ending Capital: ${equity:.2f}")
print(f"   Net Profit: ${net_profit:.2f} ({(net_profit/INITIAL_CAPITAL)*100:.1f}%)")
print(f"   Gross Profit: ${gross_profit:.2f}")
print(f"   Gross Loss: ${gross_loss:.2f}")
print(f"   Profit Factor: {profit_factor:.2f}")
print()

print(f"📊 Risk Metrics:")
print(f"   Max Drawdown: {max_drawdown*100:.1f}%")
print(f"   Peak Equity: ${peak_equity:.2f}")
print()

# Per-signal performance
print(f"📊 Performance by Signal:")
for signal_type in ['LONG', 'SHORT']:
    signal_trades = trades_df[trades_df['signal'] == signal_type]
    if len(signal_trades) > 0:
        sig_wins = (signal_trades['pnl'] > 0).sum()
        sig_wr = sig_wins / len(signal_trades)
        sig_pnl = signal_trades['pnl'].sum()
        print(f"   {signal_type:5s}: {len(signal_trades):3d} trades, WR: {sig_wr*100:.1f}%, P&L: ${sig_pnl:+.2f}")
print()

# Daily performance
trades_df['date'] = trades_df['time'].dt.date
daily_pnl = trades_df.groupby('date')['pnl'].sum()

print(f"📅 Time Analysis:")
print(f"   Trading Days: {len(daily_pnl)}")
print(f"   Avg Trades/Day: {total_trades / len(daily_pnl):.1f}")
print(f"   Best Day: ${daily_pnl.max():.2f}")
print(f"   Worst Day: ${daily_pnl.min():.2f}")
print()

print("=" * 70)
print("✅ BACKTEST COMPLETE!")
print("=" * 70)
print()

print("🎯 Summary:")
print(f"   Win Rate: {win_rate*100:.1f}%")
print(f"   Profit Factor: {profit_factor:.2f}")
print(f"   Return: {(net_profit/INITIAL_CAPITAL)*100:.1f}%")
print(f"   Max DD: {max_drawdown*100:.1f}%")
print()

# Save results
results = {
    'config': {
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'initial_capital': INITIAL_CAPITAL,
        'risk_percent': RISK_PERCENT
    },
    'performance': {
        'total_trades': int(total_trades),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'net_profit': float(net_profit),
        'return_percent': float((net_profit/INITIAL_CAPITAL)*100),
        'max_drawdown': float(max_drawdown*100)
    }
}

with open('python_training/models/backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("💾 Results saved to: python_training/models/backtest_results.json")