"""
Simple backtest on M1 test data.
"""
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path


def main():
    print("=" * 70)
    print("M1 Backtest Evaluation")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_parquet(project_root / 'data' / 'processed' / 'features_m1_test.parquet')
    print(f"  Test: {len(test_df):,} M1 bars")
    
    # Load model
    print("\nLoading model...")
    model = lgb.Booster(model_file=str(project_root / 'python_training' / 'models' / 'lightgbm_xauusd.pkl'))
    
    # Prepare features
    feature_cols = [c for c in test_df.columns if c not in ['time', 'label']]
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Predictions
    print("\nMaking predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    max_proba = np.max(y_pred_proba, axis=1)
    
    # Simple backtest
    print("\nBacktest (confidence >= 0.60)...")
    
    confidence_threshold = 0.60
    high_conf_mask = max_proba >= confidence_threshold
    
    trades = []
    for i in range(len(test_df)):
        if not high_conf_mask[i]:
            continue
        
        signal = y_pred[i]
        if signal == 1:  # HOLD
            continue
        
        # Simulate trade
        entry_price = test_df.iloc[i]['close'] if 'close' in test_df.columns else 2000
        
        if signal == 2:  # LONG
            tp = entry_price + 0.80  # 80 pips
            sl = entry_price - 0.40  # 40 pips
            direction = 1
        else:  # SHORT
            tp = entry_price - 0.80
            sl = entry_price + 0.40
            direction = -1
        
        # Check outcome (simplified)
        actual = y_test[i]
        if actual == signal:
            pnl = 0.80  # Hit TP
            win = True
        else:
            pnl = -0.40  # Hit SL
            win = False
        
        trades.append({
            'signal': signal,
            'actual': actual,
            'pnl': pnl,
            'win': win
        })
    
    if not trades:
        print("  No trades with confidence >= 0.60")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # Metrics
    total_trades = len(trades_df)
    wins = trades_df['win'].sum()
    losses = total_trades - wins
    win_rate = wins / total_trades
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['win']]['pnl'].mean() if wins > 0 else 0
    avg_loss = trades_df[~trades_df['win']]['pnl'].mean() if losses > 0 else 0
    profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 else 0
    
    # Drawdown
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()
    
    print(f"\n  Total trades: {total_trades:,}")
    print(f"  Wins: {wins:,} ({win_rate:.2%})")
    print(f"  Losses: {losses:,}")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Avg win: ${avg_win:.2f}")
    print(f"  Avg loss: ${avg_loss:.2f}")
    print(f"  Profit factor: {profit_factor:.2f}")
    print(f"  Max drawdown: ${max_drawdown:.2f}")
    
    # Per signal
    print("\nPer signal:")
    for sig in [0, 2]:
        sig_trades = trades_df[trades_df['signal'] == sig]
        if len(sig_trades) > 0:
            sig_wins = sig_trades['win'].sum()
            sig_wr = sig_wins / len(sig_trades)
            sig_name = 'SHORT' if sig == 0 else 'LONG'
            print(f"  {sig_name}: {len(sig_trades):,} trades, {sig_wr:.2%} win rate")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
