"""
M1-based backtest using trained LightGBM model on test set.
Uses the same features and labels as training for consistency.
"""
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path


def main():
    print("=" * 70)
    print("M1 BACKTEST - XAUUSD AI BOT")
    print("=" * 70)
    print()
    
    project_root = Path(__file__).parent.parent
    
    # Configuration
    with open(project_root / "config" / "model_meta.json", "r") as f:
        model_meta = json.load(f)
    
    CONFIDENCE_THRESHOLD = model_meta["confidence_threshold"]
    TP_PIPS = model_meta["label_params"]["tp_pips"]
    SL_PIPS = model_meta["label_params"]["sl_pips"]
    
    INITIAL_CAPITAL = 50.0
    RISK_PERCENT = 0.05
    SPREAD_PIPS = 2.5
    
    print("⚙️ Backtest Configuration:")
    print(f"   Initial Capital: ${INITIAL_CAPITAL}")
    print(f"   Risk per Trade: {RISK_PERCENT*100:.1f}%")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"   TP: {TP_PIPS} pips | SL: {SL_PIPS} pips")
    print(f"   Spread: {SPREAD_PIPS} pips")
    print()
    
    # Load test data
    print("📥 Loading test data...")
    test_path = project_root / "data" / "processed" / "features_m1_test.parquet"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run build_features_m1.py first.")
        return
    
    df_test = pd.read_parquet(test_path)
    
    # Sample for faster backtest (use every 10th bar to simulate ~100k trades max)
    SAMPLE_RATE = max(1, len(df_test) // 100000)
    df_test = df_test.iloc[::SAMPLE_RATE].copy().reset_index(drop=True)
    
    print(f"   Test bars: {len(df_test):,} (sampled 1/{SAMPLE_RATE})")
    print(f"   Test period: {df_test['time'].min()} → {df_test['time'].max()}")
    print()
    
    # Load feature order
    with open(project_root / "config" / "features_order.json", "r") as f:
        feature_cols = json.load(f)
    
    print(f"   Features: {len(feature_cols)}")
    
    # Load model
    model_path = project_root / "python_training" / "models" / "lightgbm_xauusd.pkl"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run train_lightgbm.py first.")
        return
    
    model = lgb.Booster(model_file=str(model_path))
    print(f"   Model loaded: {model_path.name}")
    print()
    
    # Predictions
    print("🔮 Running model predictions...")
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    max_proba = np.max(y_pred_proba, axis=1)
    
    df_test = df_test.copy()
    df_test["pred_class"] = y_pred
    df_test["pred_confidence"] = max_proba
    
    print(f"   ✓ {len(df_test):,} predictions generated")
    print()
    
    # Filter by confidence and non-HOLD predictions
    df_signals = df_test[
        (df_test["pred_confidence"] >= CONFIDENCE_THRESHOLD) &
        (df_test["pred_class"] != 1)
    ].copy()
    
    print(f"📊 High-Confidence Signals (excl. HOLD):")
    print(f"   Total: {len(df_signals):,} ({len(df_signals)/len(df_test)*100:.1f}% of test set)")
    longs = (df_signals["pred_class"] == 2).sum()
    shorts = (df_signals["pred_class"] == 0).sum()
    print(f"   LONG: {longs:,} | SHORT: {shorts:,}")
    print()
    
    # Simulate trades using label as outcome
    # label: 0=SHORT wins (price fell to TP), 1=HOLD, 2=LONG wins (price rose to TP)
    print("💰 Simulating trades...")
    
    # Fixed risk amount per trade (no compounding to avoid blowup on large trade counts)
    FIXED_RISK = INITIAL_CAPITAL * RISK_PERCENT  # $2.50 per trade
    
    trades = []
    equity = INITIAL_CAPITAL
    equity_curve = [INITIAL_CAPITAL]
    peak_equity = INITIAL_CAPITAL
    
    for idx, row in df_signals.iterrows():
        signal = int(row["pred_class"])  # 0=SHORT, 2=LONG
        label = int(row["label"])
        confidence = row["pred_confidence"]
        
        # Determine win/loss
        # LONG signal (2) wins if label==2
        # SHORT signal (0) wins if label==0
        if signal == label:
            trade_result = "WIN"
            # TP hit: gain = TP - spread
            pips_gained = TP_PIPS - SPREAD_PIPS
        else:
            trade_result = "LOSS"
            # SL hit: loss = SL + spread
            pips_gained = -(SL_PIPS + SPREAD_PIPS)
        
        # Fixed risk PnL (not compounding)
        pnl = FIXED_RISK * (pips_gained / SL_PIPS)
        
        equity += pnl
        equity_curve.append(equity)
        
        if equity > peak_equity:
            peak_equity = equity
        
        trades.append({
            "time": row["time"],
            "signal": "LONG" if signal == 2 else "SHORT",
            "label": label,
            "confidence": confidence,
            "result": trade_result,
            "pnl": pnl,
            "equity": equity
        })
    
    trades_df = pd.DataFrame(trades)
    print(f"   ✓ {len(trades_df):,} trades executed")
    print()
    
    # Calculate metrics
    print("=" * 70)
    print("📈 BACKTEST RESULTS")
    print("=" * 70)
    print()
    
    total_trades = len(trades_df)
    if total_trades == 0:
        print("No trades executed.")
        return
    
    winning_trades = (trades_df["result"] == "WIN").sum()
    losing_trades = (trades_df["result"] == "LOSS").sum()
    win_rate = winning_trades / total_trades
    
    gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    net_profit = trades_df["pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    # Drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # By direction
    long_trades = trades_df[trades_df["signal"] == "LONG"]
    short_trades = trades_df[trades_df["signal"] == "SHORT"]
    
    long_wins = (long_trades["result"] == "WIN").sum() if len(long_trades) > 0 else 0
    short_wins = (short_trades["result"] == "WIN").sum() if len(short_trades) > 0 else 0
    long_wr = long_wins / len(long_trades) if len(long_trades) > 0 else 0
    short_wr = short_wins / len(short_trades) if len(short_trades) > 0 else 0
    
    print(f"💼 Trading Performance:")
    print(f"   Total Trades: {total_trades:,}")
    print(f"   Winning: {winning_trades:,} | Losing: {losing_trades:,}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print()
    print(f"💵 Profit/Loss:")
    print(f"   Gross Profit: ${gross_profit:.2f}")
    print(f"   Gross Loss: ${gross_loss:.2f}")
    print(f"   Net Profit: ${net_profit:.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print()
    print(f"📉 Risk:")
    print(f"   Max Drawdown: {max_drawdown*100:.1f}%")
    print(f"   Final Equity: ${equity:.2f}")
    print(f"   Return: {(equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:.1f}%")
    print()
    print(f"📊 By Direction:")
    print(f"   LONG: {len(long_trades):,} trades | {long_wr*100:.1f}% WR | ${long_trades['pnl'].sum():.2f}")
    print(f"   SHORT: {len(short_trades):,} trades | {short_wr*100:.1f}% WR | ${short_trades['pnl'].sum():.2f}")
    print()
    print("=" * 70)
    
    # Save results
    results = {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2),
        "net_profit": round(net_profit, 2),
        "max_drawdown": round(max_drawdown, 4),
        "final_equity": round(equity, 2),
        "long_trades": len(long_trades),
        "long_win_rate": round(long_wr, 4),
        "short_trades": len(short_trades),
        "short_win_rate": round(short_wr, 4)
    }
    
    results_path = project_root / "python_training" / "models" / "backtest_m1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved: {results_path.name}")


if __name__ == "__main__":
    main()
