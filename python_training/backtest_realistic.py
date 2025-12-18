"""
Realistic M1 backtest with conservative assumptions:
- Proper spread modeling (2.5 pips entry + 2.5 pips exit)
- Slippage (1 pip average)
- Lower risk per trade (1-2%)
- Commission if applicable
- No compounding initially to isolate edge
"""
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Dict


def run_backtest(
    df_test: pd.DataFrame,
    feature_cols: list,
    model,
    config: Dict
) -> pd.DataFrame:
    """Run backtest with given configuration."""
    
    # Extract config
    CONFIDENCE_THRESHOLD = config["confidence_threshold"]
    TP_PIPS = config["tp_pips"]
    SL_PIPS = config["sl_pips"]
    SPREAD_PIPS = config["spread_pips"]
    SLIPPAGE_PIPS = config["slippage_pips"]
    COMMISSION_USD = config["commission_usd"]
    INITIAL_CAPITAL = config["initial_capital"]
    RISK_PERCENT = config["risk_percent"]
    USE_COMPOUNDING = config["use_compounding"]
    
    # Predictions
    X_test = df_test[feature_cols].values
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    max_proba = np.max(y_pred_proba, axis=1)
    
    df_test = df_test.copy()
    df_test["pred_class"] = y_pred
    df_test["pred_confidence"] = max_proba
    
    # Filter signals
    df_signals = df_test[
        (df_test["pred_confidence"] >= CONFIDENCE_THRESHOLD) &
        (df_test["pred_class"] != 1)
    ].copy()
    
    # Simulate trades
    trades = []
    equity = INITIAL_CAPITAL
    equity_curve = [INITIAL_CAPITAL]
    peak_equity = INITIAL_CAPITAL
    
    for idx, row in df_signals.iterrows():
        signal = int(row["pred_class"])
        label = int(row["label"])
        
        # Determine outcome
        if signal == label:
            # WIN: TP hit
            # Account for: spread on entry, spread on exit, slippage
            pips_gained = TP_PIPS - (2 * SPREAD_PIPS) - SLIPPAGE_PIPS
        else:
            # LOSS: SL hit
            pips_lost = SL_PIPS + (2 * SPREAD_PIPS) + SLIPPAGE_PIPS
            pips_gained = -pips_lost
        
        # Calculate PnL
        if USE_COMPOUNDING:
            risk_amount = equity * RISK_PERCENT
        else:
            risk_amount = INITIAL_CAPITAL * RISK_PERCENT
        
        pnl = risk_amount * (pips_gained / SL_PIPS) - COMMISSION_USD
        
        equity += pnl
        equity_curve.append(equity)
        
        if equity > peak_equity:
            peak_equity = equity
        
        trades.append({
            "time": row["time"],
            "signal": "LONG" if signal == 2 else "SHORT",
            "label": label,
            "confidence": row["pred_confidence"],
            "result": "WIN" if signal == label else "LOSS",
            "pnl": pnl,
            "equity": equity,
            "pips": pips_gained
        })
    
    return pd.DataFrame(trades), equity_curve


def calculate_metrics(trades_df: pd.DataFrame, equity_curve: list, initial_capital: float) -> Dict:
    """Calculate performance metrics."""
    
    if len(trades_df) == 0:
        return {}
    
    total_trades = len(trades_df)
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
    
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # By direction
    long_trades = trades_df[trades_df["signal"] == "LONG"]
    short_trades = trades_df[trades_df["signal"] == "SHORT"]
    
    long_wins = (long_trades["result"] == "WIN").sum() if len(long_trades) > 0 else 0
    short_wins = (short_trades["result"] == "WIN").sum() if len(short_trades) > 0 else 0
    long_wr = long_wins / len(long_trades) if len(long_trades) > 0 else 0
    short_wr = short_wins / len(short_trades) if len(short_trades) > 0 else 0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_drawdown": max_drawdown,
        "final_equity": final_equity,
        "total_return": total_return,
        "long_trades": len(long_trades),
        "long_win_rate": long_wr,
        "long_pnl": long_trades["pnl"].sum() if len(long_trades) > 0 else 0,
        "short_trades": len(short_trades),
        "short_win_rate": short_wr,
        "short_pnl": short_trades["pnl"].sum() if len(short_trades) > 0 else 0,
    }


def main():
    print("=" * 80)
    print("REALISTIC BACKTEST WITH PARAMETER SENSITIVITY")
    print("=" * 80)
    print()
    
    project_root = Path(__file__).parent.parent
    
    # Load model metadata
    with open(project_root / "config" / "model_meta.json", "r") as f:
        model_meta = json.load(f)
    
    # Load test data
    print("📥 Loading test data...")
    test_path = project_root / "data" / "processed" / "features_m1_test.parquet"
    df_test = pd.read_parquet(test_path)
    
    # Sample for speed
    SAMPLE_RATE = max(1, len(df_test) // 100000)
    df_test = df_test.iloc[::SAMPLE_RATE].copy().reset_index(drop=True)
    print(f"   Test bars: {len(df_test):,} (sampled 1/{SAMPLE_RATE})")
    print()
    
    # Load features and model
    with open(project_root / "config" / "features_order.json", "r") as f:
        feature_cols = json.load(f)
    
    model_path = project_root / "python_training" / "models" / "lightgbm_xauusd.pkl"
    model = lgb.Booster(model_file=str(model_path))
    
    # Test scenarios
    scenarios = [
        {
            "name": "ORIGINAL (Optimistic)",
            "confidence_threshold": 0.60,
            "tp_pips": 80,
            "sl_pips": 40,
            "spread_pips": 1.25,  # Half spread on entry
            "slippage_pips": 0,
            "commission_usd": 0,
            "initial_capital": 50.0,
            "risk_percent": 0.05,
            "use_compounding": False,
        },
        {
            "name": "REALISTIC (Conservative)",
            "confidence_threshold": 0.60,
            "tp_pips": 80,
            "sl_pips": 40,
            "spread_pips": 2.5,  # Full spread both entry + exit
            "slippage_pips": 1.0,
            "commission_usd": 0,
            "initial_capital": 50.0,
            "risk_percent": 0.02,  # 2% risk
            "use_compounding": False,
        },
        {
            "name": "VERY CONSERVATIVE",
            "confidence_threshold": 0.65,
            "tp_pips": 80,
            "sl_pips": 40,
            "spread_pips": 3.0,
            "slippage_pips": 1.5,
            "commission_usd": 0.10,  # $0.10 per trade
            "initial_capital": 50.0,
            "risk_percent": 0.01,  # 1% risk
            "use_compounding": False,
        },
        {
            "name": "TIGHT TP/SL",
            "confidence_threshold": 0.60,
            "tp_pips": 60,  # Tighter TP
            "sl_pips": 40,
            "spread_pips": 2.5,
            "slippage_pips": 1.0,
            "commission_usd": 0,
            "initial_capital": 50.0,
            "risk_percent": 0.02,
            "use_compounding": False,
        },
        {
            "name": "WIDE TP/SL",
            "confidence_threshold": 0.60,
            "tp_pips": 100,  # Wider TP
            "sl_pips": 50,   # Wider SL
            "spread_pips": 2.5,
            "slippage_pips": 1.0,
            "commission_usd": 0,
            "initial_capital": 50.0,
            "risk_percent": 0.02,
            "use_compounding": False,
        },
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"🔬 Testing: {scenario['name']}")
        print(f"   Confidence: {scenario['confidence_threshold']:.2f}")
        print(f"   TP/SL: {scenario['tp_pips']}/{scenario['sl_pips']} pips")
        print(f"   Spread: {scenario['spread_pips']} pips")
        print(f"   Slippage: {scenario['slippage_pips']} pips")
        print(f"   Risk: {scenario['risk_percent']*100:.1f}%")
        
        trades_df, equity_curve = run_backtest(df_test, feature_cols, model, scenario)
        metrics = calculate_metrics(trades_df, equity_curve, scenario["initial_capital"])
        
        if metrics:
            print(f"   ✓ Trades: {metrics['total_trades']:,}")
            print(f"   ✓ Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"   ✓ Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   ✓ Net Profit: ${metrics['net_profit']:.2f}")
            print(f"   ✓ Max DD: {metrics['max_drawdown']*100:.1f}%")
            print(f"   ✓ Return: {metrics['total_return']*100:.1f}%")
        else:
            print("   ✗ No trades generated")
        
        print()
        
        result = {
            "scenario": scenario["name"],
            "config": scenario,
            "metrics": metrics
        }
        results.append(result)
    
    # Summary comparison
    print("=" * 80)
    print("📊 SCENARIO COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Scenario':<25} {'Trades':<10} {'WR%':<8} {'PF':<8} {'Net $':<12} {'DD%':<8} {'Ret%':<10}")
    print("-" * 80)
    
    for result in results:
        if result["metrics"]:
            m = result["metrics"]
            print(f"{result['scenario']:<25} "
                  f"{m['total_trades']:<10,} "
                  f"{m['win_rate']*100:<8.1f} "
                  f"{m['profit_factor']:<8.2f} "
                  f"{m['net_profit']:<12,.0f} "
                  f"{m['max_drawdown']*100:<8.1f} "
                  f"{m['total_return']*100:<10.0f}")
    
    print()
    print("=" * 80)
    
    # Save results
    results_path = project_root / "python_training" / "models" / "sensitivity_analysis.json"
    with open(results_path, "w") as f:
        # Convert to serializable format
        serializable_results = []
        for r in results:
            serializable_results.append({
                "scenario": r["scenario"],
                "config": r["config"],
                "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in r["metrics"].items()} if r["metrics"] else {}
            })
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved: {results_path.name}")
    print()


if __name__ == "__main__":
    main()
