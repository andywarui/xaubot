"""
Out-of-time test: Hold out most recent year for strict OOS validation.
Re-run sensitivity analysis only on OOS period.
"""
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Dict


def run_backtest_oos(
    df_test: pd.DataFrame,
    feature_cols: list,
    model,
    config: Dict
) -> tuple:
    """Run backtest with given configuration."""
    
    CONFIDENCE_THRESHOLD = config["confidence_threshold"]
    TP_PIPS = config["tp_pips"]
    SL_PIPS = config["sl_pips"]
    SPREAD_PIPS = config["spread_pips"]
    SLIPPAGE_PIPS = config["slippage_pips"]
    COMMISSION_USD = config["commission_usd"]
    INITIAL_CAPITAL = config["initial_capital"]
    RISK_PERCENT = config["risk_percent"]
    
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
    FIXED_RISK = INITIAL_CAPITAL * RISK_PERCENT
    
    for idx, row in df_signals.iterrows():
        signal = int(row["pred_class"])
        label = int(row["label"])
        
        if signal == label:
            pips_gained = TP_PIPS - (2 * SPREAD_PIPS) - SLIPPAGE_PIPS
        else:
            pips_lost = SL_PIPS + (2 * SPREAD_PIPS) + SLIPPAGE_PIPS
            pips_gained = -pips_lost
        
        pnl = FIXED_RISK * (pips_gained / SL_PIPS) - COMMISSION_USD
        equity += pnl
        
        trades.append({
            "time": row["time"],
            "signal": "LONG" if signal == 2 else "SHORT",
            "result": "WIN" if signal == label else "LOSS",
            "pnl": pnl,
        })
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {}, []
    
    # Calculate metrics
    total_trades = len(trades_df)
    win_rate = (trades_df["result"] == "WIN").sum() / total_trades
    
    gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    net_profit = trades_df["pnl"].sum()
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "final_equity": equity,
    }, trades_df


def main():
    print("=" * 70)
    print("OUT-OF-TIME TEST (OOS Period)")
    print("=" * 70)
    print()
    
    project_root = Path(__file__).parent.parent
    
    # Load full test data
    print("📥 Loading test data...")
    test_path = project_root / "data" / "processed" / "features_m1_test.parquet"
    df_full = pd.read_parquet(test_path)
    
    # Sample for speed
    SAMPLE_RATE = max(1, len(df_full) // 100000)
    df_full = df_full.iloc[::SAMPLE_RATE].copy().reset_index(drop=True)
    
    print(f"   Full test: {len(df_full):,} bars (sampled 1/{SAMPLE_RATE})")
    print(f"   Period: {df_full['time'].min()} → {df_full['time'].max()}")
    print()
    
    # Split: hold out last 12 months as strict OOS
    cutoff_date = df_full['time'].max() - pd.DateOffset(months=12)
    df_oos = df_full[df_full['time'] >= cutoff_date].copy().reset_index(drop=True)
    
    print(f"🔒 OOS Period (Last 12 months):")
    print(f"   Start: {df_oos['time'].min()}")
    print(f"   End: {df_oos['time'].max()}")
    print(f"   Bars: {len(df_oos):,}")
    print()
    
    # Load features and model
    with open(project_root / "config" / "features_order.json", "r") as f:
        feature_cols = json.load(f)
    
    model_path = project_root / "python_training" / "models" / "lightgbm_xauusd.pkl"
    model = lgb.Booster(model_file=str(model_path))
    
    # Test scenarios on OOS
    scenarios = [
        {
            "name": "REALISTIC OOS",
            "confidence_threshold": 0.60,
            "tp_pips": 80,
            "sl_pips": 40,
            "spread_pips": 2.5,
            "slippage_pips": 1.0,
            "commission_usd": 0,
            "initial_capital": 50.0,
            "risk_percent": 0.02,
        },
        {
            "name": "CONSERVATIVE OOS",
            "confidence_threshold": 0.65,
            "tp_pips": 80,
            "sl_pips": 40,
            "spread_pips": 3.0,
            "slippage_pips": 1.5,
            "commission_usd": 0.10,
            "initial_capital": 50.0,
            "risk_percent": 0.01,
        },
        {
            "name": "LIVE-READY OOS",
            "confidence_threshold": 0.70,
            "tp_pips": 80,
            "sl_pips": 40,
            "spread_pips": 3.0,
            "slippage_pips": 1.5,
            "commission_usd": 0.10,
            "initial_capital": 50.0,
            "risk_percent": 0.005,  # 0.5% risk
        },
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"🧪 Testing: {scenario['name']}")
        
        metrics, trades_df = run_backtest_oos(df_oos, feature_cols, model, scenario)
        
        if metrics:
            print(f"   ✓ Trades: {metrics['total_trades']:,}")
            print(f"   ✓ Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"   ✓ Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   ✓ Net Profit: ${metrics['net_profit']:.2f}")
            print(f"   ✓ Final Equity: ${metrics['final_equity']:.2f}")
        else:
            print("   ✗ No trades generated")
        
        print()
        results.append({"scenario": scenario["name"], "metrics": metrics})
    
    print("=" * 70)
    print("📊 OOS RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    print(f"{'Scenario':<20} {'Trades':<10} {'WR%':<8} {'PF':<8} {'Net $':<12}")
    print("-" * 60)
    
    for result in results:
        if result["metrics"]:
            m = result["metrics"]
            print(f"{result['scenario']:<20} "
                  f"{m['total_trades']:<10,} "
                  f"{m['win_rate']*100:<8.1f} "
                  f"{m['profit_factor']:<8.2f} "
                  f"{m['net_profit']:<12,.2f}")
    
    print()
    print("=" * 70)
    print("🎯 OOS VALIDATION")
    print("=" * 70)
    print()
    print("Compare OOS metrics to in-sample:")
    print("  - Win rate should be similar (±5%)")
    print("  - Profit factor should remain >2.0")
    print("  - If OOS degrades significantly → model overfit")
    print()
    
    # Save results
    results_path = project_root / "python_training" / "models" / "oos_results.json"
    with open(results_path, "w") as f:
        serializable_results = []
        for r in results:
            serializable_results.append({
                "scenario": r["scenario"],
                "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in r["metrics"].items()} if r["metrics"] else {}
            })
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved: {results_path.name}")
    print()


if __name__ == "__main__":
    main()
