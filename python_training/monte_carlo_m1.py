"""
Monte Carlo simulation on M1 backtest trade sequence.
Randomly shuffles trade order to assess robustness of results.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_trades() -> pd.DataFrame:
    """Load trades from backtest."""
    project_root = Path(__file__).parent.parent
    
    # Re-run backtest to get trade sequence
    print("Loading backtest trades...")
    
    with open(project_root / "config" / "model_meta.json", "r") as f:
        model_meta = json.load(f)
    
    import lightgbm as lgb
    
    CONFIDENCE_THRESHOLD = model_meta["confidence_threshold"]
    TP_PIPS = model_meta["label_params"]["tp_pips"]
    SL_PIPS = model_meta["label_params"]["sl_pips"]
    INITIAL_CAPITAL = 50.0
    RISK_PERCENT = 0.05
    SPREAD_PIPS = 2.5
    FIXED_RISK = INITIAL_CAPITAL * RISK_PERCENT
    
    # Load test data
    test_path = project_root / "data" / "processed" / "features_m1_test.parquet"
    df_test = pd.read_parquet(test_path)
    
    # Sample
    SAMPLE_RATE = max(1, len(df_test) // 100000)
    df_test = df_test.iloc[::SAMPLE_RATE].copy().reset_index(drop=True)
    
    # Load feature order
    with open(project_root / "config" / "features_order.json", "r") as f:
        feature_cols = json.load(f)
    
    # Load model
    model_path = project_root / "python_training" / "models" / "lightgbm_xauusd.pkl"
    model = lgb.Booster(model_file=str(model_path))
    
    # Predictions
    X_test = df_test[feature_cols].values
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    max_proba = np.max(y_pred_proba, axis=1)
    
    df_test = df_test.copy()
    df_test["pred_class"] = y_pred
    df_test["pred_confidence"] = max_proba
    
    # Filter
    df_signals = df_test[
        (df_test["pred_confidence"] >= CONFIDENCE_THRESHOLD) &
        (df_test["pred_class"] != 1)
    ].copy()
    
    # Calculate PnL for each trade
    pnls = []
    for idx, row in df_signals.iterrows():
        signal = int(row["pred_class"])
        label = int(row["label"])
        
        if signal == label:
            pips_gained = TP_PIPS - SPREAD_PIPS
        else:
            pips_gained = -(SL_PIPS + SPREAD_PIPS)
        
        pnl = FIXED_RISK * (pips_gained / SL_PIPS)
        pnls.append(pnl)
    
    print(f"   Loaded {len(pnls):,} trades")
    return pd.DataFrame({"pnl": pnls})


def run_simulation(pnls: np.ndarray, initial_capital: float) -> Dict:
    """Run one Monte Carlo simulation by shuffling trade order."""
    shuffled = np.random.permutation(pnls)
    
    equity_curve = [initial_capital]
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    
    for pnl in shuffled:
        equity += pnl
        equity_curve.append(equity)
        
        if equity > peak:
            peak = equity
        
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    final_equity = equity
    total_return = (final_equity - initial_capital) / initial_capital
    
    wins = (shuffled > 0).sum()
    losses = (shuffled <= 0).sum()
    win_rate = wins / len(shuffled) if len(shuffled) > 0 else 0
    
    gross_profit = shuffled[shuffled > 0].sum()
    gross_loss = abs(shuffled[shuffled <= 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
    }


def main():
    print("=" * 70)
    print("MONTE CARLO SIMULATION - M1 BACKTEST")
    print("=" * 70)
    print()
    
    INITIAL_CAPITAL = 50.0
    NUM_SIMULATIONS = 10000
    RUIN_THRESHOLD = 0.25  # 25% of starting capital
    
    print(f"⚙️ Configuration:")
    print(f"   Initial Capital: ${INITIAL_CAPITAL}")
    print(f"   Simulations: {NUM_SIMULATIONS:,}")
    print(f"   Ruin Threshold: ${INITIAL_CAPITAL * RUIN_THRESHOLD:.2f}")
    print()
    
    # Load trades
    trades_df = load_trades()
    pnls = trades_df["pnl"].values
    
    print()
    print(f"📊 Trade Statistics:")
    print(f"   Total Trades: {len(pnls):,}")
    print(f"   Mean PnL: ${pnls.mean():.2f}")
    print(f"   Std Dev: ${pnls.std():.2f}")
    print(f"   Win Rate: {(pnls > 0).sum() / len(pnls) * 100:.1f}%")
    print()
    
    # Run Monte Carlo
    print(f"🎲 Running {NUM_SIMULATIONS:,} Monte Carlo simulations...")
    results = []
    
    for i in range(NUM_SIMULATIONS):
        if (i + 1) % 1000 == 0:
            print(f"   Progress: {i+1:,}/{NUM_SIMULATIONS:,} ({(i+1)/NUM_SIMULATIONS*100:.0f}%)")
        
        result = run_simulation(pnls, INITIAL_CAPITAL)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    print()
    
    # Calculate statistics
    print("=" * 70)
    print("📈 MONTE CARLO RESULTS")
    print("=" * 70)
    print()
    
    print("💰 Final Equity Distribution:")
    for pct in [5, 25, 50, 75, 95]:
        val = results_df["final_equity"].quantile(pct / 100)
        print(f"   {pct:2d}th percentile: ${val:,.2f}")
    print()
    
    print("📉 Maximum Drawdown Distribution:")
    for pct in [5, 25, 50, 75, 95]:
        val = results_df["max_drawdown"].quantile(pct / 100) * 100
        print(f"   {pct:2d}th percentile: {val:.1f}%")
    print()
    
    print("💵 Profit Factor Distribution:")
    for pct in [5, 25, 50, 75, 95]:
        val = results_df["profit_factor"].quantile(pct / 100)
        print(f"   {pct:2d}th percentile: {val:.2f}")
    print()
    
    # Risk metrics
    ruin_count = (results_df["final_equity"] < INITIAL_CAPITAL * RUIN_THRESHOLD).sum()
    ruin_prob = ruin_count / NUM_SIMULATIONS * 100
    
    profitable_count = (results_df["final_equity"] > INITIAL_CAPITAL).sum()
    profitable_prob = profitable_count / NUM_SIMULATIONS * 100
    
    print("⚠️ Risk Metrics:")
    print(f"   Probability of Ruin (<${INITIAL_CAPITAL * RUIN_THRESHOLD:.2f}): {ruin_prob:.2f}%")
    print(f"   Probability of Profit: {profitable_prob:.1f}%")
    print(f"   Expected Final Equity: ${results_df['final_equity'].mean():,.2f}")
    print(f"   Expected Max Drawdown: {results_df['max_drawdown'].mean() * 100:.1f}%")
    print()
    
    # Worst/Best cases
    worst_idx = results_df["final_equity"].idxmin()
    best_idx = results_df["final_equity"].idxmax()
    
    print("🎯 Scenarios:")
    print(f"   Worst Case:")
    print(f"      Final Equity: ${results_df.loc[worst_idx, 'final_equity']:,.2f}")
    print(f"      Max Drawdown: {results_df.loc[worst_idx, 'max_drawdown'] * 100:.1f}%")
    print(f"      Profit Factor: {results_df.loc[worst_idx, 'profit_factor']:.2f}")
    print()
    print(f"   Best Case:")
    print(f"      Final Equity: ${results_df.loc[best_idx, 'final_equity']:,.2f}")
    print(f"      Max Drawdown: {results_df.loc[best_idx, 'max_drawdown'] * 100:.1f}%")
    print(f"      Profit Factor: {results_df.loc[best_idx, 'profit_factor']:.2f}")
    print()
    
    # Save results
    project_root = Path(__file__).parent.parent
    results_path = project_root / "python_training" / "models" / "monte_carlo_results.json"
    
    summary = {
        "num_simulations": NUM_SIMULATIONS,
        "initial_capital": INITIAL_CAPITAL,
        "num_trades": int(len(pnls)),
        "percentiles": {
            "final_equity": {
                "p5": float(results_df["final_equity"].quantile(0.05)),
                "p25": float(results_df["final_equity"].quantile(0.25)),
                "p50": float(results_df["final_equity"].quantile(0.50)),
                "p75": float(results_df["final_equity"].quantile(0.75)),
                "p95": float(results_df["final_equity"].quantile(0.95)),
            },
            "max_drawdown": {
                "p5": float(results_df["max_drawdown"].quantile(0.05)),
                "p25": float(results_df["max_drawdown"].quantile(0.25)),
                "p50": float(results_df["max_drawdown"].quantile(0.50)),
                "p75": float(results_df["max_drawdown"].quantile(0.75)),
                "p95": float(results_df["max_drawdown"].quantile(0.95)),
            },
        },
        "risk_metrics": {
            "probability_of_ruin": float(ruin_prob),
            "probability_of_profit": float(profitable_prob),
            "expected_final_equity": float(results_df["final_equity"].mean()),
            "expected_max_drawdown": float(results_df["max_drawdown"].mean()),
        },
    }
    
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved: {results_path.name}")
    print()
    
    # Create visualizations
    print("📊 Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Final Equity Distribution
    ax = axes[0, 0]
    ax.hist(results_df["final_equity"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(results_df["final_equity"].median(), color="red", linestyle="--", label="Median")
    ax.axvline(results_df["final_equity"].quantile(0.05), color="orange", linestyle="--", label="5th %ile")
    ax.axvline(results_df["final_equity"].quantile(0.95), color="green", linestyle="--", label="95th %ile")
    ax.set_xlabel("Final Equity ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Final Equity Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Max Drawdown Distribution
    ax = axes[0, 1]
    ax.hist(results_df["max_drawdown"] * 100, bins=50, edgecolor="black", alpha=0.7, color="red")
    ax.axvline(results_df["max_drawdown"].median() * 100, color="darkred", linestyle="--", label="Median")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Maximum Drawdown Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Profit Factor Distribution
    ax = axes[1, 0]
    pf_capped = results_df["profit_factor"].clip(upper=10)  # Cap for visualization
    ax.hist(pf_capped, bins=50, edgecolor="black", alpha=0.7, color="green")
    ax.axvline(results_df["profit_factor"].median(), color="darkgreen", linestyle="--", label="Median")
    ax.set_xlabel("Profit Factor")
    ax.set_ylabel("Frequency")
    ax.set_title("Profit Factor Distribution (capped at 10)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Return Distribution
    ax = axes[1, 1]
    returns = results_df["total_return"] * 100
    ax.hist(returns, bins=50, edgecolor="black", alpha=0.7, color="purple")
    ax.axvline(returns.median(), color="darkviolet", linestyle="--", label="Median")
    ax.axvline(0, color="black", linestyle="-", linewidth=2, label="Break-even")
    ax.set_xlabel("Total Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Total Return Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = project_root / "python_training" / "models" / "monte_carlo_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"   Plots saved: {plot_path.name}")
    
    print()
    print("=" * 70)
    print("✅ Monte Carlo simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
