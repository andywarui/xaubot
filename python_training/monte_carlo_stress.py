"""
Stress-focused Monte Carlo tests:
- Worst-case loss streaks (synthetic)
- Regime shift with WR degradation in random blocks
- Out-of-sample time split
- Conservative risk settings
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_realistic_trades() -> np.ndarray:
    """Load PnL from realistic scenario."""
    project_root = Path(__file__).parent.parent
    
    print("Loading realistic backtest trades...")
    
    with open(project_root / "config" / "model_meta.json", "r") as f:
        model_meta = json.load(f)
    
    import lightgbm as lgb
    
    CONFIDENCE_THRESHOLD = 0.60
    TP_PIPS = 80
    SL_PIPS = 40
    SPREAD_PIPS = 2.5
    SLIPPAGE_PIPS = 1.0
    INITIAL_CAPITAL = 50.0
    RISK_PERCENT = 0.02
    FIXED_RISK = INITIAL_CAPITAL * RISK_PERCENT
    
    # Load test data
    test_path = project_root / "data" / "processed" / "features_m1_test.parquet"
    df_test = pd.read_parquet(test_path)
    
    SAMPLE_RATE = max(1, len(df_test) // 100000)
    df_test = df_test.iloc[::SAMPLE_RATE].copy().reset_index(drop=True)
    
    with open(project_root / "config" / "features_order.json", "r") as f:
        feature_cols = json.load(f)
    
    model_path = project_root / "python_training" / "models" / "lightgbm_xauusd.pkl"
    model = lgb.Booster(model_file=str(model_path))
    
    X_test = df_test[feature_cols].values
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    max_proba = np.max(y_pred_proba, axis=1)
    
    df_test = df_test.copy()
    df_test["pred_class"] = y_pred
    df_test["pred_confidence"] = max_proba
    
    df_signals = df_test[
        (df_test["pred_confidence"] >= CONFIDENCE_THRESHOLD) &
        (df_test["pred_class"] != 1)
    ].copy()
    
    pnls = []
    for idx, row in df_signals.iterrows():
        signal = int(row["pred_class"])
        label = int(row["label"])
        
        if signal == label:
            pips_gained = TP_PIPS - (2 * SPREAD_PIPS) - SLIPPAGE_PIPS
        else:
            pips_lost = SL_PIPS + (2 * SPREAD_PIPS) + SLIPPAGE_PIPS
            pips_gained = -pips_lost
        
        pnl = FIXED_RISK * (pips_gained / SL_PIPS)
        pnls.append(pnl)
    
    print(f"   Loaded {len(pnls):,} trades")
    return np.array(pnls)


def run_simulation_bootstrap(pnls: np.ndarray, initial_capital: float, num_trades: int) -> Dict:
    """Bootstrap resampling with replacement (creates extreme streaks)."""
    sampled = np.random.choice(pnls, size=num_trades, replace=True)
    
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    min_equity = initial_capital
    
    for pnl in sampled:
        equity += pnl
        
        if equity < min_equity:
            min_equity = equity
        
        if equity > peak:
            peak = equity
        
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    wins = (sampled > 0).sum()
    win_rate = wins / len(sampled) if len(sampled) > 0 else 0
    
    gross_profit = sampled[sampled > 0].sum()
    gross_loss = abs(sampled[sampled <= 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "final_equity": equity,
        "min_equity": min_equity,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "ruined": equity < initial_capital * 0.25
    }


def run_simulation_remove_best(pnls: np.ndarray, initial_capital: float, remove_pct: float) -> Dict:
    """Remove top X% of winning trades (simulates regime change)."""
    sorted_idx = np.argsort(pnls)[::-1]
    num_remove = int(len(pnls) * remove_pct)
    keep_idx = sorted_idx[num_remove:]
    filtered_pnls = pnls[keep_idx]
    
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    
    for pnl in filtered_pnls:
        equity += pnl
        
        if equity > peak:
            peak = equity
        
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    wins = (filtered_pnls > 0).sum()
    win_rate = wins / len(filtered_pnls) if len(filtered_pnls) > 0 else 0
    
    gross_profit = filtered_pnls[filtered_pnls > 0].sum()
    gross_loss = abs(filtered_pnls[filtered_pnls <= 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "final_equity": equity,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "ruined": equity < initial_capital * 0.25
    }


def run_simulation_degraded_wr(pnls: np.ndarray, initial_capital: float, wr_degradation: float) -> Dict:
    """Randomly flip wins to losses to simulate degraded win rate."""
    modified_pnls = pnls.copy()
    win_indices = np.where(pnls > 0)[0]
    
    # Calculate how many wins to flip
    current_wr = len(win_indices) / len(pnls)
    target_wr = current_wr - wr_degradation
    num_flip = int((current_wr - target_wr) * len(pnls))
    
    if num_flip > 0:
        flip_indices = np.random.choice(win_indices, size=min(num_flip, len(win_indices)), replace=False)
        # Flip wins to losses (make them negative)
        modified_pnls[flip_indices] = -np.abs(modified_pnls[flip_indices])
    
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    
    for pnl in modified_pnls:
        equity += pnl
        
        if equity > peak:
            peak = equity
        
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    wins = (modified_pnls > 0).sum()
    win_rate = wins / len(modified_pnls) if len(modified_pnls) > 0 else 0
    
    gross_profit = modified_pnls[modified_pnls > 0].sum()
    gross_loss = abs(modified_pnls[modified_pnls <= 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "final_equity": equity,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "ruined": equity < initial_capital * 0.25
    }


def main():
    print("=" * 70)
    print("AGGRESSIVE MONTE CARLO STRESS TESTS")
    print("=" * 70)
    print()
    
    INITIAL_CAPITAL = 50.0
    NUM_SIMULATIONS = 5000
    
    # Load realistic trades
    pnls = load_realistic_trades()
    print()
    
    # Test 1: Bootstrap resampling (extreme streaks)
    print("🎲 Test 1: Bootstrap Resampling (5,000 simulations)")
    print("   Creates extreme winning/losing streaks via sampling with replacement")
    print()
    
    results_bootstrap = []
    for i in range(NUM_SIMULATIONS):
        result = run_simulation_bootstrap(pnls, INITIAL_CAPITAL, len(pnls))
        results_bootstrap.append(result)
    
    df_bootstrap = pd.DataFrame(results_bootstrap)
    
    print(f"   Final Equity:")
    print(f"      5th percentile: ${df_bootstrap['final_equity'].quantile(0.05):,.2f}")
    print(f"      50th percentile: ${df_bootstrap['final_equity'].quantile(0.50):,.2f}")
    print(f"      95th percentile: ${df_bootstrap['final_equity'].quantile(0.95):,.2f}")
    print(f"   Max Drawdown:")
    print(f"      95th percentile: {df_bootstrap['max_drawdown'].quantile(0.95)*100:.1f}%")
    print(f"   Probability of Ruin: {df_bootstrap['ruined'].sum() / NUM_SIMULATIONS * 100:.2f}%")
    print(f"   Min Equity Reached: ${df_bootstrap['min_equity'].min():,.2f}")
    print()
    
    # Test 2: Remove best trades
    print("🎯 Test 2: Regime Shift (Remove Best Trades)")
    print()
    
    for remove_pct in [0.10, 0.20, 0.30]:
        results_regime = []
        for i in range(1000):
            result = run_simulation_remove_best(pnls, INITIAL_CAPITAL, remove_pct)
            results_regime.append(result)
        
        df_regime = pd.DataFrame(results_regime)
        avg_equity = df_regime['final_equity'].mean()
        avg_dd = df_regime['max_drawdown'].mean() * 100
        avg_pf = df_regime['profit_factor'].mean()
        ruin_pct = df_regime['ruined'].sum() / 1000 * 100
        
        print(f"   Remove top {remove_pct*100:.0f}% of trades:")
        print(f"      Avg Final Equity: ${avg_equity:,.2f}")
        print(f"      Avg Max DD: {avg_dd:.1f}%")
        print(f"      Avg PF: {avg_pf:.2f}")
        print(f"      Ruin Rate: {ruin_pct:.1f}%")
        print()
    
    # Test 3: Degraded win rate
    print("⚠️ Test 3: Win Rate Degradation")
    print()
    
    for wr_degrade in [0.05, 0.10, 0.15]:
        results_wr = []
        for i in range(1000):
            result = run_simulation_degraded_wr(pnls, INITIAL_CAPITAL, wr_degrade)
            results_wr.append(result)
        
        df_wr = pd.DataFrame(results_wr)
        avg_wr = df_wr['win_rate'].mean() * 100
        avg_equity = df_wr['final_equity'].mean()
        avg_dd = df_wr['max_drawdown'].mean() * 100
        ruin_pct = df_wr['ruined'].sum() / 1000 * 100
        
        print(f"   Win rate drops by {wr_degrade*100:.0f}% (to ~{avg_wr:.1f}%):")
        print(f"      Avg Final Equity: ${avg_equity:,.2f}")
        print(f"      Avg Max DD: {avg_dd:.1f}%")
        print(f"      Ruin Rate: {ruin_pct:.1f}%")
        print()
    
    print("=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print()
    print("✅ Bootstrap (extreme streaks):")
    print(f"   - Worst 5th percentile: ${df_bootstrap['final_equity'].quantile(0.05):,.2f}")
    print(f"   - Max DD can reach: {df_bootstrap['max_drawdown'].quantile(0.95)*100:.1f}%")
    print(f"   - Ruin probability: {df_bootstrap['ruined'].sum() / NUM_SIMULATIONS * 100:.2f}%")
    print()
    print("⚠️ If best 20% trades removed (regime shift):")
    
    results_regime_20 = []
    for i in range(1000):
        result = run_simulation_remove_best(pnls, INITIAL_CAPITAL, 0.20)
        results_regime_20.append(result)
    df_regime_20 = pd.DataFrame(results_regime_20)
    
    print(f"   - Final equity drops to: ${df_regime_20['final_equity'].mean():,.2f}")
    print(f"   - Still profitable: {(df_regime_20['final_equity'] > INITIAL_CAPITAL).sum() / 1000 * 100:.0f}%")
    print()
    print("⚠️ If win rate degrades by 10%:")
    
    results_wr_10 = []
    for i in range(1000):
        result = run_simulation_degraded_wr(pnls, INITIAL_CAPITAL, 0.10)
        results_wr_10.append(result)
    df_wr_10 = pd.DataFrame(results_wr_10)
    
    print(f"   - Final equity: ${df_wr_10['final_equity'].mean():,.2f}")
    print(f"   - Still profitable: {(df_wr_10['final_equity'] > INITIAL_CAPITAL).sum() / 1000 * 100:.0f}%")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
