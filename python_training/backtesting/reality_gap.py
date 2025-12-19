"""
Reality Gap Testing for XAUUSD Neural Trading Bot.

Tests strategy performance with increasing levels of real-world friction:
- Level 0: Baseline (perfect execution)
- Level 1: + Spread ($0.20-0.30)
- Level 2: + Slippage ($0.00-0.10 random)
- Level 3: + Commission ($7 per lot RT)
- Level 4: + Swap (overnight costs)
- Level 5: + Partial Fills (80% fill rate)
- Level 6: + Latency (100-500ms)
- Level 7: + Weekend Gaps

Phase 2.6 Implementation per XAUBOT_DEVELOPMENT_PLAN.md
"""
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RealityGapTester:
    """Reality gap testing to validate strategy under real-world conditions."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data" / "processed"
        self.models_dir = project_root / "python_training" / "models"
        self.results_dir = project_root / "python_training" / "backtesting" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model config
        with open(self.models_dir / "lightgbm_balanced_config.json", "r") as f:
            self.config = json.load(f)
        
        self.feature_cols = self.config["feature_cols"]
        self.thresholds = self.config["thresholds"]
        
        # Base trading parameters
        self.initial_capital = 10000.0
        self.risk_per_trade = 0.02
        self.tp_pips = 30
        self.sl_pips = 20
        self.pip_value = 10.0  # $10 per pip per lot
        
        # Reality friction levels
        self.friction_levels = {
            0: {"name": "Baseline (Perfect)", "spread": 0, "slippage": 0, 
                "commission": 0, "swap": 0, "fill_rate": 1.0, "latency": 0},
            1: {"name": "+ Spread", "spread": 2.5, "slippage": 0,
                "commission": 0, "swap": 0, "fill_rate": 1.0, "latency": 0},
            2: {"name": "+ Slippage", "spread": 2.5, "slippage": 1.0,
                "commission": 0, "swap": 0, "fill_rate": 1.0, "latency": 0},
            3: {"name": "+ Commission", "spread": 2.5, "slippage": 1.0,
                "commission": 7.0, "swap": 0, "fill_rate": 1.0, "latency": 0},
            4: {"name": "+ Swap", "spread": 2.5, "slippage": 1.0,
                "commission": 7.0, "swap": 2.0, "fill_rate": 1.0, "latency": 0},
            5: {"name": "+ Partial Fills", "spread": 2.5, "slippage": 1.0,
                "commission": 7.0, "swap": 2.0, "fill_rate": 0.8, "latency": 0},
            6: {"name": "+ Latency", "spread": 2.5, "slippage": 1.0,
                "commission": 7.0, "swap": 2.0, "fill_rate": 0.8, "latency": 300},
            7: {"name": "+ Weekend Gaps", "spread": 2.5, "slippage": 1.0,
                "commission": 7.0, "swap": 2.0, "fill_rate": 0.8, "latency": 300,
                "weekend_gaps": True},
        }
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data with timestamps."""
        print("📥 Loading test data...")
        
        path = self.data_dir / "hybrid_features_test.parquet"
        df = pd.read_parquet(path)
        
        # Load M1 for timestamps
        m1_path = self.data_dir / "xauusd_M1.parquet"
        if m1_path.exists():
            m1_df = pd.read_parquet(m1_path)
            # Convert time column to datetime if needed (with error handling)
            if m1_df["time"].dtype == "object":
                m1_df["time"] = pd.to_datetime(m1_df["time"], errors="coerce")
                m1_df = m1_df.dropna(subset=["time"])
            # Get test portion (last ~20% based on typical train/val/test split)
            test_start = int(len(m1_df) * 0.8)
            if len(m1_df) - test_start >= len(df):
                df["time"] = m1_df["time"].iloc[test_start:test_start+len(df)].values
            else:
                df["time"] = m1_df["time"].iloc[-len(df):].values
        
        print(f"   Loaded: {len(df):,} rows")
        
        return df
    
    def load_model(self) -> lgb.Booster:
        """Load trained LightGBM model.
        
        Note: Uses model_str instead of model_file to work around
        LightGBM 4.6.0 Windows threading bug with file loading.
        """
        model_path = self.models_dir / "lightgbm_balanced.txt"
        with open(model_path, 'r', encoding='utf-8') as f:
            model_str = f.read()
        return lgb.Booster(model_str=model_str)
    
    def generate_signals(self, df: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:
        """Generate trading signals."""
        X = df[self.feature_cols].values
        proba = model.predict(X)

        short_thresh = self.thresholds.get("SHORT", 0.48)
        long_thresh = self.thresholds.get("LONG", 0.40)

        signals = []
        for i in range(len(df)):
            p = proba[i]
            # Convert label to int to ensure proper comparison (handles float labels like 0.0, 1.0, 2.0)
            label_raw = df["label"].iloc[i]
            if pd.isna(label_raw):
                continue  # Skip rows with NaN labels
            label = int(round(label_raw))
            time = df["time"].iloc[i] if "time" in df.columns else None

            if p[0] >= short_thresh and p[0] >= p[2]:
                direction = -1
                confidence = p[0]
            elif p[2] >= long_thresh and p[2] > p[0]:
                direction = 1
                confidence = p[2]
            else:
                continue

            signals.append({
                "idx": i,
                "time": time,
                "direction": direction,
                "confidence": confidence,
                "label": label,
            })

        return pd.DataFrame(signals)
    
    def simulate_with_friction(self, signals_df: pd.DataFrame, level: int) -> Dict:
        """Simulate trading with specified friction level."""
        friction = self.friction_levels[level]

        capital = self.initial_capital
        peak = capital
        max_drawdown = 0.0

        trades_executed = 0
        trades_skipped = 0
        hold_trades = 0  # Trades where label == 1 (no TP/SL hit)
        total_costs = 0.0
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0

        for _, signal in signals_df.iterrows():
            direction = signal["direction"]
            label = int(signal["label"])  # Ensure integer comparison
            time = signal["time"]

            # Check partial fill
            if np.random.random() > friction["fill_rate"]:
                trades_skipped += 1
                continue

            # Calculate base PnL
            if direction == 1:  # LONG
                if label == 2:
                    base_pnl = self.tp_pips
                elif label == 0:
                    base_pnl = -self.sl_pips
                else:
                    base_pnl = 0
                    hold_trades += 1
            else:  # SHORT
                if label == 0:
                    base_pnl = self.tp_pips
                elif label == 2:
                    base_pnl = -self.sl_pips
                else:
                    base_pnl = 0
                    hold_trades += 1

            # Apply friction costs
            spread_cost = friction["spread"]
            slippage_cost = np.random.uniform(0, friction["slippage"]) if friction["slippage"] > 0 else 0

            # Position sizing with overflow protection
            risk_amount = min(capital * self.risk_per_trade, 1e12)  # Cap at $1T
            lot_size = risk_amount / (self.sl_pips * self.pip_value)
            lot_size = min(lot_size, 1e9)  # Cap lot size to prevent overflow

            # Calculate costs
            commission_cost = friction["commission"] * lot_size

            # Swap cost (simplified: apply to any trade held overnight)
            swap_cost = 0.0
            if friction["swap"] > 0 and time is not None:
                # Assume 10% of trades are held overnight
                if np.random.random() < 0.10:
                    swap_cost = friction["swap"] * lot_size

            # Weekend gap impact
            gap_impact = 0.0
            if friction.get("weekend_gaps", False) and time is not None:
                # Check if trade would be held over weekend
                if isinstance(time, pd.Timestamp) and time.dayofweek == 4:  # Friday
                    if np.random.random() < 0.05:  # 5% of Friday trades affected
                        gap_impact = np.random.uniform(-10, 10)  # Random gap impact

            # Latency impact (can cause missed entries or worse fills)
            latency_impact = 0.0
            if friction["latency"] > 0:
                # Higher latency = more slippage
                latency_slip = friction["latency"] / 1000 * np.random.uniform(0, 0.5)
                latency_impact = latency_slip

            # Total costs in pips
            total_cost = spread_cost + slippage_cost + latency_impact

            # Final PnL
            pnl_pips = base_pnl - total_cost + gap_impact
            pnl_dollars = pnl_pips * self.pip_value * lot_size - commission_cost - swap_cost

            # Protect against NaN/Inf
            if not np.isfinite(pnl_dollars):
                pnl_dollars = 0.0

            # Track
            trades_executed += 1
            trade_costs = total_cost * self.pip_value * lot_size + commission_cost + swap_cost
            if np.isfinite(trade_costs):
                total_costs += trade_costs
            total_pnl += pnl_dollars

            # Only count as win/loss if there was actual P&L (not hold trades)
            if base_pnl > 0:
                winning_trades += 1
            elif base_pnl < 0:
                losing_trades += 1

            # Update capital
            capital += pnl_dollars

            # Protect against overflow
            capital = max(min(capital, 1e15), -1e15)

            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            # Check for ruin
            if capital <= 0:
                break

        total_trades = trades_executed + trades_skipped
        total_return = (capital - self.initial_capital) / self.initial_capital
        # Win rate based on trades that actually hit TP or SL (not hold trades)
        decisive_trades = winning_trades + losing_trades
        win_rate = winning_trades / decisive_trades if decisive_trades > 0 else 0

        # Ensure finite values for output
        total_return = total_return if np.isfinite(total_return) else 0.0
        total_costs = total_costs if np.isfinite(total_costs) else 0.0

        return {
            "level": level,
            "level_name": friction["name"],
            "friction_params": friction,
            "total_signals": len(signals_df),
            "trades_executed": trades_executed,
            "trades_skipped": trades_skipped,
            "hold_trades": hold_trades,
            "decisive_trades": decisive_trades,
            "fill_rate_actual": trades_executed / len(signals_df) if len(signals_df) > 0 else 0,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": float(win_rate),
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "final_capital": float(capital),
            "total_costs": float(total_costs),
            "cost_per_trade": float(total_costs / trades_executed) if trades_executed > 0 else 0,
            "profitable": total_return > 0,
        }
    
    def run_reality_gap_test(self) -> Dict:
        """Execute full reality gap testing."""
        print("=" * 70)
        print("REALITY GAP TESTING")
        print("=" * 70)
        print()
        
        df = self.load_test_data()
        model = self.load_model()
        signals_df = self.generate_signals(df, model)
        
        print(f"\n📊 Generated {len(signals_df):,} trading signals")

        # Show label distribution in signals
        if len(signals_df) > 0:
            label_counts = signals_df["label"].value_counts().sort_index()
            print("\n📋 Signal label distribution:")
            for label, count in label_counts.items():
                label_name = {0: "SHORT (SL for LONG)", 1: "HOLD", 2: "LONG (TP for LONG)"}
                print(f"   Label {label} ({label_name.get(label, 'Unknown')}): {count:,} ({count/len(signals_df)*100:.1f}%)")
        print()

        # Run simulations for each friction level
        level_results = []
        
        print("=" * 70)
        print("FRICTION LEVEL ANALYSIS")
        print("=" * 70)
        
        for level in range(8):
            print(f"\n{'─' * 60}")
            print(f"Level {level}: {self.friction_levels[level]['name']}")
            print("─" * 60)
            
            # Run multiple simulations for stochastic elements
            n_sims = 100 if level >= 2 else 1  # More sims for levels with randomness
            sim_results = []
            
            for _ in range(n_sims):
                result = self.simulate_with_friction(signals_df, level)
                sim_results.append(result)
            
            # Aggregate
            if n_sims > 1:
                returns = [r["total_return"] for r in sim_results]
                drawdowns = [r["max_drawdown"] for r in sim_results]
                win_rates = [r["win_rate"] for r in sim_results]
                
                aggregated = {
                    "level": level,
                    "level_name": self.friction_levels[level]["name"],
                    "friction_params": self.friction_levels[level],
                    "n_simulations": n_sims,
                    "return_mean": float(np.mean(returns)),
                    "return_std": float(np.std(returns)),
                    "return_min": float(np.min(returns)),
                    "return_max": float(np.max(returns)),
                    "drawdown_mean": float(np.mean(drawdowns)),
                    "drawdown_max": float(np.max(drawdowns)),
                    "win_rate_mean": float(np.mean(win_rates)),
                    "profitable_simulations": sum(1 for r in returns if r > 0) / n_sims,
                    "total_costs_mean": float(np.mean([r["total_costs"] for r in sim_results])),
                    "hold_trades": int(np.mean([r["hold_trades"] for r in sim_results])),
                    "decisive_trades": int(np.mean([r["decisive_trades"] for r in sim_results])),
                }
            else:
                result = sim_results[0]
                aggregated = {
                    "level": level,
                    "level_name": self.friction_levels[level]["name"],
                    "friction_params": self.friction_levels[level],
                    "n_simulations": 1,
                    "return_mean": result["total_return"],
                    "return_std": 0,
                    "return_min": result["total_return"],
                    "return_max": result["total_return"],
                    "drawdown_mean": result["max_drawdown"],
                    "drawdown_max": result["max_drawdown"],
                    "win_rate_mean": result["win_rate"],
                    "profitable_simulations": 1 if result["total_return"] > 0 else 0,
                    "total_costs_mean": result["total_costs"],
                    "hold_trades": result["hold_trades"],
                    "decisive_trades": result["decisive_trades"],
                }
            
            level_results.append(aggregated)
            
            if n_sims > 1:
                print(f"   Return: {aggregated['return_mean']*100:+.1f}% (±{aggregated['return_std']*100:.1f}%)")
            else:
                print(f"   Return: {aggregated['return_mean']*100:+.1f}%")
            print(f"   Max DD: {aggregated['drawdown_mean']*100:.1f}%")
            print(f"   Win Rate: {aggregated['win_rate_mean']*100:.1f}% (of {aggregated['decisive_trades']:,} decisive trades)")
            print(f"   Hold Trades: {aggregated['hold_trades']:,}")
            print(f"   Avg Costs: ${aggregated['total_costs_mean']:,.2f}")
            if n_sims > 1:
                print(f"   Profitable Sims: {aggregated['profitable_simulations']*100:.0f}%")
        
        # Calculate degradation
        baseline_return = level_results[0]["return_mean"]
        
        print("\n" + "=" * 70)
        print("REALITY GAP SUMMARY")
        print("=" * 70)
        
        print("\n📉 Return Degradation by Level:")
        print("-" * 50)
        print(f"{'Level':<25} {'Return':>10} {'vs Baseline':>12}")
        print("-" * 50)
        
        for result in level_results:
            degradation = (result["return_mean"] - baseline_return) / abs(baseline_return) * 100 if baseline_return != 0 else 0
            print(f"{result['level_name']:<25} {result['return_mean']*100:>+9.1f}% {degradation:>+11.1f}%")
        
        # Find breakeven level
        breakeven_level = None
        for result in level_results:
            if result["return_mean"] <= 0 and breakeven_level is None:
                breakeven_level = result["level"]
        
        results = {
            "run_date": datetime.now().isoformat(),
            "total_signals": len(signals_df),
            "level_results": level_results,
            "baseline_return": float(baseline_return),
            "breakeven_level": breakeven_level,
            "final_level_return": float(level_results[-1]["return_mean"]),
            "final_level_profitable": level_results[-1]["return_mean"] > 0,
        }
        
        # Save results
        output_path = self.results_dir / "reality_gap_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Check success criteria
        self._check_success_criteria(results)
        
        return results
    
    def _check_success_criteria(self, results: Dict):
        """Check if reality gap results meet success criteria."""
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 70)
        
        level_results = results["level_results"]
        
        # Get specific levels
        level_3 = next((r for r in level_results if r["level"] == 3), None)
        level_7 = next((r for r in level_results if r["level"] == 7), None)
        
        criteria = {
            "Baseline (Level 0) profitable": level_results[0]["return_mean"] > 0,
            "With Spread+Slippage+Comm (Level 3) profitable": level_3["return_mean"] > 0 if level_3 else False,
            "Final level (all friction) return > -20%": level_7["return_mean"] > -0.20 if level_7 else False,
            "Win rate remains > 45% at Level 7": level_7["win_rate_mean"] > 0.45 if level_7 else False,
            "Max DD < 40% at any level": all(r["drawdown_max"] < 0.40 for r in level_results),
        }
        
        all_pass = True
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
            if not passed:
                all_pass = False
        
        print()
        if all_pass:
            print("🎉 ALL REALITY GAP CRITERIA PASSED!")
            print("   Strategy is viable for live trading with realistic conditions!")
        else:
            print("⚠️ Some criteria not met - strategy may underperform in live trading")


def main():
    project_root = Path(__file__).parent.parent.parent
    rg = RealityGapTester(project_root)
    results = rg.run_reality_gap_test()
    return results


if __name__ == "__main__":
    main()
