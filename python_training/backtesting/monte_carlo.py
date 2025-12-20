"""
Monte Carlo Simulation for XAUUSD Neural Trading Bot.

Implements 10,000 simulation paths using:
- Trade shuffling (random order)
- Bootstrap resampling (with replacement)
- Return perturbation (±5% noise)
- Drawdown path simulation

Phase 2.3 Implementation per XAUBOT_DEVELOPMENT_PLAN.md
"""
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """Monte Carlo simulation for trading strategy validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data" / "processed"
        self.models_dir = project_root / "python_training" / "models"
        self.results_dir = project_root / "python_training" / "backtesting" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulation parameters (5K paths - good balance for local execution)
        self.n_simulations = 5000
        self.confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
        
        # Trading parameters
        self.initial_capital = 10000.0
        self.risk_per_trade = 0.01  # 1% risk (reduced from 2% for better drawdown control)
        self.tp_pips = 30
        self.sl_pips = 20
        self.spread_pips = 2.5
        self.pip_value = 10.0  # $10 per pip per lot
        
        # Load model config
        with open(self.models_dir / "lightgbm_balanced_config.json", "r") as f:
            self.config = json.load(f)
        
        self.feature_cols = self.config["feature_cols"]
        self.thresholds = self.config["thresholds"]
        
    def load_test_data(self) -> pd.DataFrame:
        """Load test data for backtesting."""
        print("📥 Loading test data...")
        
        path = self.data_dir / "hybrid_features_test.parquet"
        df = pd.read_parquet(path)
        
        # Ensure time column
        if "time" not in df.columns and "close_time" in df.columns:
            df["time"] = pd.to_datetime(df["close_time"])
        elif "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        
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
    
    def generate_trades(self, df: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:
        """Generate trade signals from model predictions."""
        print("🔮 Generating trade signals...")
        
        X = df[self.feature_cols].values
        proba = model.predict(X)
        
        # Apply thresholds
        short_thresh = self.thresholds.get("SHORT", 0.48)
        long_thresh = self.thresholds.get("LONG", 0.40)
        
        trades = []
        for i in range(len(df)):
            p = proba[i]
            
            if p[0] >= short_thresh and p[0] >= p[2]:
                direction = -1  # SHORT
                confidence = p[0]
            elif p[2] >= long_thresh and p[2] > p[0]:
                direction = 1  # LONG
                confidence = p[2]
            else:
                continue  # HOLD - no trade
            
            # Simulate trade outcome based on label
            label = df["label"].iloc[i]
            
            # Calculate PnL
            if direction == 1:  # LONG
                if label == 2:  # Correct - hit TP
                    pnl_pips = self.tp_pips - self.spread_pips
                elif label == 0:  # Wrong - hit SL
                    pnl_pips = -(self.sl_pips + self.spread_pips)
                else:  # HOLD label - breakeven
                    pnl_pips = -self.spread_pips
            else:  # SHORT
                if label == 0:  # Correct - hit TP
                    pnl_pips = self.tp_pips - self.spread_pips
                elif label == 2:  # Wrong - hit SL
                    pnl_pips = -(self.sl_pips + self.spread_pips)
                else:  # HOLD label - breakeven
                    pnl_pips = -self.spread_pips
            
            trades.append({
                "idx": i,
                "direction": direction,
                "confidence": confidence,
                "label": label,
                "pnl_pips": pnl_pips,
                "is_winner": pnl_pips > 0,
            })
        
        trades_df = pd.DataFrame(trades)
        print(f"   Generated: {len(trades_df):,} trades")
        print(f"   Win rate: {trades_df['is_winner'].mean()*100:.1f}%")
        
        return trades_df
    
    def simulate_equity_curve(self, trades_df: pd.DataFrame,
                               shuffle: bool = False,
                               bootstrap: bool = False,
                               noise: float = 0.0) -> Dict:
        """Simulate single equity curve path."""

        if bootstrap:
            # Sample with replacement
            trades = trades_df.sample(n=len(trades_df), replace=True)
        elif shuffle:
            # Random order
            trades = trades_df.sample(frac=1.0)
        else:
            # Original order
            trades = trades_df

        pnl_pips = trades["pnl_pips"].values.copy()

        # Add noise if specified
        if noise > 0:
            noise_factor = 1 + np.random.uniform(-noise, noise, len(pnl_pips))
            pnl_pips = pnl_pips * noise_factor

        # Calculate equity curve with overflow protection
        capital = self.initial_capital
        equity_curve = [capital]
        peak = capital
        max_drawdown = 0

        # Cap maximum capital to prevent overflow (1 billion max)
        MAX_CAPITAL = 1e9

        for pnl in pnl_pips:
            # Position size based on risk with cap
            risk_amount = min(capital * self.risk_per_trade, MAX_CAPITAL * self.risk_per_trade)
            lot_size = risk_amount / (self.sl_pips * self.pip_value)
            lot_size = min(lot_size, 1e6)  # Cap lot size

            # Calculate dollar PnL
            dollar_pnl = pnl * self.pip_value * lot_size

            # Protect against overflow
            if not np.isfinite(dollar_pnl):
                dollar_pnl = 0

            capital += dollar_pnl
            capital = min(capital, MAX_CAPITAL)  # Cap capital growth
            equity_curve.append(capital)

            # Track drawdown
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            # Stop if account blown
            if capital <= 0:
                break
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        returns = returns[np.isfinite(returns)]
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized
        else:
            sharpe = 0
        
        winning_trades = (trades["pnl_pips"] > 0).sum()
        win_rate = winning_trades / len(trades) if len(trades) > 0 else 0
        
        gross_profit = trades[trades["pnl_pips"] > 0]["pnl_pips"].sum()
        gross_loss = abs(trades[trades["pnl_pips"] < 0]["pnl_pips"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "recovery_factor": recovery_factor,
            "final_capital": capital,
        }
    
    def run_monte_carlo(self, trades_df: pd.DataFrame, method: str = "shuffle") -> Dict:
        """Run Monte Carlo simulation with specified method."""
        print(f"\n🎲 Running Monte Carlo ({method})...")
        
        results = []
        
        for i in range(self.n_simulations):
            if (i + 1) % 1000 == 0:
                print(f"   Progress: {i+1:,}/{self.n_simulations:,}")
            
            if method == "shuffle":
                sim = self.simulate_equity_curve(trades_df, shuffle=True)
            elif method == "bootstrap":
                sim = self.simulate_equity_curve(trades_df, bootstrap=True)
            elif method == "noise":
                sim = self.simulate_equity_curve(trades_df, noise=0.05)
            else:
                sim = self.simulate_equity_curve(trades_df)
            
            results.append(sim)
        
        results_df = pd.DataFrame(results)
        
        # Calculate percentiles
        metrics_summary = {}
        for metric in results_df.columns:
            values = results_df[metric].values
            metrics_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "percentiles": {
                    f"p{int(p*100)}": float(np.percentile(values, p*100)) 
                    for p in self.confidence_levels
                }
            }
        
        # Risk of ruin calculation
        ruin_threshold = 0.50  # 50% drawdown
        risk_of_ruin = (results_df["max_drawdown"] > ruin_threshold).mean()
        
        return {
            "method": method,
            "n_simulations": self.n_simulations,
            "metrics": metrics_summary,
            "risk_of_ruin": float(risk_of_ruin),
        }
    
    def run_full_simulation(self) -> Dict:
        """Execute complete Monte Carlo analysis."""
        print("=" * 70)
        print("MONTE CARLO SIMULATION")
        print("=" * 70)
        print(f"\nParameters:")
        print(f"   Simulations: {self.n_simulations:,}")
        print(f"   Initial Capital: ${self.initial_capital:,.0f}")
        print(f"   Risk per Trade: {self.risk_per_trade*100:.1f}%")
        print(f"   TP: {self.tp_pips} pips | SL: {self.sl_pips} pips")
        print()
        
        # Load data and model
        df = self.load_test_data()
        model = self.load_model()
        
        # Generate trades
        trades_df = self.generate_trades(df, model)
        
        # Run different methods
        methods = ["shuffle", "bootstrap", "noise"]
        all_results = {}
        
        for method in methods:
            all_results[method] = self.run_monte_carlo(trades_df, method)
        
        # Compile final results
        results = {
            "run_date": datetime.now().isoformat(),
            "n_trades": len(trades_df),
            "original_win_rate": float(trades_df["is_winner"].mean()),
            "methods": all_results,
            "parameters": {
                "n_simulations": self.n_simulations,
                "initial_capital": self.initial_capital,
                "risk_per_trade": self.risk_per_trade,
                "tp_pips": self.tp_pips,
                "sl_pips": self.sl_pips,
            }
        }
        
        # Print summary
        print("\n" + "=" * 70)
        print("MONTE CARLO RESULTS SUMMARY")
        print("=" * 70)
        
        for method, method_results in all_results.items():
            print(f"\n📊 {method.upper()} Method:")
            metrics = method_results["metrics"]
            
            print(f"   Total Return: {metrics['total_return']['mean']*100:.1f}% "
                  f"[{metrics['total_return']['percentiles']['p5']*100:.1f}%, "
                  f"{metrics['total_return']['percentiles']['p95']*100:.1f}%]")
            print(f"   Max Drawdown: {metrics['max_drawdown']['mean']*100:.1f}% "
                  f"[{metrics['max_drawdown']['percentiles']['p5']*100:.1f}%, "
                  f"{metrics['max_drawdown']['percentiles']['p95']*100:.1f}%]")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']['mean']:.2f} "
                  f"[{metrics['sharpe_ratio']['percentiles']['p5']:.2f}, "
                  f"{metrics['sharpe_ratio']['percentiles']['p95']:.2f}]")
            print(f"   Win Rate: {metrics['win_rate']['mean']*100:.1f}%")
            print(f"   Profit Factor: {metrics['profit_factor']['mean']:.2f}")
            print(f"   Risk of Ruin (>50% DD): {method_results['risk_of_ruin']*100:.2f}%")
        
        # Save results
        output_path = self.results_dir / "monte_carlo_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Check success criteria
        self._check_success_criteria(results)
        
        return results
    
    def _check_success_criteria(self, results: Dict):
        """Check if Monte Carlo results meet Phase 2 criteria."""
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 70)
        
        # Use shuffle method as primary
        shuffle = results["methods"]["shuffle"]["metrics"]
        
        criteria = {
            "Mean Return > 0%": shuffle["total_return"]["mean"] > 0,
            "5th percentile Return > -20%": shuffle["total_return"]["percentiles"]["p5"] > -0.20,
            "Mean Drawdown < 30%": shuffle["max_drawdown"]["mean"] < 0.30,
            "95th percentile Drawdown < 50%": shuffle["max_drawdown"]["percentiles"]["p95"] < 0.50,
            "Mean Sharpe > 0.5": shuffle["sharpe_ratio"]["mean"] > 0.5,
            "Risk of Ruin < 5%": results["methods"]["shuffle"]["risk_of_ruin"] < 0.05,
            "Profit Factor > 1.0": shuffle["profit_factor"]["mean"] > 1.0,
        }
        
        all_pass = True
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
            if not passed:
                all_pass = False
        
        print()
        if all_pass:
            print("🎉 ALL MONTE CARLO CRITERIA PASSED!")
        else:
            print("⚠️ Some criteria not met - review results")


def main():
    project_root = Path(__file__).parent.parent.parent
    mc = MonteCarloSimulator(project_root)
    results = mc.run_full_simulation()
    return results


if __name__ == "__main__":
    main()
