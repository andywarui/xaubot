"""
Advanced Robustness Testing for XAUUSD Neural Trading Bot.

Implements additional tests for strategy validation:
1. Sensitivity Analysis - Parameter sensitivity testing
2. Synthetic Stress Tests - Artificial extreme scenarios
3. Black Swan Probability - Tail risk analysis

Phase 2.5 Advanced Validation
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


class SensitivityAnalyzer:
    """Analyze strategy sensitivity to parameter changes."""

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

        # Base parameters
        self.base_params = {
            "risk_per_trade": 0.01,
            "tp_pips": 30,
            "sl_pips": 20,
            "spread_pips": 2.5,
        }

        # Parameter ranges to test
        self.param_ranges = {
            "risk_per_trade": [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02],
            "tp_pips": [20, 25, 30, 35, 40],
            "sl_pips": [15, 20, 25, 30],
            "spread_pips": [2.0, 2.5, 3.0, 4.0, 5.0],
        }

    def load_trades(self) -> pd.DataFrame:
        """Load trade results from Monte Carlo."""
        print("📥 Loading trade data for sensitivity analysis...")

        # Load test data
        path = self.data_dir / "hybrid_features_test.parquet"
        df = pd.read_parquet(path)

        # Load model and generate predictions
        model = lgb.Booster(model_file=str(self.models_dir / "lightgbm_balanced_model.txt"))

        X = df[self.feature_cols].values
        proba = model.predict(X)

        # Generate trades based on predictions
        short_thresh = self.thresholds["SHORT"]
        long_thresh = self.thresholds["LONG"]

        trades = []
        for i in range(len(df)):
            p = proba[i]
            label = df["label"].iloc[i]

            if p[0] >= short_thresh and p[0] >= p[2]:
                direction = -1
            elif p[2] >= long_thresh and p[2] > p[0]:
                direction = 1
            else:
                continue

            # Calculate base PnL in pips
            if direction == 1:
                if label == 2:
                    pnl_pips = self.base_params["tp_pips"] - self.base_params["spread_pips"]
                elif label == 0:
                    pnl_pips = -(self.base_params["sl_pips"] + self.base_params["spread_pips"])
                else:
                    pnl_pips = -self.base_params["spread_pips"]
            else:
                if label == 0:
                    pnl_pips = self.base_params["tp_pips"] - self.base_params["spread_pips"]
                elif label == 2:
                    pnl_pips = -(self.base_params["sl_pips"] + self.base_params["spread_pips"])
                else:
                    pnl_pips = -self.base_params["spread_pips"]

            trades.append({
                "direction": direction,
                "pnl_pips": pnl_pips,
                "label": label,
                "is_winner": pnl_pips > 0
            })

        print(f"   Generated {len(trades):,} trades")
        return pd.DataFrame(trades)

    def simulate_with_params(self, trades_df: pd.DataFrame, params: Dict) -> Dict:
        """Simulate equity curve with specific parameters."""
        capital = 10000.0
        peak = capital
        max_drawdown = 0
        pip_value = 10.0
        MAX_CAPITAL = 1e9

        for _, trade in trades_df.iterrows():
            # Recalculate PnL with new TP/SL
            if trade["direction"] == 1:
                if trade["label"] == 2:
                    pnl_pips = params["tp_pips"] - params["spread_pips"]
                elif trade["label"] == 0:
                    pnl_pips = -(params["sl_pips"] + params["spread_pips"])
                else:
                    pnl_pips = -params["spread_pips"]
            else:
                if trade["label"] == 0:
                    pnl_pips = params["tp_pips"] - params["spread_pips"]
                elif trade["label"] == 2:
                    pnl_pips = -(params["sl_pips"] + params["spread_pips"])
                else:
                    pnl_pips = -params["spread_pips"]

            risk_amount = min(capital * params["risk_per_trade"], MAX_CAPITAL * params["risk_per_trade"])
            lot_size = risk_amount / (params["sl_pips"] * pip_value)
            lot_size = min(lot_size, 1e6)
            dollar_pnl = pnl_pips * pip_value * lot_size

            capital += dollar_pnl
            capital = min(capital, MAX_CAPITAL)

            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            if capital <= 0:
                break

        total_return = (capital - 10000.0) / 10000.0
        win_rate = trades_df["is_winner"].mean()

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "final_capital": capital,
            "win_rate": win_rate,
        }

    def run_sensitivity_analysis(self) -> Dict:
        """Run full sensitivity analysis across all parameters."""
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSIS")
        print("=" * 70)

        trades_df = self.load_trades()

        results = {
            "run_date": datetime.now().isoformat(),
            "base_params": self.base_params,
            "sensitivity_results": {},
        }

        for param_name, param_values in self.param_ranges.items():
            print(f"\n📊 Testing {param_name}...")
            param_results = []

            for value in param_values:
                test_params = self.base_params.copy()
                test_params[param_name] = value

                metrics = self.simulate_with_params(trades_df, test_params)
                metrics["param_value"] = value
                param_results.append(metrics)

                print(f"   {param_name}={value}: Return={metrics['total_return']*100:.1f}%, DD={metrics['max_drawdown']*100:.1f}%")

            results["sensitivity_results"][param_name] = param_results

        # Calculate sensitivity scores
        print("\n📈 Sensitivity Summary:")
        sensitivity_scores = {}
        for param_name, param_results in results["sensitivity_results"].items():
            returns = [r["total_return"] for r in param_results]
            drawdowns = [r["max_drawdown"] for r in param_results]

            return_range = max(returns) - min(returns)
            dd_range = max(drawdowns) - min(drawdowns)

            sensitivity_scores[param_name] = {
                "return_sensitivity": return_range,
                "drawdown_sensitivity": dd_range,
                "stability_score": 1.0 / (1.0 + return_range + dd_range)
            }
            print(f"   {param_name}: Return range={return_range*100:.1f}%, DD range={dd_range*100:.1f}%")

        results["sensitivity_scores"] = sensitivity_scores

        # Save results
        output_path = self.results_dir / "sensitivity_analysis_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")

        return results


class SyntheticStressTester:
    """Generate and test synthetic extreme market scenarios."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data" / "processed"
        self.models_dir = project_root / "python_training" / "models"
        self.results_dir = project_root / "python_training" / "backtesting" / "results"

        # Synthetic scenarios
        self.scenarios = [
            {
                "name": "Flash Crash 10%",
                "description": "10% gap down in 1 minute",
                "win_rate_modifier": 0.3,  # Win rate drops to 30%
                "n_trades": 100,
            },
            {
                "name": "Dead Market",
                "description": "30-day ranging, no trend",
                "win_rate_modifier": 0.5,  # Win rate drops to 50%
                "n_trades": 500,
            },
            {
                "name": "Whipsaw",
                "description": "Rapid reversals every few bars",
                "win_rate_modifier": 0.35,
                "n_trades": 200,
            },
            {
                "name": "Black Monday",
                "description": "20% crash in one session",
                "win_rate_modifier": 0.25,
                "n_trades": 50,
            },
            {
                "name": "Gap and Reverse",
                "description": "$50 gap up then immediate $100 drop",
                "win_rate_modifier": 0.2,
                "n_trades": 30,
            },
            {
                "name": "Extended Drawdown",
                "description": "6-month losing streak simulation",
                "win_rate_modifier": 0.45,
                "n_trades": 5000,
            },
        ]

        # Trading parameters
        self.initial_capital = 10000.0
        self.risk_per_trade = 0.01
        self.tp_pips = 30
        self.sl_pips = 20
        self.pip_value = 10.0

    def simulate_scenario(self, scenario: Dict) -> Dict:
        """Simulate a synthetic stress scenario."""
        capital = self.initial_capital
        peak = capital
        max_drawdown = 0
        MAX_CAPITAL = 1e9

        # Generate synthetic trades based on modified win rate
        np.random.seed(42)  # Reproducible
        wins = np.random.random(scenario["n_trades"]) < scenario["win_rate_modifier"]

        for is_win in wins:
            if is_win:
                pnl_pips = self.tp_pips - 2.5  # TP minus spread
            else:
                pnl_pips = -(self.sl_pips + 2.5)  # SL plus spread

            risk_amount = min(capital * self.risk_per_trade, MAX_CAPITAL * self.risk_per_trade)
            lot_size = risk_amount / (self.sl_pips * self.pip_value)
            lot_size = min(lot_size, 1e6)
            dollar_pnl = pnl_pips * self.pip_value * lot_size

            capital += dollar_pnl
            capital = min(capital, MAX_CAPITAL)

            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            if capital <= 0:
                break

        total_return = (capital - self.initial_capital) / self.initial_capital
        survived = capital > 0 and max_drawdown < 0.5  # Survive if DD < 50%

        return {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "win_rate_used": scenario["win_rate_modifier"],
            "n_trades": scenario["n_trades"],
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "final_capital": capital,
            "survived": survived,
        }

    def run_synthetic_stress_tests(self) -> Dict:
        """Run all synthetic stress scenarios."""
        print("\n" + "=" * 70)
        print("SYNTHETIC STRESS TESTS")
        print("=" * 70)

        results = {
            "run_date": datetime.now().isoformat(),
            "scenarios": [],
            "summary": {},
        }

        passed = 0
        for scenario in self.scenarios:
            print(f"\n🔥 Testing: {scenario['name']}...")
            scenario_result = self.simulate_scenario(scenario)
            results["scenarios"].append(scenario_result)

            status = "✅ SURVIVED" if scenario_result["survived"] else "❌ FAILED"
            print(f"   {status}: Return={scenario_result['total_return']*100:.1f}%, DD={scenario_result['max_drawdown']*100:.1f}%")

            if scenario_result["survived"]:
                passed += 1

        results["summary"] = {
            "total_scenarios": len(self.scenarios),
            "passed": passed,
            "failed": len(self.scenarios) - passed,
            "survival_rate": passed / len(self.scenarios),
        }

        print(f"\n📊 Summary: {passed}/{len(self.scenarios)} scenarios survived ({results['summary']['survival_rate']*100:.0f}%)")

        # Save results
        output_path = self.results_dir / "synthetic_stress_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_path}")

        return results


class BlackSwanAnalyzer:
    """Analyze tail risk and black swan probability."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data" / "processed"
        self.models_dir = project_root / "python_training" / "models"
        self.results_dir = project_root / "python_training" / "backtesting" / "results"

        # Parameters
        self.n_simulations = 100000  # 100K paths for tail analysis
        self.initial_capital = 10000.0
        self.risk_per_trade = 0.01
        self.tp_pips = 30
        self.sl_pips = 20
        self.pip_value = 10.0

    def run_black_swan_analysis(self) -> Dict:
        """Run extensive Monte Carlo for tail risk analysis."""
        print("\n" + "=" * 70)
        print("BLACK SWAN PROBABILITY ANALYSIS")
        print("=" * 70)
        print(f"Running {self.n_simulations:,} Monte Carlo paths...")

        # Load actual win rate from previous results
        mc_results_path = self.results_dir / "monte_carlo_results.json"
        if mc_results_path.exists():
            with open(mc_results_path, "r") as f:
                mc_data = json.load(f)
            base_win_rate = mc_data.get("original_win_rate", 0.66)
        else:
            base_win_rate = 0.66

        print(f"Base win rate: {base_win_rate*100:.1f}%")

        # Run simulations
        max_drawdowns = []
        final_capitals = []
        ruin_count = 0

        n_trades_per_sim = 1000  # 1000 trades per simulation
        MAX_CAPITAL = 1e9

        np.random.seed(42)

        for sim in range(self.n_simulations):
            if (sim + 1) % 10000 == 0:
                print(f"   Progress: {sim+1:,}/{self.n_simulations:,}")

            capital = self.initial_capital
            peak = capital
            max_dd = 0

            # Generate random trades
            wins = np.random.random(n_trades_per_sim) < base_win_rate

            for is_win in wins:
                if is_win:
                    pnl_pips = self.tp_pips - 2.5
                else:
                    pnl_pips = -(self.sl_pips + 2.5)

                risk_amount = min(capital * self.risk_per_trade, MAX_CAPITAL * self.risk_per_trade)
                lot_size = risk_amount / (self.sl_pips * self.pip_value)
                lot_size = min(lot_size, 1e6)
                dollar_pnl = pnl_pips * self.pip_value * lot_size

                capital += dollar_pnl
                capital = min(capital, MAX_CAPITAL)

                if capital > peak:
                    peak = capital
                dd = (peak - capital) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                if capital <= 0:
                    ruin_count += 1
                    break

            max_drawdowns.append(max_dd)
            final_capitals.append(capital)

        # Calculate tail statistics
        max_drawdowns = np.array(max_drawdowns)
        final_capitals = np.array(final_capitals)

        results = {
            "run_date": datetime.now().isoformat(),
            "n_simulations": self.n_simulations,
            "n_trades_per_sim": n_trades_per_sim,
            "base_win_rate": base_win_rate,
            "tail_risk": {
                "risk_of_ruin": ruin_count / self.n_simulations,
                "max_dd_percentiles": {
                    "p50": float(np.percentile(max_drawdowns, 50)),
                    "p90": float(np.percentile(max_drawdowns, 90)),
                    "p95": float(np.percentile(max_drawdowns, 95)),
                    "p99": float(np.percentile(max_drawdowns, 99)),
                    "p99.9": float(np.percentile(max_drawdowns, 99.9)),
                    "max": float(np.max(max_drawdowns)),
                },
                "worst_case_capital": {
                    "p1": float(np.percentile(final_capitals, 1)),
                    "p5": float(np.percentile(final_capitals, 5)),
                    "min": float(np.min(final_capitals)),
                },
            },
            "probability_analysis": {
                "prob_dd_over_25pct": float(np.mean(max_drawdowns > 0.25)),
                "prob_dd_over_50pct": float(np.mean(max_drawdowns > 0.50)),
                "prob_dd_over_75pct": float(np.mean(max_drawdowns > 0.75)),
                "prob_loss_over_50pct": float(np.mean(final_capitals < 5000)),
                "prob_total_ruin": float(np.mean(final_capitals <= 0)),
            },
        }

        print("\n📊 Tail Risk Summary:")
        print(f"   Risk of Ruin: {results['tail_risk']['risk_of_ruin']*100:.4f}%")
        print(f"   99th percentile DD: {results['tail_risk']['max_dd_percentiles']['p99']*100:.1f}%")
        print(f"   99.9th percentile DD: {results['tail_risk']['max_dd_percentiles']['p99.9']*100:.1f}%")
        print(f"   Maximum DD observed: {results['tail_risk']['max_dd_percentiles']['max']*100:.1f}%")
        print(f"   Prob of >50% DD: {results['probability_analysis']['prob_dd_over_50pct']*100:.2f}%")

        # Save results
        output_path = self.results_dir / "black_swan_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")

        return results


def run_all_robustness_tests(project_root: Path) -> Dict:
    """Run all advanced robustness tests."""
    print("\n" + "=" * 70)
    print("ADVANCED ROBUSTNESS TESTING SUITE")
    print("=" * 70)

    results = {}

    # 1. Sensitivity Analysis
    print("\n[1/3] Running Sensitivity Analysis...")
    sensitivity = SensitivityAnalyzer(project_root)
    results["sensitivity"] = sensitivity.run_sensitivity_analysis()

    # 2. Synthetic Stress Tests
    print("\n[2/3] Running Synthetic Stress Tests...")
    synthetic = SyntheticStressTester(project_root)
    results["synthetic_stress"] = synthetic.run_synthetic_stress_tests()

    # 3. Black Swan Analysis
    print("\n[3/3] Running Black Swan Analysis...")
    black_swan = BlackSwanAnalyzer(project_root)
    results["black_swan"] = black_swan.run_black_swan_analysis()

    print("\n" + "=" * 70)
    print("ROBUSTNESS TESTING COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    run_all_robustness_tests(project_root)
