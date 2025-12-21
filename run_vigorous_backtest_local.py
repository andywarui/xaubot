#!/usr/bin/env python3
"""
Vigorous Backtesting Suite for XAUBOT

A comprehensive backtesting runner that tests all aspects of the trading strategy:
1. Walk-Forward Optimization (WFO) - validates model robustness across time
2. Monte Carlo Simulation - tests statistical significance
3. Historical Stress Testing - validates against market events
4. Robustness Tests - sensitivity, regime analysis, reality gap

Usage:
    python run_vigorous_backtest.py
    python run_vigorous_backtest.py --quick    # Skip slow tests
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(title: str, step: int = None, total: int = None):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    if step and total:
        print(f"  {step}/{total} - {title}")
    else:
        print(f"  {title}")
    print("=" * 70)
    print()


def run_walk_forward(project_root: Path, n_folds: int = 7) -> Dict:
    """Run Walk-Forward Optimization."""
    from python_training.backtesting.walk_forward import WalkForwardOptimizer
    
    wfo = WalkForwardOptimizer(project_root)
    
    # Limit folds if requested
    if n_folds < len(wfo.folds):
        wfo.folds = wfo.folds[:n_folds]
    
    results = wfo.run_wfo()
    return results


def run_monte_carlo(project_root: Path, n_simulations: int = 5000) -> Dict:
    """Run Monte Carlo Simulation."""
    from python_training.backtesting.monte_carlo import MonteCarloSimulator
    
    mc = MonteCarloSimulator(project_root)
    mc.n_simulations = n_simulations
    
    results = mc.run_full_simulation()
    return results


def run_stress_test(project_root: Path) -> Dict:
    """Run Historical Stress Testing."""
    from python_training.backtesting.stress_test import StressTester
    
    st = StressTester(project_root)
    results = st.run_stress_test()
    return results


def run_regime_analysis(project_root: Path) -> Dict:
    """Run Regime-Based Analysis."""
    from python_training.backtesting.regime_analysis import RegimeAnalyzer
    
    ra = RegimeAnalyzer(project_root)
    results = ra.run_regime_analysis()
    return results


def run_reality_gap(project_root: Path) -> Dict:
    """Run Reality Gap Testing."""
    from python_training.backtesting.reality_gap import RealityGapTester
    
    rg = RealityGapTester(project_root)
    results = rg.run_reality_gap_test()
    return results


def main():
    """Main entry point for vigorous backtesting."""
    parser = argparse.ArgumentParser(description="XAUBOT Vigorous Backtesting Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--wfo-folds", type=int, default=7, help="Number of WFO folds")
    parser.add_argument("--mc-paths", type=int, default=5000, help="Monte Carlo simulation paths")
    parser.add_argument("--skip-wfo", action="store_true", help="Skip WFO")
    parser.add_argument("--skip-mc", action="store_true", help="Skip Monte Carlo")
    parser.add_argument("--skip-stress", action="store_true", help="Skip Stress Test")
    parser.add_argument("--skip-regime", action="store_true", help="Skip Regime Analysis")
    parser.add_argument("--skip-reality", action="store_true", help="Skip Reality Gap")
    args = parser.parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.wfo_folds = 3
        args.mc_paths = 1000
    
    print_header("VIGOROUS BACKTESTING SUITE - XAUBOT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    
    results = {}
    start_time = datetime.now()
    
    # ========== 1. Walk-Forward Optimization ==========
    if not args.skip_wfo:
        print_header(f"WALK-FORWARD OPTIMIZATION ({args.wfo_folds} folds)", 1, 5)
        step_start = datetime.now()
        try:
            wfo_results = run_walk_forward(project_root, args.wfo_folds)
            results["walk_forward"] = {
                "status": "PASSED",
                "wfo_score": wfo_results.get("wfo_score", {}),
                "folds_completed": len(wfo_results.get("fold_results", [])),
            }
            print(f"✅ WFO Complete: {results['walk_forward']['folds_completed']} folds")
        except Exception as e:
            results["walk_forward"] = {"status": "ERROR", "error": str(e)}
            print(f"❌ WFO Error: {e}")
        print(f"⏱️  Elapsed: {(datetime.now() - step_start).total_seconds() / 60:.1f} minutes")
    
    # ========== 2. Monte Carlo Simulation ==========
    if not args.skip_mc:
        print_header(f"MONTE CARLO SIMULATION ({args.mc_paths:,} paths)", 2, 5)
        step_start = datetime.now()
        try:
            mc_results = run_monte_carlo(project_root, args.mc_paths)
            results["monte_carlo"] = {
                "status": "PASSED",
                "risk_of_ruin": mc_results.get("methods", {}).get("shuffle", {}).get("risk_of_ruin"),
                "median_return": mc_results.get("methods", {}).get("shuffle", {}).get("median_return"),
            }
            print(f"✅ Monte Carlo Complete")
            print(f"   Risk of Ruin: {results['monte_carlo'].get('risk_of_ruin', 'N/A')}")
        except Exception as e:
            results["monte_carlo"] = {"status": "ERROR", "error": str(e)}
            print(f"❌ Monte Carlo Error: {e}")
        print(f"⏱️  Elapsed: {(datetime.now() - step_start).total_seconds() / 60:.1f} minutes")
    
    # ========== 3. Historical Stress Test ==========
    if not args.skip_stress:
        print_header("HISTORICAL STRESS TEST", 3, 5)
        step_start = datetime.now()
        try:
            stress_results = run_stress_test(project_root)
            summary = stress_results.get("summary", {})
            results["stress_test"] = {
                "status": "PASSED" if summary.get("survival_rate", 0) >= 0.75 else "FAILED",
                "survival_rate": summary.get("survival_rate"),
                "events_passed": summary.get("passed_count"),
                "events_total": summary.get("total_events"),
            }
            print(f"✅ Stress Test Complete: {results['stress_test']['events_passed']}/{results['stress_test']['events_total']} passed ({results['stress_test']['survival_rate']*100:.0f}% survival)")
        except Exception as e:
            results["stress_test"] = {"status": "ERROR", "error": str(e)}
            print(f"❌ Stress Test Error: {e}")
        print(f"⏱️  Elapsed: {(datetime.now() - step_start).total_seconds() / 60:.1f} minutes")
    
    # ========== 4. Regime Analysis ==========
    if not args.skip_regime:
        print_header("REGIME ANALYSIS", 4, 5)
        step_start = datetime.now()
        try:
            regime_results = run_regime_analysis(project_root)
            results["regime_analysis"] = {
                "status": "PASSED",
                "regimes_analyzed": len(regime_results.get("regime_results", {})),
            }
            print(f"✅ Regime Analysis Complete: {results['regime_analysis']['regimes_analyzed']} regimes")
        except Exception as e:
            results["regime_analysis"] = {"status": "ERROR", "error": str(e)}
            print(f"❌ Regime Analysis Error: {e}")
        print(f"⏱️  Elapsed: {(datetime.now() - step_start).total_seconds() / 60:.1f} minutes")
    
    # ========== 5. Reality Gap ==========
    if not args.skip_reality:
        print_header("REALITY GAP TEST", 5, 5)
        step_start = datetime.now()
        try:
            reality_results = run_reality_gap(project_root)
            results["reality_gap"] = {
                "status": "PASSED" if reality_results.get("final_level_profitable") else "FAILED",
                "final_profitable": reality_results.get("final_level_profitable"),
            }
            print(f"✅ Reality Gap Complete: Final profitable = {results['reality_gap']['final_profitable']}")
        except Exception as e:
            results["reality_gap"] = {"status": "ERROR", "error": str(e)}
            print(f"❌ Reality Gap Error: {e}")
        print(f"⏱️  Elapsed: {(datetime.now() - step_start).total_seconds() / 60:.1f} minutes")
    
    # ========== Final Summary ==========
    print_header("FINAL SUMMARY")
    
    total_time = (datetime.now() - start_time).total_seconds() / 60
    
    print("📊 VIGOROUS BACKTEST RESULTS")
    print("-" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = result.get("status", "UNKNOWN")
        if status == "PASSED":
            icon = "✅"
        elif status == "ERROR":
            icon = "❌"
            all_passed = False
        elif status == "FAILED":
            icon = "❌"
            all_passed = False
        else:
            icon = "⚠️"
        print(f"   {icon} {test_name}: {status}")
    
    print("-" * 50)
    print(f"⏱️  Total Time: {total_time:.1f} minutes")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Save results
    output_path = project_root / "python_training" / "backtesting" / "results" / "vigorous_backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "all_passed": all_passed,
            "total_time_minutes": total_time,
            "results": results,
        }, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {output_path}")
    print()
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("⚠️ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
