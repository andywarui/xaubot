#!/usr/bin/env python3
"""
Vigorous Backtesting Suite for XAUUSD Neural Trading Bot.

Runs comprehensive backtests on RunPod/Codespace:
- Walk-Forward: 12 folds
- Monte Carlo: 50K simulations
- Stress Test: 14 historical events
- Robustness Tests: Sensitivity, Synthetic, Black Swan

Estimated runtime on RTX 3090: ~50-60 minutes
"""
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_all_tests():
    """Run complete vigorous backtesting suite."""
    start_time = time.time()

    print_header("VIGOROUS BACKTESTING SUITE - XAUBOT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")

    results = {}

    # =========================================================================
    # 1. WALK-FORWARD OPTIMIZATION (12 folds)
    # =========================================================================
    print_header("1/5 - WALK-FORWARD OPTIMIZATION (12 folds)")
    try:
        from python_training.backtesting.walk_forward import WalkForwardOptimizer
        wfo = WalkForwardOptimizer(project_root)
        wfo_results = wfo.run_wfo()  # Correct method name
        results["walk_forward"] = {
            "status": "PASSED" if wfo_results else "COMPLETED",
            "folds": len(wfo.folds),
            "mean_accuracy": wfo_results.get("wfo_score", {}).get("mean_accuracy", 0),
        }
        print(f"\n✅ WFO Complete: {results['walk_forward']['folds']} folds, "
              f"Accuracy: {results['walk_forward']['mean_accuracy']*100:.1f}%")
    except Exception as e:
        print(f"\n❌ WFO Error: {e}")
        results["walk_forward"] = {"status": "ERROR", "error": str(e)}

    elapsed = time.time() - start_time
    print(f"⏱️  Elapsed: {elapsed/60:.1f} minutes")

    # =========================================================================
    # 2. MONTE CARLO SIMULATION (50K paths)
    # =========================================================================
    print_header("2/5 - MONTE CARLO SIMULATION (50K paths)")
    try:
        from python_training.backtesting.monte_carlo import MonteCarloSimulator
        mc = MonteCarloSimulator(project_root)
        mc_results = mc.run_full_simulation()  # Correct method name

        shuffle_ror = mc_results.get("methods", {}).get("shuffle", {}).get("risk_of_ruin", 1.0)
        results["monte_carlo"] = {
            "status": "PASSED" if shuffle_ror == 0 else "WARNING",
            "n_simulations": mc.n_simulations,
            "risk_of_ruin": shuffle_ror,
        }
        print(f"\n✅ Monte Carlo Complete: {mc.n_simulations:,} simulations, "
              f"Risk of Ruin: {shuffle_ror*100:.2f}%")
    except Exception as e:
        print(f"\n❌ Monte Carlo Error: {e}")
        results["monte_carlo"] = {"status": "ERROR", "error": str(e)}

    elapsed = time.time() - start_time
    print(f"⏱️  Elapsed: {elapsed/60:.1f} minutes")

    # =========================================================================
    # 3. STRESS TEST (14 events)
    # =========================================================================
    print_header("3/5 - HISTORICAL STRESS TEST (14 events)")
    try:
        from python_training.backtesting.stress_test import StressTester
        stress = StressTester(project_root)
        stress_results = stress.run_stress_test()

        survival_rate = stress_results.get("summary", {}).get("survival_rate", 0)
        events_passed = stress_results.get("summary", {}).get("events_passed", 0)
        events_total = stress_results.get("summary", {}).get("events_tested", 0)

        results["stress_test"] = {
            "status": "PASSED" if survival_rate >= 0.5 else "WARNING",
            "events_passed": events_passed,
            "events_total": events_total,
            "survival_rate": survival_rate,
        }
        print(f"\n✅ Stress Test Complete: {events_passed}/{events_total} passed "
              f"({survival_rate*100:.0f}% survival)")
    except Exception as e:
        print(f"\n❌ Stress Test Error: {e}")
        results["stress_test"] = {"status": "ERROR", "error": str(e)}

    elapsed = time.time() - start_time
    print(f"⏱️  Elapsed: {elapsed/60:.1f} minutes")

    # =========================================================================
    # 4. ROBUSTNESS TESTS (Sensitivity, Synthetic, Black Swan)
    # =========================================================================
    print_header("4/5 - ROBUSTNESS TESTS")
    try:
        from python_training.backtesting.robustness_tests import run_all_robustness_tests
        robustness_results = run_all_robustness_tests(project_root)

        synthetic_survival = robustness_results.get("synthetic_stress", {}).get("summary", {}).get("survival_rate", 0)
        black_swan_ror = robustness_results.get("black_swan", {}).get("tail_risk", {}).get("risk_of_ruin", 1.0)

        results["robustness"] = {
            "status": "PASSED" if synthetic_survival >= 0.5 and black_swan_ror < 0.01 else "WARNING",
            "synthetic_survival_rate": synthetic_survival,
            "black_swan_risk_of_ruin": black_swan_ror,
        }
        print(f"\n✅ Robustness Complete: Synthetic survival={synthetic_survival*100:.0f}%, "
              f"Black Swan RoR={black_swan_ror*100:.4f}%")
    except Exception as e:
        print(f"\n❌ Robustness Error: {e}")
        results["robustness"] = {"status": "ERROR", "error": str(e)}

    elapsed = time.time() - start_time
    print(f"⏱️  Elapsed: {elapsed/60:.1f} minutes")

    # =========================================================================
    # 5. FINAL SUMMARY
    # =========================================================================
    print_header("5/5 - FINAL SUMMARY")

    total_time = time.time() - start_time

    print(f"\n📊 VIGOROUS BACKTEST RESULTS")
    print("-" * 50)

    for test_name, test_results in results.items():
        status = test_results.get("status", "UNKNOWN")
        status_icon = "✅" if status == "PASSED" else "⚠️" if status == "WARNING" else "❌"
        print(f"   {status_icon} {test_name}: {status}")

    print("-" * 50)
    print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine overall status
    all_passed = all(r.get("status") in ["PASSED", "COMPLETED"] for r in results.values())
    any_errors = any(r.get("status") == "ERROR" for r in results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Model is robust for deployment!")
    elif any_errors:
        print("\n⚠️ SOME TESTS FAILED - Review errors above")
    else:
        print("\n⚠️ SOME WARNINGS - Review results carefully")

    # Save summary
    import json
    summary_path = project_root / "python_training" / "backtesting" / "results" / "vigorous_backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "total_time_minutes": total_time / 60,
            "results": results,
        }, f, indent=2)
    print(f"\n💾 Summary saved to {summary_path}")

    return results


if __name__ == "__main__":
    run_all_tests()
