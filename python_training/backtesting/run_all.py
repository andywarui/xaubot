"""
Phase 2 Backtesting Suite Runner for XAUUSD Neural Trading Bot.

Runs all backtesting modules in sequence:
1. Walk-Forward Optimization (WFO)
2. Monte Carlo Simulation
3. Historical Stress Testing
4. Regime-Based Analysis
5. Reality Gap Testing

Usage: python -m python_training.backtesting.run_all
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

from .walk_forward import WalkForwardOptimizer
from .monte_carlo import MonteCarloSimulator
from .stress_test import StressTester
from .regime_analysis import RegimeAnalyzer
from .reality_gap import RealityGapTester


def run_all_backtests(project_root: Path = None) -> Dict:
    """Run complete Phase 2 backtesting suite."""
    
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    results_dir = project_root / "python_training" / "backtesting" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("XAUBOT PHASE 2: COMPREHENSIVE BACKTESTING SUITE")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_results = {}
    all_passed = True
    
    # 1. Walk-Forward Optimization
    print("\n" + "█" * 80)
    print("STEP 1/5: WALK-FORWARD OPTIMIZATION")
    print("█" * 80)
    try:
        wfo = WalkForwardOptimizer(project_root)
        wfo_results = wfo.run_wfo()
        all_results["walk_forward"] = {
            "status": "COMPLETED",
            "wfo_score": wfo_results.get("wfo_score", {}),
        }
    except Exception as e:
        print(f"❌ WFO failed: {e}")
        all_results["walk_forward"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
    
    # 2. Monte Carlo Simulation
    print("\n" + "█" * 80)
    print("STEP 2/5: MONTE CARLO SIMULATION")
    print("█" * 80)
    try:
        mc = MonteCarloSimulator(project_root)
        mc_results = mc.run_full_simulation()
        all_results["monte_carlo"] = {
            "status": "COMPLETED",
            "risk_of_ruin": mc_results.get("methods", {}).get("shuffle", {}).get("risk_of_ruin"),
        }
    except Exception as e:
        print(f"❌ Monte Carlo failed: {e}")
        all_results["monte_carlo"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
    
    # 3. Stress Testing
    print("\n" + "█" * 80)
    print("STEP 3/5: HISTORICAL STRESS TESTING")
    print("█" * 80)
    try:
        st = StressTester(project_root)
        st_results = st.run_stress_test()
        all_results["stress_test"] = {
            "status": "COMPLETED",
            "survival_rate": st_results.get("summary", {}).get("survival_rate"),
        }
    except Exception as e:
        print(f"❌ Stress Test failed: {e}")
        all_results["stress_test"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
    
    # 4. Regime Analysis
    print("\n" + "█" * 80)
    print("STEP 4/5: REGIME-BASED ANALYSIS")
    print("█" * 80)
    try:
        ra = RegimeAnalyzer(project_root)
        ra_results = ra.run_regime_analysis()
        all_results["regime_analysis"] = {
            "status": "COMPLETED",
            "regimes_analyzed": len(ra_results.get("regime_results", {})),
        }
    except Exception as e:
        print(f"❌ Regime Analysis failed: {e}")
        all_results["regime_analysis"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
    
    # 5. Reality Gap Testing
    print("\n" + "█" * 80)
    print("STEP 5/5: REALITY GAP TESTING")
    print("█" * 80)
    try:
        rg = RealityGapTester(project_root)
        rg_results = rg.run_reality_gap_test()
        all_results["reality_gap"] = {
            "status": "COMPLETED",
            "final_level_profitable": rg_results.get("final_level_profitable"),
        }
    except Exception as e:
        print(f"❌ Reality Gap failed: {e}")
        all_results["reality_gap"] = {"status": "FAILED", "error": str(e)}
        all_passed = False
    
    # Final Summary
    print("\n" + "=" * 80)
    print("PHASE 2 BACKTESTING COMPLETE")
    print("=" * 80)
    
    completed = sum(1 for r in all_results.values() if r["status"] == "COMPLETED")
    print(f"\n📊 Tests Completed: {completed}/5")
    
    for test_name, result in all_results.items():
        status = "✅" if result["status"] == "COMPLETED" else "❌"
        print(f"   {status} {test_name.replace('_', ' ').title()}: {result['status']}")
    
    # Save combined results
    combined_results = {
        "run_date": datetime.now().isoformat(),
        "all_passed": all_passed,
        "tests_completed": completed,
        "tests_total": 5,
        "results": all_results,
    }
    
    output_path = results_dir / "phase2_combined_results.json"
    with open(output_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n💾 Combined results saved to: {output_path}")
    
    if all_passed:
        print("\n🎉 PHASE 2 COMPLETE - Ready for Phase 3 (MT5 Integration)!")
    else:
        print("\n⚠️ Some tests failed - review individual results")
    
    return combined_results


def main():
    return run_all_backtests()


if __name__ == "__main__":
    main()
