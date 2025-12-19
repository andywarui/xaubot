"""
Run all Phase 2 backtests sequentially.
Designed for local execution with optimized parameters.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_backtest(module_name: str) -> bool:
    """Run a single backtest module."""
    print(f"\n{'='*70}")
    print(f"STARTING: {module_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, "-m", f"python_training.backtesting.{module_name}"],
        cwd=Path(__file__).parent
    )
    
    success = result.returncode == 0
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\n{status}: {module_name}")
    return success

def main():
    print("\n" + "="*70)
    print("XAUBOT PHASE 2: COMPREHENSIVE BACKTESTING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    backtests = [
        "monte_carlo",
        "stress_test", 
        "regime_analysis",
        "reality_gap"
    ]
    
    results = {}
    
    for module in backtests:
        results[module] = run_backtest(module)
    
    # Summary
    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for module, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {module}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
