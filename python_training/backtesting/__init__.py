"""
Phase 2: Comprehensive Backtesting Module for XAUUSD Neural Trading Bot.

Modules:
- walk_forward: Walk-Forward Optimization (7 folds, 2019-2024)
- monte_carlo: Monte Carlo Simulation (10,000 paths)
- stress_test: Historical Stress Testing (8 major events)
- regime_analysis: Regime-Based Performance Analysis
- reality_gap: Reality Gap Testing (8 friction levels)
- run_all: Complete backtesting suite runner

Usage:
    # Run all tests
    python -m python_training.backtesting.run_all
    
    # Run individual tests
    python -m python_training.backtesting.walk_forward
    python -m python_training.backtesting.monte_carlo
    python -m python_training.backtesting.stress_test
    python -m python_training.backtesting.regime_analysis
    python -m python_training.backtesting.reality_gap
"""

from .walk_forward import WalkForwardOptimizer
from .monte_carlo import MonteCarloSimulator
from .stress_test import StressTester
from .regime_analysis import RegimeAnalyzer
from .reality_gap import RealityGapTester
from .run_all import run_all_backtests

__all__ = [
    "WalkForwardOptimizer",
    "MonteCarloSimulator", 
    "StressTester",
    "RegimeAnalyzer",
    "RealityGapTester",
    "run_all_backtests",
]
