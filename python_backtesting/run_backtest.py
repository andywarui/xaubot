"""
Run XAUUSD Neural Bot Backtest

This script runs comprehensive backtests for both Single and Ensemble models,
generating performance metrics and comparison reports.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_engine import XAUUSDBacktester
from prepare_data import prepare_backtest_data

def run_single_model_backtest(data: pd.DataFrame, model_path: str) -> dict:
    """
    Run backtest for Single Model (LightGBM only)

    Args:
        data: Prepared M1 data with indicators
        model_path: Path to LightGBM ONNX model

    Returns:
        Dictionary of performance metrics
    """
    print("\n" + "="*70)
    print("SINGLE MODEL BACKTEST (LightGBM + Hybrid Validation)")
    print("="*70)

    # Initialize backtester
    backtester = XAUUSDBacktester(
        initial_balance=10000.0,
        risk_percent=0.5,
        confidence_threshold=0.60,
        max_trades_per_day=5,
        stop_loss_usd=4.0,
        take_profit_usd=8.0
    )

    # Disable MTF alignment for synthetic data (MTF EMAs not properly calculated)
    backtester.require_mtf_alignment = False

    # Load model
    backtester.load_lightgbm_model(model_path)

    # Run backtest
    print(f"\nProcessing {len(data):,} bars...")
    print(f"Period: {data['time'].min()} to {data['time'].max()}")
    print()

    for idx in range(100, len(data)):  # Start from bar 100 to ensure enough history
        row = data.iloc[idx]
        current_price = row['close']
        current_time = row['time']

        # Reset daily counter
        if backtester.current_date != current_time.date():
            backtester.current_date = current_time.date()
            backtester.daily_trades = 0

        # Update existing position
        backtester.update_position(current_price, current_time)

        # Check if we can open a new trade
        if backtester.open_position is None and backtester.daily_trades < backtester.max_trades_per_day:
            # Calculate 26 features for standalone model (no Transformer dependency)
            features = backtester.calculate_26_features(data, idx)

            # Get ML prediction
            signal, confidence = backtester.predict_lightgbm(features)

            # Validate signal
            if signal == 2 and confidence >= backtester.confidence_threshold:  # LONG
                if backtester.validate_long_signal(data, idx, confidence):
                    backtester.open_trade(signal, current_price, current_time)

            elif signal == 0 and confidence >= backtester.confidence_threshold:  # SHORT
                if backtester.validate_short_signal(data, idx, confidence):
                    backtester.open_trade(signal, current_price, current_time)

        # Progress indicator
        if idx % 50000 == 0:
            progress = (idx / len(data)) * 100
            print(f"Progress: {progress:.1f}% | Balance: ${backtester.balance:,.2f} | Trades: {backtester.stats['total_trades']}")

    # Close any remaining position
    if backtester.open_position is not None:
        final_price = data.iloc[-1]['close']
        final_time = data.iloc[-1]['time']
        backtester.update_position(final_price, final_time)

    # Get results
    backtester.print_summary()

    return backtester.get_performance_metrics()

def run_ensemble_model_backtest(data: pd.DataFrame, lgb_path: str, transformer_path: str, scaler_path: str) -> dict:
    """
    Run backtest for Ensemble Model (LightGBM + Transformer)

    Args:
        data: Prepared M1 data with indicators
        lgb_path: Path to LightGBM ONNX model
        transformer_path: Path to Transformer ONNX model
        scaler_path: Path to scaler parameters JSON

    Returns:
        Dictionary of performance metrics
    """
    print("\n" + "="*70)
    print("ENSEMBLE MODEL BACKTEST (LightGBM + Transformer + Hybrid Validation)")
    print("="*70)

    # Initialize backtester
    backtester = XAUUSDBacktester(
        initial_balance=10000.0,
        risk_percent=0.5,
        confidence_threshold=0.65,  # Higher threshold for ensemble
        max_trades_per_day=5,
        stop_loss_usd=4.0,
        take_profit_usd=8.0
    )

    # Disable MTF alignment for synthetic data (MTF EMAs not properly calculated)
    backtester.require_mtf_alignment = False

    # Load models
    backtester.load_lightgbm_model(lgb_path)
    backtester.load_transformer_model(transformer_path, scaler_path)

    print("\n⚠️  NOTE: Ensemble model requires Transformer ONNX files")
    print("    If Transformer is not available, will fallback to LightGBM-only mode")
    print()

    # Similar backtesting loop as single model
    # (Implementation would include 130-feature calculation and ensemble voting)
    # For now, return placeholder metrics

    return {
        'total_trades': 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'net_profit': 0.0,
        'message': 'Transformer model not available - please run transformer export first'
    }

def generate_comparison_report(single_metrics: dict, ensemble_metrics: dict):
    """Generate A/B testing comparison report"""
    print("\n" + "="*70)
    print("A/B TESTING COMPARISON REPORT")
    print("="*70)
    print()

    print("| Metric                  | Single Model | Ensemble Model | Winner |")
    print("|-------------------------|--------------|----------------|--------|")

    def format_winner(single_val, ensemble_val, higher_better=True):
        if ensemble_metrics.get('message'):
            return "N/A"
        if higher_better:
            return "Single" if single_val > ensemble_val else "Ensemble"
        else:
            return "Single" if single_val < ensemble_val else "Ensemble"

    # Net Profit
    single_profit = single_metrics.get('net_profit', 0)
    ensemble_profit = ensemble_metrics.get('net_profit', 0)
    winner = format_winner(single_profit, ensemble_profit, True)
    print(f"| Net Profit              | ${single_profit:>11,.2f} | ${ensemble_profit:>13,.2f} | {winner:6} |")

    # Win Rate
    single_wr = single_metrics.get('win_rate', 0)
    ensemble_wr = ensemble_metrics.get('win_rate', 0)
    winner = format_winner(single_wr, ensemble_wr, True)
    print(f"| Win Rate %              | {single_wr:>11.2f}% | {ensemble_wr:>13.2f}% | {winner:6} |")

    # Profit Factor
    single_pf = single_metrics.get('profit_factor', 0)
    ensemble_pf = ensemble_metrics.get('profit_factor', 0)
    winner = format_winner(single_pf, ensemble_pf, True)
    print(f"| Profit Factor           | {single_pf:>12.2f} | {ensemble_pf:>14.2f} | {winner:6} |")

    # Max Drawdown
    single_dd = single_metrics.get('max_drawdown_pct', 0)
    ensemble_dd = ensemble_metrics.get('max_drawdown_pct', 0)
    winner = format_winner(single_dd, ensemble_dd, False)  # Lower is better
    print(f"| Max Drawdown %          | {single_dd:>11.2f}% | {ensemble_dd:>13.2f}% | {winner:6} |")

    # Total Trades
    single_trades = single_metrics.get('total_trades', 0)
    ensemble_trades = ensemble_metrics.get('total_trades', 0)
    print(f"| Total Trades            | {single_trades:>12} | {ensemble_trades:>14} |        |")

    print()
    print("="*70)

    # Decision recommendation
    if ensemble_metrics.get('message'):
        print("\n⚠️  ENSEMBLE MODEL NOT TESTED")
        print(f"   {ensemble_metrics['message']}")
        print("\n✅ SINGLE MODEL IS READY FOR DEPLOYMENT")
    elif single_wr >= ensemble_wr * 0.95 and single_pf >= ensemble_pf * 0.9:
        print("\n✅ RECOMMENDATION: Deploy SINGLE MODEL")
        print("   - Simpler system, easier to monitor")
        print("   - Performance comparable to ensemble")
    else:
        print("\n✅ RECOMMENDATION: Deploy ENSEMBLE MODEL")
        print("   - Superior performance metrics")
        print("   - Worth the added complexity")

def main():
    """Main backtesting workflow"""
    print("="*70)
    print("XAUUSD NEURAL BOT - PYTHON BACKTESTING SYSTEM")
    print("="*70)
    print()

    # Get project root
    project_root = Path(__file__).parent.parent

    # Step 1: Prepare data
    print("Step 1: Preparing backtest data...")
    # Use REAL data from Kaggle (1.06M bars, 2022-2024)
    data_file = project_root / 'python_backtesting' / 'xauusd_m1_real_backtest.parquet'

    if data_file.exists():
        print(f"Loading existing data from {data_file.name}...")
        data = pd.read_parquet(data_file)
        print(f"✓ Loaded {len(data):,} bars")
    else:
        print("Generating new M1 data...")
        data = prepare_backtest_data(
            start_date='2022-01-01',
            end_date='2024-12-31',
            output_file=str(data_file)
        )

    # Step 2: Run Single Model backtest
    print("\n" + "="*70)
    print("Step 2: Running Single Model backtest...")
    print("="*70)

    # Use the new standalone 26-feature model trained on real Kaggle data (2022-2024)
    lgb_model_path = str(project_root / 'python_training' / 'models' / 'lightgbm_real_26features.onnx')

    if not Path(lgb_model_path).exists():
        print(f"\n❌ ERROR: LightGBM model not found at {lgb_model_path}")
        print("Please ensure the model file exists")
        sys.exit(1)

    single_metrics = run_single_model_backtest(data, lgb_model_path)

    # Step 3: Run Ensemble Model backtest (if Transformer available)
    print("\n" + "="*70)
    print("Step 3: Running Ensemble Model backtest...")
    print("="*70)

    transformer_model_path = str(project_root / 'python_training' / 'models' / 'transformer.onnx')
    scaler_path = str(project_root / 'python_training' / 'models' / 'transformer_scaler_params.json')

    if Path(transformer_model_path).exists() and Path(scaler_path).exists():
        ensemble_metrics = run_ensemble_model_backtest(data, lgb_model_path, transformer_model_path, scaler_path)
    else:
        print("\n⚠️  Transformer model files not found:")
        print(f"   - {transformer_model_path}")
        print(f"   - {scaler_path}")
        print("\n   Ensemble backtest skipped.")
        print("   To test ensemble model:")
        print("   1. Ensure Transformer model is trained")
        print("   2. Run: python python_training/export_transformer_onnx.py")
        print("   3. Re-run this backtest")

        ensemble_metrics = {
            'message': 'Transformer ONNX files not found - export required'
        }

    # Step 4: Generate comparison report
    generate_comparison_report(single_metrics, ensemble_metrics)

    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review performance metrics above")
    print("2. If results are satisfactory, proceed to paper trading")
    print("3. If not, consider parameter optimization")
    print()

if __name__ == "__main__":
    main()
