#!/bin/bash
# Complete pipeline: Train → Export → Backtest
# Run this after training completes

set -e  # Exit on error

echo "======================================================================"
echo "COMPLETE PIPELINE: TRAIN → EXPORT → BACKTEST"
echo "======================================================================"
echo

# Step 1: Check if training is complete
echo "Step 1: Checking if model exists..."
if [ ! -f "python_training/models/lightgbm_real_26features.txt" ]; then
    echo "❌ Model not found. Please wait for training to complete."
    exit 1
fi
echo "✓ Model found"
echo

# Step 2: Export to ONNX
echo "Step 2: Exporting model to ONNX..."
python python_training/export_real_model_onnx.py
echo

# Step 3: Run backtest
echo "Step 3: Running backtest on real data..."
python python_backtesting/run_backtest.py 2>&1 | tee python_backtesting/backtest_real_26features_results.log
echo

echo "======================================================================"
echo "PIPELINE COMPLETE"
echo "======================================================================"
echo
echo "Results saved to: python_backtesting/backtest_real_26features_results.log"
echo
