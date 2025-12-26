"""
Diagnostic Script: Analyze where ML signals are being filtered

This script examines the first 100,000 bars to understand:
1. How many signals the ML model predicts (LONG/SHORT)
2. What are the confidence scores
3. Which validation layers are blocking signals
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from backtest_engine import XAUUSDBacktester

print("="*70)
print("SIGNAL DIAGNOSTIC ANALYSIS")
print("="*70)
print()

# Load data
print("Loading backtest data...")
data_path = Path(__file__).parent / 'xauusd_m1_backtest.parquet'
df = pd.read_parquet(data_path)
print(f"✓ Loaded {len(df):,} bars")
print()

# Load model
print("Loading LightGBM model...")
model_path = Path(__file__).parent.parent / 'python_training' / 'models' / 'lightgbm_synthetic.onnx'
session = ort.InferenceSession(str(model_path))
print(f"✓ Model loaded from {model_path.name}")
print()

# Create backtest engine for feature calculation and validation
engine = XAUUSDBacktester(
    initial_balance=10000.0,
    confidence_threshold=0.60,
    risk_percent=0.5
)

# Analyze first 100k bars
sample_size = 100000
print(f"Analyzing first {sample_size:,} bars...")
print()

stats = {
    'total_bars': 0,
    'ml_long': 0,
    'ml_short': 0,
    'ml_hold': 0,
    'long_confidence_low': 0,
    'short_confidence_low': 0,
    'long_spread_blocked': 0,
    'long_rsi_blocked': 0,
    'long_macd_blocked': 0,
    'long_adx_blocked': 0,
    'long_atr_blocked': 0,
    'long_mtf_blocked': 0,
    'short_spread_blocked': 0,
    'short_rsi_blocked': 0,
    'short_macd_blocked': 0,
    'short_adx_blocked': 0,
    'short_atr_blocked': 0,
    'short_mtf_blocked': 0,
    'long_passed_all': 0,
    'short_passed_all': 0
}

confidence_scores = {'long': [], 'short': []}

# Process bars
for idx in range(100, min(sample_size, len(df))):
    stats['total_bars'] += 1

    # Calculate features
    features = engine.calculate_26_features(df, idx)
    if features is None:
        continue

    # Get ML prediction
    features_input = features.reshape(1, -1).astype(np.float32)
    outputs = session.run(None, {'input': features_input})
    probabilities = outputs[1][0]  # [prob_short, prob_hold, prob_long]

    predicted_class = int(outputs[0][0])

    # Track predictions
    if predicted_class == 0:
        stats['ml_short'] += 1
        confidence = probabilities[0]
        confidence_scores['short'].append(confidence)

        # Check why SHORT might be blocked
        if confidence < 0.60:
            stats['short_confidence_low'] += 1
        else:
            # Check each validation layer
            row = df.iloc[idx]
            spread = row['high'] - row['low']

            if spread > 2.0:
                stats['short_spread_blocked'] += 1
            elif row.get('rsi_14', 50) < 30:
                stats['short_rsi_blocked'] += 1
            elif not engine._check_macd_bearish(df, idx):
                stats['short_macd_blocked'] += 1
            elif row.get('adx_14', 0) < 20:
                stats['short_adx_blocked'] += 1
            elif not (1.5 <= row.get('atr_14', 0) <= 8.0):
                stats['short_atr_blocked'] += 1
            elif not engine._check_mtf_bearish(df, idx):
                stats['short_mtf_blocked'] += 1
            else:
                stats['short_passed_all'] += 1

    elif predicted_class == 2:
        stats['ml_long'] += 1
        confidence = probabilities[2]
        confidence_scores['long'].append(confidence)

        # Check why LONG might be blocked
        if confidence < 0.60:
            stats['long_confidence_low'] += 1
        else:
            # Check each validation layer
            row = df.iloc[idx]
            spread = row['high'] - row['low']

            if spread > 2.0:
                stats['long_spread_blocked'] += 1
            elif row.get('rsi_14', 50) > 70:
                stats['long_rsi_blocked'] += 1
            elif not engine._check_macd_bullish(df, idx):
                stats['long_macd_blocked'] += 1
            elif row.get('adx_14', 0) < 20:
                stats['long_adx_blocked'] += 1
            elif not (1.5 <= row.get('atr_14', 0) <= 8.0):
                stats['long_atr_blocked'] += 1
            elif not engine._check_mtf_bullish(df, idx):
                stats['long_mtf_blocked'] += 1
            else:
                stats['long_passed_all'] += 1
    else:
        stats['ml_hold'] += 1

    if (idx + 1) % 20000 == 0:
        print(f"  Processed {idx+1:,} bars...")

print()
print("="*70)
print("DIAGNOSTIC RESULTS")
print("="*70)
print()

print(f"Total bars analyzed: {stats['total_bars']:,}")
print()

print("ML Model Predictions:")
print(f"  LONG:  {stats['ml_long']:,} ({stats['ml_long']/stats['total_bars']*100:.1f}%)")
print(f"  SHORT: {stats['ml_short']:,} ({stats['ml_short']/stats['total_bars']*100:.1f}%)")
print(f"  HOLD:  {stats['ml_hold']:,} ({stats['ml_hold']/stats['total_bars']*100:.1f}%)")
print()

if confidence_scores['long']:
    print(f"LONG Confidence Scores:")
    print(f"  Mean: {np.mean(confidence_scores['long']):.3f}")
    print(f"  Max:  {np.max(confidence_scores['long']):.3f}")
    print(f"  Min:  {np.min(confidence_scores['long']):.3f}")
    print(f"  >60%: {sum(1 for c in confidence_scores['long'] if c > 0.6)} signals")
    print()

if confidence_scores['short']:
    print(f"SHORT Confidence Scores:")
    print(f"  Mean: {np.mean(confidence_scores['short']):.3f}")
    print(f"  Max:  {np.max(confidence_scores['short']):.3f}")
    print(f"  Min:  {np.min(confidence_scores['short']):.3f}")
    print(f"  >60%: {sum(1 for c in confidence_scores['short'] if c > 0.6)} signals")
    print()

print("LONG Signal Filtering:")
print(f"  Total LONG predictions: {stats['ml_long']:,}")
print(f"  Blocked by confidence (<60%): {stats['long_confidence_low']:,}")
print(f"  Blocked by spread filter: {stats['long_spread_blocked']:,}")
print(f"  Blocked by RSI filter: {stats['long_rsi_blocked']:,}")
print(f"  Blocked by MACD filter: {stats['long_macd_blocked']:,}")
print(f"  Blocked by ADX filter: {stats['long_adx_blocked']:,}")
print(f"  Blocked by ATR filter: {stats['long_atr_blocked']:,}")
print(f"  Blocked by MTF filter: {stats['long_mtf_blocked']:,}")
print(f"  ✓ PASSED ALL FILTERS: {stats['long_passed_all']:,}")
print()

print("SHORT Signal Filtering:")
print(f"  Total SHORT predictions: {stats['ml_short']:,}")
print(f"  Blocked by confidence (<60%): {stats['short_confidence_low']:,}")
print(f"  Blocked by spread filter: {stats['short_spread_blocked']:,}")
print(f"  Blocked by RSI filter: {stats['short_rsi_blocked']:,}")
print(f"  Blocked by MACD filter: {stats['short_macd_blocked']:,}")
print(f"  Blocked by ADX filter: {stats['short_adx_blocked']:,}")
print(f"  Blocked by ATR filter: {stats['short_atr_blocked']:,}")
print(f"  Blocked by MTF filter: {stats['short_mtf_blocked']:,}")
print(f"  ✓ PASSED ALL FILTERS: {stats['short_passed_all']:,}")
print()

total_passed = stats['long_passed_all'] + stats['short_passed_all']
print(f"Total signals that should execute: {total_passed:,} ({total_passed/stats['total_bars']*100:.2f}%)")
print()

# Recommendations
print("="*70)
print("RECOMMENDATIONS")
print("="*70)
print()

if stats['long_confidence_low'] + stats['short_confidence_low'] > stats['ml_long'] + stats['ml_short'] - total_passed:
    print("❌ PRIMARY ISSUE: Low confidence scores")
    print("   → Model is predicting LONG/SHORT but with <60% confidence")
    print("   → Solution: Lower confidence_threshold to 0.50 or 0.45")
    print()

if stats['long_macd_blocked'] > stats['long_passed_all']:
    print("❌ MACD filter blocking most LONG signals")
    print("   → Consider relaxing MACD requirements")
    print()

if stats['short_macd_blocked'] > stats['short_passed_all']:
    print("❌ MACD filter blocking most SHORT signals")
    print("   → Consider relaxing MACD requirements")
    print()

if stats['long_mtf_blocked'] > stats['long_passed_all']:
    print("❌ Multi-timeframe filter blocking most LONG signals")
    print("   → MTF features may be too restrictive (all zeros in synthetic data)")
    print()

if stats['short_mtf_blocked'] > stats['short_passed_all']:
    print("❌ Multi-timeframe filter blocking most SHORT signals")
    print("   → MTF features may be too restrictive (all zeros in synthetic data)")
    print()

if total_passed == 0:
    print("⚠️  ZERO signals passing validation!")
    print("   → Hybrid validation is too strict for this data")
    print("   → Recommend disabling MTF filter or lowering confidence threshold")
    print()
