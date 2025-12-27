"""
Debug script to compare feature values between training and backtesting
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
project_root = Path(__file__).parent.parent
data_path = project_root / 'python_backtesting' / 'xauusd_m1_real_backtest.parquet'
df = pd.read_parquet(data_path)

print("="*70)
print("FEATURE DEBUGGING - TRAINING VS BACKTESTING")
print("="*70)
print()

# Sample bar 1000
idx = 1000
row = df.iloc[idx]

print(f"Sample bar {idx}")
print(f"Time: {row['time']}")
print(f"OHLC: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")
print()

# Calculate features TRAINING WAY (vectorized)
print("TRAINING FEATURES (vectorized pandas):")
print("-" * 70)

training_features = pd.DataFrame()
training_features['body'] = df['close'] - df['open']
training_features['body_abs'] = training_features['body'].abs()
training_features['candle_range'] = df['high'] - df['low']
training_features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
training_features['return_1'] = df['close'].pct_change(1)
training_features['return_5'] = df['close'].pct_change(5)
training_features['return_15'] = df['close'].pct_change(15)
training_features['return_60'] = df['close'].pct_change(60)
training_features['tr'] = df['tr']
training_features['atr_14'] = df['atr_14']
training_features['rsi_14'] = df['rsi_14']
training_features['ema_10'] = df['ema_10']
training_features['ema_20'] = df['ema_20']
training_features['ema_50'] = df['ema_50']
training_features['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
training_features['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)

# MTF placeholders
for i in range(10):
    training_features[f'mtf_{i}'] = 0.0

train_vec = training_features.iloc[idx].values
for i, name in enumerate(training_features.columns):
    print(f"  [{i:2d}] {name:20s} = {train_vec[i]:15.8f}")

print()

# Calculate features BACKTEST WAY (manual calculation)
print("BACKTEST FEATURES (manual calculation):")
print("-" * 70)

backtest_features = np.zeros(26, dtype=np.float32)

# Price features
body = row['close'] - row['open']
backtest_features[0] = body
backtest_features[1] = abs(body)
backtest_features[2] = row['high'] - row['low']
backtest_features[3] = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-8)

# Returns
backtest_features[4] = (df.iloc[idx]['close'] / df.iloc[idx-1]['close']) - 1.0
backtest_features[5] = (df.iloc[idx]['close'] / df.iloc[idx-5]['close']) - 1.0
backtest_features[6] = (df.iloc[idx]['close'] / df.iloc[idx-15]['close']) - 1.0
backtest_features[7] = (df.iloc[idx]['close'] / df.iloc[idx-60]['close']) - 1.0

# Technical indicators
backtest_features[8] = row.get('tr', 0.0)
backtest_features[9] = row.get('atr_14', 0.0)
backtest_features[10] = row.get('rsi_14', 50.0)
backtest_features[11] = row.get('ema_10', row['close'])
backtest_features[12] = row.get('ema_20', row['close'])
backtest_features[13] = row.get('ema_50', row['close'])

# Time features
hour = row['time'].hour
backtest_features[14] = np.sin(2 * np.pi * hour / 24)
backtest_features[15] = np.cos(2 * np.pi * hour / 24)

# MTF placeholders
for i in range(16, 26):
    backtest_features[i] = 0.0

feature_names = list(training_features.columns)
for i in range(26):
    print(f"  [{i:2d}] {feature_names[i]:20s} = {backtest_features[i]:15.8f}")

print()
print("="*70)
print("COMPARISON (Training - Backtest):")
print("="*70)

max_diff = 0.0
max_diff_idx = -1

for i in range(26):
    diff = abs(train_vec[i] - backtest_features[i])
    status = "✓ MATCH" if diff < 1e-6 else f"❌ DIFF: {diff:.8f}"
    print(f"  [{i:2d}] {feature_names[i]:20s} {status}")
    if diff > max_diff:
        max_diff = diff
        max_diff_idx = i

print()
if max_diff < 1e-6:
    print("✅ ALL FEATURES MATCH!")
else:
    print(f"❌ LARGEST DIFFERENCE at feature {max_diff_idx} ({feature_names[max_diff_idx]}): {max_diff:.8f}")
    print(f"   Training: {train_vec[max_diff_idx]:.8f}")
    print(f"   Backtest: {backtest_features[max_diff_idx]:.8f}")

print()
