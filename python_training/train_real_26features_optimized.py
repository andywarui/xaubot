"""
OPTIMIZED Training: LightGBM on Real Data
Uses vectorized operations and simpler target calculation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json

print("="*70)
print("OPTIMIZED TRAINING - LIGHTGBM 26 FEATURES")
print("="*70)
print()

# Paths
project_root = Path(__file__).parent.parent
data_path = project_root / 'python_backtesting' / 'xauusd_m1_real_backtest.parquet'
model_dir = project_root / 'python_training' / 'models'

print(f"Using pre-processed data: {data_path.name}")
print()

# Load pre-processed data (already has indicators)
print("Loading data...")
df = pd.read_parquet(data_path)
print(f"✓ Loaded {len(df):,} bars")
print()

# Create simplified target: Future return direction
print("Creating targets (vectorized)...")
df['future_return'] = df['close'].shift(-10) / df['close'] - 1.0

# Simple thresholds
long_threshold = 0.001  # 0.1% up
short_threshold = -0.001  # 0.1% down

df['label'] = 1  # HOLD default
df.loc[df['future_return'] > long_threshold, 'label'] = 2  # LONG
df.loc[df['future_return'] < short_threshold, 'label'] = 0  # SHORT

print(f"✓ Targets created")
print()

# Build feature matrix (vectorized)
print("Building feature matrix...")

features = pd.DataFrame()

# Price features
features['body'] = df['close'] - df['open']
features['body_abs'] = features['body'].abs()
features['candle_range'] = df['high'] - df['low']
features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

# Returns
features['return_1'] = df['close'].pct_change(1)
features['return_5'] = df['close'].pct_change(5)
features['return_15'] = df['close'].pct_change(15)
features['return_60'] = df['close'].pct_change(60)

# Technical indicators (already calculated)
features['tr'] = df['tr']
features['atr_14'] = df['atr_14']
features['rsi_14'] = df['rsi_14']
features['ema_10'] = df['ema_10']
features['ema_20'] = df['ema_20']
features['ema_50'] = df['ema_50']

# Time features
features['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
features['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)

# MTF placeholders (10 features)
for i in range(10):
    features[f'mtf_{i}'] = 0.0

# Fill NaN
features = features.ffill().bfill().fillna(0)

print(f"✓ Features: {features.shape}")
print()

# Remove samples without valid targets
valid_mask = (~df['future_return'].isna()) & (features.index >= 100)
X = features[valid_mask].values.astype(np.float32)
y = df['label'][valid_mask].values.astype(np.int32)

print(f"Valid samples: {len(X):,}")
print()

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print("Target distribution:")
for cls, count in zip(unique, counts):
    cls_name = ['SHORT', 'HOLD', 'LONG'][cls]
    print(f"  {cls_name}: {count:,} ({count/len(y)*100:.1f}%)")
print()

# Train/test split (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train):,}")
print(f"Test:  {len(X_test):,}")
print()

# Train LightGBM
print("Training LightGBM...")
print()

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'max_depth': 6
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[test_data],
    valid_names=['test'],
    callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=25)]
)

print()

# Evaluate
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_class = np.argmax(y_pred, axis=1)
accuracy = (y_pred_class == y_test).mean()

print(f"Test Accuracy: {accuracy*100:.2f}%")
print()

# Save
model_path = model_dir / 'lightgbm_real_26features.txt'
model.save_model(str(model_path))
print(f"✓ Saved: {model_path.name}")
print()

config = {
    'accuracy': float(accuracy),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'best_iteration': model.best_iteration
}

config_path = model_dir / 'lightgbm_real_26features_config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("="*70)
print("TRAINING COMPLETE")
print("="*70)
print()
print(f"Accuracy: {accuracy*100:.1f}%")
print(f"Model: {model_path.name}")
print()
print("Next: python python_training/export_real_model_onnx.py")
