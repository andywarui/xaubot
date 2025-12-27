"""
Debug script to test model predictions directly
"""

import pandas as pd
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Load data
project_root = Path(__file__).parent.parent
data_path = project_root / 'python_backtesting' / 'xauusd_m1_real_backtest.parquet'
model_path = project_root / 'python_training' / 'models' / 'lightgbm_real_26features.onnx'

df = pd.read_parquet(data_path)

print("="*70)
print("MODEL PREDICTION DEBUGGING")
print("="*70)
print()

# Load model
session = ort.InferenceSession(str(model_path))
print(f"Model loaded: {model_path.name}")
print(f"Input name: {session.get_inputs()[0].name}")
print(f"Input shape: {session.get_inputs()[0].shape}")
print()

# Prepare features (training way)
features = pd.DataFrame()
features['body'] = df['close'] - df['open']
features['body_abs'] = features['body'].abs()
features['candle_range'] = df['high'] - df['low']
features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
features['return_1'] = df['close'].pct_change(1)
features['return_5'] = df['close'].pct_change(5)
features['return_15'] = df['close'].pct_change(15)
features['return_60'] = df['close'].pct_change(60)
features['tr'] = df['tr']
features['atr_14'] = df['atr_14']
features['rsi_14'] = df['rsi_14']
features['ema_10'] = df['ema_10']
features['ema_20'] = df['ema_20']
features['ema_50'] = df['ema_50']
features['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
features['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
for i in range(10):
    features[f'mtf_{i}'] = 0.0

features = features.ffill().bfill().fillna(0)

# Test on 10 random samples
print("Testing 10 random samples:")
print("-" * 70)

np.random.seed(42)
test_indices = np.random.choice(range(1000, len(df)), size=10, replace=False)

class_counts = {0: 0, 1: 0, 2: 0}
confidence_sum = {0: 0.0, 1: 0.0, 2: 0.0}

for i, idx in enumerate(test_indices):
    row = df.iloc[idx]
    feat_vec = features.iloc[idx].values.astype(np.float32).reshape(1, -1)

    # Run model
    outputs = session.run(None, {'input': feat_vec})
    label_output = outputs[0][0]
    probabilities = outputs[1][0]

    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    class_counts[predicted_class] += 1
    confidence_sum[predicted_class] += confidence

    class_name = ['SHORT', 'HOLD', 'LONG'][predicted_class]

    print(f"Sample {i+1} (idx={idx}):")
    print(f"  Time: {row['time']}, Close: {row['close']:.2f}")
    print(f"  Probabilities: SHORT={probabilities[0]:.4f}, HOLD={probabilities[1]:.4f}, LONG={probabilities[2]:.4f}")
    print(f"  Prediction: {class_name} (confidence={confidence:.4f})")
    print()

print("="*70)
print("SUMMARY:")
print("-" * 70)
print(f"SHORT predictions: {class_counts[0]}/10 (avg conf: {confidence_sum[0]/max(1,class_counts[0]):.4f})")
print(f"HOLD predictions:  {class_counts[1]}/10 (avg conf: {confidence_sum[1]/max(1,class_counts[1]):.4f})")
print(f"LONG predictions:  {class_counts[2]}/10 (avg conf: {confidence_sum[2]/max(1,class_counts[2]):.4f})")
print()

# Now test on first 100,000 bars to see overall distribution
print("Testing on first 100,000 bars...")
X = features.iloc[1000:101000].values.astype(np.float32)

all_class_counts = {0: 0, 1: 0, 2: 0}
high_conf_counts = {0: 0, 1: 0, 2: 0}

for i in range(len(X)):
    outputs = session.run(None, {'input': X[i:i+1]})
    probabilities = outputs[1][0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    all_class_counts[predicted_class] += 1

    if confidence >= 0.35:
        high_conf_counts[predicted_class] += 1

print()
print("Overall distribution (100K bars):")
print(f"  SHORT: {all_class_counts[0]:,} ({all_class_counts[0]/100000*100:.2f}%)")
print(f"  HOLD:  {all_class_counts[1]:,} ({all_class_counts[1]/100000*100:.2f}%)")
print(f"  LONG:  {all_class_counts[2]:,} ({all_class_counts[2]/100000*100:.2f}%)")
print()
print("High confidence (>= 0.35) distribution:")
print(f"  SHORT: {high_conf_counts[0]:,}")
print(f"  HOLD:  {high_conf_counts[1]:,}")
print(f"  LONG:  {high_conf_counts[2]:,}")
print()
