"""
Debug ONNX model outputs to understand structure
"""

import pandas as pd
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Load model
project_root = Path(__file__).parent.parent
model_path = project_root / 'python_training' / 'models' / 'lightgbm_real_26features.onnx'
data_path = project_root / 'python_backtesting' / 'xauusd_m1_real_backtest.parquet'

session = ort.InferenceSession(str(model_path))

print("="*70)
print("ONNX MODEL OUTPUT STRUCTURE")
print("="*70)
print()

# Print model metadata
print("Model Inputs:")
for inp in session.get_inputs():
    print(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
print()

print("Model Outputs:")
for i, out in enumerate(session.get_outputs()):
    print(f"  [{i}] Name: {out.name}, Shape: {out.shape}, Type: {out.type}")
print()

# Load data and prepare one sample
df = pd.read_parquet(data_path)

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

# Test one sample
idx = 1000
feat_vec = features.iloc[idx].values.astype(np.float32).reshape(1, -1)

print(f"Testing sample {idx}:")
print(f"  Input shape: {feat_vec.shape}")
print(f"  Input dtype: {feat_vec.dtype}")
print()

outputs = session.run(None, {'input': feat_vec})

print(f"Number of outputs: {len(outputs)}")
print()

for i, output in enumerate(outputs):
    print(f"Output [{i}]:")
    print(f"  Type: {type(output)}")
    if hasattr(output, 'shape'):
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
    print(f"  Value: {output}")
    print()

# Detailed analysis
if len(outputs) >= 2:
    print("Detailed Analysis:")
    print("-" * 70)

    output_0 = outputs[0]
    output_1 = outputs[1]

    print(f"Output[0] (label?):")
    print(f"  Value: {output_0}")
    if len(output_0.shape) > 0 and output_0.shape[0] > 0:
        print(f"  Predicted class: {output_0[0]}")
    print()

    print(f"Output[1] (probabilities?):")
    print(f"  Raw type: {type(output_1)}")
    print(f"  Raw value: {output_1}")

    # LightGBM ONNX outputs probabilities as list of dicts
    if isinstance(output_1, list) and len(output_1) > 0:
        prob_dict = output_1[0]  # First (and only) sample
        print(f"  Dict keys: {prob_dict.keys() if isinstance(prob_dict, dict) else 'Not a dict'}")
        if isinstance(prob_dict, dict):
            # Extract probabilities
            print(f"  Probabilities:")
            for cls in sorted(prob_dict.keys()):
                prob = prob_dict[cls]
                cls_name = ['SHORT', 'HOLD', 'LONG'][cls]
                print(f"    [{cls}] {cls_name}: {prob:.6f}")

            # Find max
            max_cls = max(prob_dict.keys(), key=lambda k: prob_dict[k])
            max_prob = prob_dict[max_cls]
            max_name = ['SHORT', 'HOLD', 'LONG'][max_cls]
            print(f"  Max probability: {max_prob:.6f} at class {max_cls} ({max_name})")
    print()
