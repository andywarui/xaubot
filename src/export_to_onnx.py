"""
Export LightGBM model to ONNX format for MT5 integration
"""

import lightgbm as lgb
import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import json

print("=" * 70)
print("EXPORT LIGHTGBM MODEL TO ONNX")
print("=" * 70)
print()

# Load LightGBM model
print("📥 Loading LightGBM model...")
model = lgb.Booster(model_file='python_training/models/lightgbm_xauusd_v1.txt')

# Load feature list
with open('python_training/models/feature_list.json', 'r') as f:
    feature_cols = json.load(f)

print(f"   Model loaded: {model.num_trees()} trees")
print(f"   Features: {len(feature_cols)}")
print()

# Convert to ONNX
print("🔄 Converting to ONNX format...")

# Define initial types (input shape)
initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]

# Convert
onnx_model = onnxmltools.convert_lightgbm(
    model,
    initial_types=initial_types,
    target_opset=12
)

print("   ✓ Conversion complete")
print()

# Save ONNX model
onnx_path = 'mt5_expert_advisor/Files/models/xauusd_ai_v1.onnx'
print(f"💾 Saving ONNX model to: {onnx_path}")

import os
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

with open(onnx_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

file_size = os.path.getsize(onnx_path) / 1024
print(f"   File size: {file_size:.1f} KB")
print()

# Test ONNX inference
print("🧪 Testing ONNX inference...")
import onnxruntime as ort

# Create session
session = ort.InferenceSession(onnx_path)

# Test input (dummy data)
test_input = np.random.randn(1, len(feature_cols)).astype(np.float32)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
output = session.run([output_name], {input_name: test_input})
predictions = output[0]

print(f"   Input shape: {test_input.shape}")
print(f"   Output shape: {predictions.shape}")
print(f"   ✓ ONNX inference successful")
print()

# Save inference config for MT5
config = {
    'model_file': 'xauusd_ai_v1.onnx',
    'num_features': len(feature_cols),
    'num_classes': 3,
    'feature_names': feature_cols,
    'class_mapping': {
        '0': 'SHORT',
        '1': 'HOLD',
        '2': 'LONG'
    },
    'confidence_threshold': 0.55,
    'recommended_thresholds': {
        '50%': 'Balanced (5 trades/day, 58% accuracy)',
        '55%': 'Recommended (5 trades/day, 63% accuracy)',
        '60%': 'Conservative (1-2 trades/day, 72% accuracy)',
        '65%': 'Ultra-conservative (0.5 trades/day, 90% accuracy)'
    },
    'performance': {
        'overall_accuracy': 0.475,
        'confidence_55_accuracy': 0.633,
        'confidence_60_accuracy': 0.724,
        'long_signal_accuracy_55': 0.834
    }
}

config_path = 'mt5_expert_advisor/Files/config/model_config.json'
os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"💾 Saved config to: {config_path}")
print()

print("=" * 70)
print("✅ ONNX EXPORT COMPLETE!")
print("=" * 70)
print()
print("📊 Summary:")
print(f"   Model: LightGBM → ONNX")
print(f"   Input: {len(feature_cols)} features")
print(f"   Output: 3 classes (SHORT/HOLD/LONG)")
print(f"   File: {onnx_path}")
print(f"   Size: {file_size:.1f} KB")
print()
print("🎯 Next Steps:")
print("   1. Model is ready for MT5 integration")
print("   2. Use 55% confidence threshold")
print("   3. Focus on LONG signals (83% win rate)")
print("   4. Expected: 4-5 trades/day")
print()
print("📝 Files created:")
print(f"   - {onnx_path}")
print(f"   - {config_path}")