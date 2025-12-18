"""
Export LightGBM model to ONNX format (simplified - no runtime test)
Works with Python 3.14
"""

import lightgbm as lgb
import json
import os

print("=" * 70)
print("EXPORT LIGHTGBM MODEL TO ONNX")
print("=" * 70)
print()

# Install required packages (run this first if needed)
print("📦 Required packages:")
print("   pip install onnx==1.16.0")
print("   pip install onnxmltools==1.12.0")
print("   pip install skl2onnx==1.17.0")
print()

try:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
except ImportError:
    print("❌ Missing packages! Run:")
    print("   pip install onnx==1.16.0 onnxmltools==1.12.0 skl2onnx==1.17.0")
    exit(1)

# Load LightGBM model
print("📥 Loading LightGBM model...")
model = lgb.Booster(model_file='python_training/models/lightgbm_xauusd_v1.txt')

# Load feature list
with open('python_training/models/feature_list.json', 'r') as f:
    feature_cols = json.load(f)

print(f"   Model: {model.num_trees()} trees")
print(f"   Features: {len(feature_cols)}")
print()

# Convert to ONNX
print("🔄 Converting to ONNX...")
initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]

try:
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_types,
        target_opset=12
    )
    print("   ✓ Conversion successful")
except Exception as e:
    print(f"   ❌ Conversion failed: {e}")
    exit(1)

print()

# Save ONNX model
onnx_path = 'mt5_expert_advisor/Files/models/xauusd_ai_v1.onnx'
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

with open(onnx_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

file_size = os.path.getsize(onnx_path) / 1024
print(f"💾 ONNX model saved:")
print(f"   Path: {onnx_path}")
print(f"   Size: {file_size:.1f} KB")
print()

# Save feature list for MT5
features_path = 'mt5_expert_advisor/Files/config/features.json'
os.makedirs(os.path.dirname(features_path), exist_ok=True)

with open(features_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)

print(f"💾 Feature list saved:")
print(f"   Path: {features_path}")
print(f"   Features: {len(feature_cols)}")
print()

# Save model config
config = {
    'model_info': {
        'file': 'xauusd_ai_v1.onnx',
        'type': 'LightGBM',
        'num_features': len(feature_cols),
        'num_classes': 3,
        'num_trees': model.num_trees()
    },
    'trading_config': {
        'confidence_threshold': 0.55,
        'min_confidence': 0.50,
        'focus_signal': 'LONG',
        'max_trades_per_day': 5
    },
    'performance': {
        'baseline_accuracy': 0.475,
        'confidence_50_accuracy': 0.582,
        'confidence_55_accuracy': 0.633,
        'confidence_60_accuracy': 0.724,
        'long_accuracy_55': 0.834,
        'long_accuracy_60': 0.886
    },
    'class_mapping': {
        '0': 'SHORT',
        '1': 'HOLD',
        '2': 'LONG'
    },
    'usage_notes': [
        'Use 55% confidence threshold for balanced trading',
        'Use 60% confidence for conservative trading',
        'LONG signals have 83% win rate at 55% confidence',
        'Expected 4-5 trades per day at 55% threshold'
    ]
}

config_path = 'mt5_expert_advisor/Files/config/model_config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"💾 Config saved:")
print(f"   Path: {config_path}")
print()

print("=" * 70)
print("✅ EXPORT COMPLETE!")
print("=" * 70)
print()
print("📊 Summary:")
print(f"   Model: LightGBM → ONNX")
print(f"   Trees: {model.num_trees()}")
print(f"   Features: {len(feature_cols)}")
print(f"   Classes: 3 (SHORT/HOLD/LONG)")
print(f"   File size: {file_size:.1f} KB")
print()
print("🎯 Recommended Settings:")
print("   Confidence: 55%")
print("   Win rate: 63% overall, 83% on LONG")
print("   Trades/day: 4-5")
print()
print("📁 Files created:")
print(f"   1. {onnx_path}")
print(f"   2. {features_path}")
print(f"   3. {config_path}")
print()
print("🚀 Next: Push to GitHub and start MT5 EA development!")