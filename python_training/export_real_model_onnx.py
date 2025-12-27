"""
Export the real 26-feature LightGBM model to ONNX format

Converts: lightgbm_real_26features.txt → lightgbm_real_26features.onnx
"""

import lightgbm as lgb
from pathlib import Path
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

print("="*70)
print("EXPORTING REAL 26-FEATURE MODEL TO ONNX")
print("="*70)
print()

# Paths
project_root = Path(__file__).parent.parent
model_dir = project_root / 'python_training' / 'models'
lgb_txt_path = model_dir / 'lightgbm_real_26features.txt'
onnx_output_path = model_dir / 'lightgbm_real_26features.onnx'

print(f"Input:  {lgb_txt_path}")
print(f"Output: {onnx_output_path}")
print()

# Load LightGBM model
print("Loading LightGBM model from .txt file...")
model = lgb.Booster(model_file=str(lgb_txt_path))
print(f"✓ Model loaded")
print(f"  Number of trees: {model.num_trees()}")
print(f"  Number of features: {model.num_feature()}")
print()

# Convert to ONNX
print("Converting to ONNX format...")
num_features = model.num_feature()
print(f"  Model expects {num_features} features")
initial_types = [('input', FloatTensorType([None, num_features]))]
onnx_model = onnxmltools.convert_lightgbm(
    model,
    initial_types=initial_types,
    target_opset=14
)

# Save ONNX model
print("Saving ONNX model...")
with open(onnx_output_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

file_size = onnx_output_path.stat().st_size / 1024
print(f"✓ ONNX model saved: {file_size:.1f} KB")
print()

print("="*70)
print("EXPORT COMPLETE")
print("="*70)
print()
print("Model Details:")
print(f"  Trained on: Real Kaggle XAUUSD data (2022-2024)")
print(f"  Features: {num_features}")
print(f"  File: {onnx_output_path.name}")
print()
print("Next step: Run backtest")
print("  python python_backtesting/run_backtest.py")
print()
