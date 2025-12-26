"""
Export trained LightGBM model to ONNX format
"""

import lightgbm as lgb
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from pathlib import Path

print("Loading LightGBM model...")
model_dir = Path(__file__).parent / 'models'
lgb_model_path = model_dir / 'lightgbm_synthetic.txt'

# Load the model
model = lgb.Booster(model_file=str(lgb_model_path))
print(f"✓ Loaded model from {lgb_model_path.name}")

# Define input type (26 features)
initial_types = [('input', FloatTensorType([None, 26]))]

print("\nConverting to ONNX...")
# Convert to ONNX
onnx_model = convert_lightgbm(
    model,
    initial_types=initial_types,
    target_opset=14
)

# Save ONNX model
onnx_path = model_dir / 'lightgbm_synthetic.onnx'
with open(onnx_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

print(f"✓ Saved ONNX model: {onnx_path.name}")
print(f"  File size: {onnx_path.stat().st_size / 1024:.2f} KB")
print("\n✅ ONNX export complete!")
