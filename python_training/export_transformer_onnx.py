"""
Export Multi-TF Transformer to ONNX for MT5 Integration
Handles PyTorch → ONNX conversion with proper shapes and validation
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Import model architecture from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_multi_tf_transformer import MultiTFTransformer, FEATURE_COLS_26, SEQ_LENGTH, TIMEFRAMES


print("=" * 70)
print("EXPORT MULTI-TF TRANSFORMER TO ONNX")
print("=" * 70)
print()

# Configuration
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.15
FEATURE_SIZE = 130  # 26 × 5 timeframes

project_root = Path(__file__).parent.parent
model_dir = project_root / "python_training" / "models"


# =============================================================================
# Step 1: Load Trained PyTorch Model
# =============================================================================
print("📥 Step 1: Loading trained PyTorch model...")

model_path = model_dir / "multi_tf_transformer_price.pth"
if not model_path.exists():
    print(f"❌ ERROR: Model not found at {model_path}")
    print("   Please train the model first:")
    print("   python python_training/train_multi_tf_transformer.py")
    sys.exit(1)

# Create model instance
model = MultiTFTransformer(
    feature_size=FEATURE_SIZE,
    d_model=D_MODEL,
    nhead=N_HEADS,
    num_layers=N_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    seq_length=SEQ_LENGTH
)

# Load trained weights
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
else:
    model.load_state_dict(checkpoint)
    print("✓ Loaded model weights")

model.eval()  # Set to evaluation mode
print(f"   Model architecture: {FEATURE_SIZE} features, {SEQ_LENGTH} timesteps")
print()


# =============================================================================
# Step 2: Load Scaler for Normalization Parameters
# =============================================================================
print("📊 Step 2: Loading scaler parameters...")

scaler_path = model_dir / "multi_tf_scaler.pkl"
if not scaler_path.exists():
    print("⚠️  WARNING: Scaler file not found!")
    print(f"   Expected: {scaler_path}")
    print("   Creating dummy scaler (model may not perform well)")

    # Create dummy scaler
    scaler = MinMaxScaler()
    dummy_data = np.random.randn(1000, FEATURE_SIZE)
    scaler.fit(dummy_data)
else:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Loaded scaler from {scaler_path.name}")

print(f"   Features: {FEATURE_SIZE}")
print(f"   Min shape: {scaler.data_min_.shape}")
print(f"   Max shape: {scaler.data_max_.shape}")
print()


# =============================================================================
# Step 3: Export Scaler Parameters to JSON (for MT5)
# =============================================================================
print("💾 Step 3: Exporting scaler parameters to JSON...")

scaler_json = {
    'num_features': int(FEATURE_SIZE),
    'data_min': scaler.data_min_.tolist(),
    'data_max': scaler.data_max_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_range': list(scaler.feature_range)
}

scaler_json_path = model_dir / "transformer_scaler_params.json"
with open(scaler_json_path, 'w') as f:
    json.dump(scaler_json, f, indent=2)

print(f"✓ Saved scaler parameters to {scaler_json_path.name}")
print(f"   Min/Max values for {FEATURE_SIZE} features")
print()


# =============================================================================
# Step 4: Convert to ONNX
# =============================================================================
print("🔄 Step 4: Converting PyTorch → ONNX...")

# Create dummy input matching expected shape: [batch=1, seq=30, features=130]
dummy_input = torch.randn(1, SEQ_LENGTH, FEATURE_SIZE, dtype=torch.float32)

# ONNX export
onnx_path = model_dir / "transformer.onnx"

try:
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},   # Dynamic batch size
            'output': {0: 'batch'}
        },
        opset_version=14,  # Use opset 14 for better compatibility
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )

    print(f"✓ ONNX export successful: {onnx_path.name}")
    print(f"   Input shape: [1, {SEQ_LENGTH}, {FEATURE_SIZE}]")
    print(f"   Output shape: [1, 1]")

except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()


# =============================================================================
# Step 5: Validate ONNX Model
# =============================================================================
print("🧪 Step 5: Validating ONNX model...")

# Check ONNX model structure
try:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model structure is valid")
except Exception as e:
    print(f"❌ ONNX validation failed: {e}")
    sys.exit(1)

# Test ONNX Runtime inference
try:
    ort_session = ort.InferenceSession(str(onnx_path))

    # Get input/output info
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    print(f"   Input name: {input_name}")
    print(f"   Output name: {output_name}")

    # Test inference with random normalized data
    test_input = np.random.rand(1, SEQ_LENGTH, FEATURE_SIZE).astype(np.float32)
    test_input = test_input * 0.8 + 0.1  # Scale to [0.1, 0.9] (typical normalized range)

    ort_output = ort_session.run([output_name], {input_name: test_input})[0]

    print(f"   Test input shape: {test_input.shape}")
    print(f"   Test output shape: {ort_output.shape}")
    print(f"   Test output value: {ort_output[0][0]:.4f} (% price change)")

    # Compare PyTorch vs ONNX
    with torch.no_grad():
        torch_output = model(torch.from_numpy(test_input)).numpy()

    diff = np.abs(torch_output - ort_output).max()
    print(f"   PyTorch vs ONNX diff: {diff:.6f}")

    if diff < 1e-4:
        print("   ✓ ONNX inference matches PyTorch (excellent)")
    elif diff < 1e-2:
        print("   ✓ ONNX inference matches PyTorch (acceptable)")
    else:
        print(f"   ⚠️  WARNING: Large difference detected ({diff:.6f})")

except Exception as e:
    print(f"❌ ONNX Runtime test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()


# =============================================================================
# Step 6: Create Model Config for MT5
# =============================================================================
print("📝 Step 6: Creating model configuration...")

config = {
    'model_name': 'Multi-TF Transformer',
    'model_file': 'transformer.onnx',
    'scaler_file': 'transformer_scaler_params.json',
    'architecture': {
        'input_shape': [1, SEQ_LENGTH, FEATURE_SIZE],
        'output_shape': [1, 1],
        'sequence_length': SEQ_LENGTH,
        'num_features': FEATURE_SIZE,
        'features_per_tf': 26,
        'timeframes': TIMEFRAMES
    },
    'model_params': {
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT
    },
    'features': FEATURE_COLS_26,
    'normalization': {
        'method': 'MinMaxScaler',
        'feature_range': [0, 1],
        'note': 'Apply scaler_params.json normalization before inference'
    },
    'output': {
        'type': 'regression',
        'unit': 'percentage',
        'description': 'Predicted % price change over next 5 bars',
        'interpretation': {
            'positive': 'Predicted price increase → LONG signal',
            'negative': 'Predicted price decrease → SHORT signal',
            'threshold': 'Use absolute value > 0.1% for signal confidence'
        }
    },
    'usage_mt5': {
        'sequence_buffer': 'Maintain rolling 30-bar window of normalized features',
        'prediction_logic': 'if output > 0.1: LONG, elif output < -0.1: SHORT, else: HOLD',
        'ensemble_with': 'lightgbm_xauusd.onnx (both models must agree)'
    }
}

config_path = model_dir / "transformer_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Saved configuration to {config_path.name}")
print()


# =============================================================================
# Step 7: File Summary
# =============================================================================
print("=" * 70)
print("✅ TRANSFORMER ONNX EXPORT COMPLETE!")
print("=" * 70)
print()

print("📊 Files created:")
print(f"   1. {onnx_path.name} ({onnx_path.stat().st_size / 1024:.1f} KB)")
print(f"      - Input: [1, {SEQ_LENGTH}, {FEATURE_SIZE}] float32")
print(f"      - Output: [1, 1] float32 (% price change)")
print()
print(f"   2. {scaler_json_path.name} ({scaler_json_path.stat().st_size / 1024:.1f} KB)")
print(f"      - Normalization parameters for {FEATURE_SIZE} features")
print(f"      - MinMaxScaler: {scaler.feature_range}")
print()
print(f"   3. {config_path.name} ({config_path.stat().st_size / 1024:.1f} KB)")
print(f"      - Model metadata and usage instructions")
print()

print("🎯 Next steps:")
print("   1. Copy files to MT5:")
print(f"      cp {onnx_path} <MT5_DATA_PATH>/MQL5/Files/")
print(f"      cp {scaler_json_path} <MT5_DATA_PATH>/MQL5/Files/")
print()
print("   2. Implement ensemble EA:")
print("      - Load both LightGBM and Transformer models")
print("      - Maintain 30-bar sequence buffer")
print("      - Apply normalization using scaler_params.json")
print("      - Require both models to agree for trade signal")
print()
print("   3. Test in Strategy Tester:")
print("      - Compare single model vs ensemble performance")
print("      - Validate on 3 years of historical data")
print()

print("=" * 70)
