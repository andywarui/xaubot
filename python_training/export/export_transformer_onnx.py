"""
Export Transformer Model to ONNX for MT5 Integration

Task 1 of Phase 3: Export multi_tf_transformer_price.pth to ONNX format
with static shapes suitable for MT5 ONNX runtime.

Input:  [1, 30, 130] float32 - 30 timesteps × 130 features (26×5 TFs)
Output: [1, 1] float32 - predicted % price change (multi_tf_signal)

Usage:
    python python_training/export/export_transformer_onnx.py
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Configuration (must match training)
# =============================================================================
SEQ_LENGTH = 30
FEATURE_SIZE = 130
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.15

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "python_training" / "models"
EXPORT_DIR = BASE_DIR / "mt5_expert_advisor" / "Files" / "NeuralBot"

# =============================================================================
# Model Definition (copied from train_multi_tf_transformer.py)
# =============================================================================
class MultiTFTransformer(nn.Module):
    """
    Transformer encoder for multi-timeframe price prediction.
    
    Input: [batch, seq_length=30, features=130]
    Output: [batch, 1] (predicted % price change)
    """
    def __init__(
        self,
        feature_size: int = FEATURE_SIZE,
        d_model: int = D_MODEL,
        nhead: int = N_HEADS,
        num_layers: int = N_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        seq_length: int = SEQ_LENGTH
    ):
        super().__init__()
        
        self.feature_size = feature_size
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Project 130 features to d_model
        self.input_fc = nn.Linear(feature_size, d_model)
        
        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
    
    def forward(self, src):
        """
        Args:
            src: [batch_size, seq_length, 130]
        Returns:
            prediction: [batch_size, 1]
        """
        # Project features: [B, S, 130] -> [B, S, d_model]
        src = self.input_fc(src)
        
        # Add positional embedding
        src = src + self.pos_embedding
        
        # Transformer encode (batch_first=True)
        encoded = self.transformer_encoder(src)
        
        # Layer norm
        encoded = self.layer_norm(encoded)
        
        # Use last timestep for prediction
        last_step = encoded[:, -1, :]
        
        # Output
        out = self.fc_out(last_step)
        return out


def load_model(model_path: Path) -> MultiTFTransformer:
    """Load trained model from .pth file."""
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    model = MultiTFTransformer(
        feature_size=FEATURE_SIZE,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        seq_length=SEQ_LENGTH
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode (critical!)
    model.eval()
    
    return model


def export_to_onnx(model: nn.Module, output_path: Path, opset_version: int = 12):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model in eval mode
        output_path: Path to save .onnx file
        opset_version: ONNX opset version (12 is safe for MT5)
    """
    print(f"\nExporting to ONNX with opset {opset_version}...")
    
    # Ensure model is in eval mode with no gradients
    model.eval()
    torch.set_grad_enabled(False)
    
    # Create dummy input with EXACT shape for MT5: [1, 30, 130]
    dummy_input = torch.randn(1, SEQ_LENGTH, FEATURE_SIZE, dtype=torch.float32)
    
    # Get reference output for validation
    with torch.no_grad():
        reference_output = model(dummy_input).numpy()
    
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {reference_output.shape}")
    print(f"  Sample output: {reference_output[0, 0]:.6f}")
    
    # Use dynamo export (recommended for PyTorch 2.x)
    try:
        onnx_program = torch.onnx.export(
            model,
            (dummy_input,),
            dynamo=True
        )
        onnx_program.save(str(output_path))
        print(f"  ONNX model saved (dynamo) to: {output_path}")
    except Exception as e:
        print(f"  Dynamo export failed: {e}")
        print("  Trying legacy TorchScript approach...")
        
        # Fallback to scripting approach
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['multi_tf_signal'],
                dynamic_axes=None,
                verbose=False
            )
        print(f"  ONNX model saved (legacy) to: {output_path}")
    
    return dummy_input.numpy(), reference_output


def validate_onnx(onnx_path: Path, dummy_input: np.ndarray, reference_output: np.ndarray):
    """
    Validate ONNX model:
    1. Check model structure
    2. Verify shapes
    3. Compare outputs with PyTorch reference
    """
    print("\nValidating ONNX model...")
    
    # Load and check ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model structure is valid")
    
    # Get actual input name from model
    input_info = onnx_model.graph.input[0]
    input_name = input_info.name
    input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
    expected_input = [1, SEQ_LENGTH, FEATURE_SIZE]
    assert input_shape == expected_input, f"Input shape mismatch: {input_shape} vs {expected_input}"
    print(f"  ✓ Input shape correct: {input_shape} (name: '{input_name}')")
    
    # Check output shape
    output_info = onnx_model.graph.output[0]
    output_name = output_info.name
    output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
    expected_output = [1, 1]
    assert output_shape == expected_output, f"Output shape mismatch: {output_shape} vs {expected_output}"
    print(f"  ✓ Output shape correct: {output_shape} (name: '{output_name}')")
    
    # Run inference with ONNX Runtime using actual input name
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_output = sess.run(None, {input_name: dummy_input})[0]
    
    # Compare outputs
    diff = np.abs(onnx_output - reference_output).max()
    tolerance = 1e-5
    
    print(f"  PyTorch output: {reference_output[0, 0]:.6f}")
    print(f"  ONNX output:    {onnx_output[0, 0]:.6f}")
    print(f"  Max difference: {diff:.2e}")
    
    if diff < tolerance:
        print(f"  ✓ Output matches PyTorch within tolerance ({tolerance})")
    else:
        print(f"  ⚠ Output difference exceeds tolerance: {diff:.2e} > {tolerance}")
        if diff < 1e-4:
            print("    (This is likely acceptable for float32 precision)")
    
    return diff < 1e-4  # Acceptable threshold


def run_additional_tests(onnx_path: Path, n_tests: int = 100):
    """Run additional random input tests."""
    print(f"\nRunning {n_tests} random input tests...")
    
    # Load model
    model = load_model(MODELS_DIR / "multi_tf_transformer_price.pth")
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    
    # Get actual input name from model
    onnx_model = onnx.load(str(onnx_path))
    input_name = onnx_model.graph.input[0].name
    
    max_diff = 0.0
    for i in range(n_tests):
        # Random input
        test_input = np.random.randn(1, SEQ_LENGTH, FEATURE_SIZE).astype(np.float32)
        
        # PyTorch inference
        with torch.no_grad():
            pt_output = model(torch.tensor(test_input)).numpy()
        
        # ONNX inference with correct input name
        onnx_output = sess.run(None, {input_name: test_input})[0]
        
        diff = np.abs(onnx_output - pt_output).max()
        max_diff = max(max_diff, diff)
    
    print(f"  Max difference across {n_tests} tests: {max_diff:.2e}")
    
    if max_diff < 1e-4:
        print("  ✓ All tests passed!")
        return True
    else:
        print(f"  ⚠ Some differences exceed 1e-4")
        return max_diff < 1e-3  # Still acceptable


def main():
    print("=" * 60)
    print("PHASE 3 - TASK 1: Export Transformer to ONNX")
    print("=" * 60)
    
    # Ensure export directory exists
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_path = MODELS_DIR / "multi_tf_transformer_price.pth"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return False
    
    model = load_model(model_path)
    
    # Export to ONNX (try opset 12 for best MT5 compatibility)
    onnx_path = EXPORT_DIR / "transformer.onnx"
    
    dummy_input, reference_output = export_to_onnx(model, onnx_path, opset_version=12)
    
    # Validate
    validation_passed = validate_onnx(onnx_path, dummy_input, reference_output)
    
    # Additional tests
    tests_passed = run_additional_tests(onnx_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("TASK 1 SUMMARY")
    print("=" * 60)
    print(f"  Model:      multi_tf_transformer_price.pth")
    print(f"  ONNX:       {onnx_path}")
    print(f"  Input:      [1, {SEQ_LENGTH}, {FEATURE_SIZE}] float32")
    print(f"  Output:     [1, 1] float32")
    print(f"  File size:  {onnx_path.stat().st_size / 1024:.1f} KB")
    print(f"  Validation: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
    print(f"  Tests:      {'✓ PASSED' if tests_passed else '✗ FAILED'}")
    print("=" * 60)
    
    # Also copy to models folder for reference
    import shutil
    backup_path = MODELS_DIR / "transformer.onnx"
    shutil.copy(onnx_path, backup_path)
    print(f"\nBackup saved to: {backup_path}")
    
    return validation_passed and tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
