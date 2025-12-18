"""
Robust ONNX export with MT5-friendly shapes and validation.
Ensures model has fixed batch size (1) and documented feature order.
"""
import json
import numpy as np
import lightgbm as lgb
import onnx
import onnxruntime as ort
from pathlib import Path


def export_onnx_mt5_friendly():
    """Export LightGBM to ONNX with MT5-compatible settings."""
    
    print("=" * 70)
    print("MT5-FRIENDLY ONNX EXPORT")
    print("=" * 70)
    print()
    
    project_root = Path(__file__).parent.parent
    
    # Load model
    model_path = project_root / "python_training" / "models" / "lightgbm_xauusd.pkl"
    print(f"Loading model: {model_path.name}")
    
    if not model_path.exists():
        print(f"  ✗ ERROR: Model not found at {model_path}")
        print("  Run: python python_training/train_lightgbm.py")
        return False
    
    model = lgb.Booster(model_file=str(model_path))
    print(f"  ✓ Trees: {model.num_trees()}")
    print(f"  ✓ Features: {model.num_feature()}")
    print()
    
    # Load feature order
    with open(project_root / "config" / "features_order.json", "r") as f:
        feature_cols = json.load(f)
    
    print(f"Features: {len(feature_cols)}")
    print()
    
    # Create dummy input with fixed batch size (1 x 26)
    dummy_input = np.zeros((1, len(feature_cols)), dtype=np.float32)
    
    # Export to ONNX
    onnx_path = project_root / "python_training" / "models" / "lightgbm_xauusd.onnx"
    print("Exporting to ONNX...")
    print(f"  Input shape: [1, {len(feature_cols)}]")
    print(f"  Output shape: [1, 3] (probabilities for SHORT, HOLD, LONG)")
    print()
    
    # Export with initial types to lock shapes
    try:
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        # Define input with fixed shape [1, 26]
        initial_types = [('input', FloatTensorType([1, len(feature_cols)]))]
        
        # Convert with zipmap=False to get flat probability array instead of dictionary
        # This is CRITICAL for MT5 compatibility - MT5 cannot handle ZipMap output types
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=12,  # Compatible with MT5
            zipmap=False      # Output probabilities as tensor, NOT dictionary
        )
        
        # Save
        onnx.save_model(onnx_model, str(onnx_path))
        print(f"  ✓ Saved: {onnx_path.name}")
        print(f"  Note: ONNX has 2 outputs:")
        print(f"        - Output 0 (label): int64[1] - predicted class index")
        print(f"        - Output 1 (probabilities): float[1,3] - class probabilities")
        print(f"  IMPORTANT: zipmap=False ensures probabilities are a tensor, not a dict")
        
    except ImportError:
        print("ERROR: onnxmltools not installed")
        print("Install: pip install onnxmltools")
        return False
    
    print()
    
    # Validate ONNX model
    print("Validating ONNX model...")
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model is valid")
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False
    
    print()
    
    # Test inference with onnxruntime
    print("Testing ONNX inference...")
    sess = ort.InferenceSession(str(onnx_path))
    
    # Check input/output
    print(f"  Model has {len(sess.get_outputs())} outputs:")
    for i, output in enumerate(sess.get_outputs()):
        print(f"    Output {i}: '{output.name}' | Shape: {output.shape} | Type: {output.type}")
    
    input_name = sess.get_inputs()[0].name
    print()
    
    # Test inference
    test_input = np.random.randn(1, len(feature_cols)).astype(np.float32)
    outputs = sess.run(None, {input_name: test_input})
    
    # Check if we have probabilities output (zipmap=False should give us array)
    if len(outputs) == 2:
        # Two outputs: label and probabilities
        if isinstance(outputs[1], list) and isinstance(outputs[1][0], dict):
            # Dictionary format (zipmap=True), extract probabilities
            probs_dict = outputs[1][0]
            onnx_probs = np.array([probs_dict[i] for i in range(3)]).reshape(1, 3)
        else:
            # Array format (zipmap=False)
            onnx_probs = outputs[1]
    else:
        # Single output (just probabilities)
        onnx_probs = outputs[0]
    
    print(f"  Test output (probabilities): {onnx_probs[0]}")
    print(f"  Sum: {onnx_probs.sum():.6f} (should be ~1.0)")
    print("  ✓ ONNX inference successful")
    print()
    
    # Compare LightGBM vs ONNX
    print("Comparing LightGBM vs ONNX predictions...")
    lgb_output = model.predict(test_input.astype(np.float64))
    
    max_diff = np.abs(lgb_output - onnx_probs).max()
    print(f"  LightGBM: {lgb_output[0]}")
    print(f"  ONNX:     {onnx_probs[0]}")
    print(f"  Max difference: {max_diff:.10f}")
    
    if max_diff < 1e-5:
        print("  ✓ LightGBM and ONNX match perfectly")
    elif max_diff < 1e-3:
        print("  ✓ LightGBM and ONNX match (acceptable tolerance)")
    else:
        print(f"  ⚠ WARNING: Difference {max_diff:.10f} may be too large")
    
    print()
    
    # Save metadata for MT5
    metadata = {
        "model_file": "lightgbm_xauusd.onnx",
        "input_shape": [1, len(feature_cols)],
        "output_shape": [1, 3],
        "num_features": len(feature_cols),
        "num_classes": 3,
        "feature_order": feature_cols,
        "class_labels": ["SHORT", "HOLD", "LONG"],
        "opset_version": 12,
        "export_date": "2025-12-12",
        "notes": [
            "Input: float32[1, 26] (batch_size=1, features=26)",
            "Output: float32[1, 3] (batch_size=1, classes=3)",
            "Probabilities sum to 1.0",
            "Class 0 = SHORT, Class 1 = HOLD, Class 2 = LONG"
        ]
    }
    
    metadata_path = project_root / "mt5_expert_advisor" / "Files" / "config" / "model_config.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_path.name}")
    print()
    
    # Copy ONNX to MT5 Files folder
    mt5_onnx_path = project_root / "mt5_expert_advisor" / "Files" / "lightgbm_xauusd.onnx"
    mt5_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy(str(onnx_path), str(mt5_onnx_path))
    print(f"Copied ONNX to: {mt5_onnx_path.relative_to(project_root)}")
    print()
    
    print("=" * 70)
    print("✅ ONNX EXPORT COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Copy mt5_expert_advisor/Files/ folder to MQL5/Files/")
    print("2. Compile EA in MetaEditor")
    print("3. Run parity test: python python_training/onnx_parity_test.py")
    print()
    
    return True


if __name__ == "__main__":
    success = export_onnx_mt5_friendly()
    exit(0 if success else 1)
