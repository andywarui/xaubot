"""
Export LightGBM to ONNX with MT5-compatible settings.
Forces Opset 14 and fixed batch size.
"""
import numpy as np
import lightgbm as lgb
import onnx
import onnxruntime as ort
from pathlib import Path

def export_lightgbm_mt5():
    """Export with MT5-compatible ONNX format."""

    print("=" * 70)
    print("MT5 ONNX EXPORT - FIXED OPSET")
    print("=" * 70)

    # Load LightGBM model
    model_path = Path("python_training/models/lightgbm_xauusd.pkl")
    print(f"\nLoading: {model_path}")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return False

    model = lgb.Booster(model_file=str(model_path))
    print(f"  Trees: {model.num_trees()}")
    print(f"  Features: {model.num_feature()}")

    # Export to ONNX with specific opset
    try:
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        # Fixed input shape [1, 26]
        initial_types = [('input', FloatTensorType([1, 26]))]

        print("\nExporting to ONNX...")
        print(f"  Opset: 14 (MT5 compatible)")
        print(f"  Input: [1, 26] (fixed batch)")
        print(f"  ZipMap: False (tensor output)")

        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=14,  # Higher opset for better compatibility
            zipmap=False
        )

        # Save to temp location
        temp_path = Path("python_training/models/lightgbm_xauusd_temp.onnx")
        onnx.save_model(onnx_model, str(temp_path))

        # Load and fix shapes
        onnx_model = onnx.load(str(temp_path))

        # Force output shapes to be fixed [1, 3]
        for output in onnx_model.graph.output:
            if output.name == "probabilities":
                # Clear existing shape
                while len(output.type.tensor_type.shape.dim) > 0:
                    output.type.tensor_type.shape.dim.pop()
                # Add fixed shape [1, 3]
                dim1 = output.type.tensor_type.shape.dim.add()
                dim1.dim_value = 1
                dim2 = output.type.tensor_type.shape.dim.add()
                dim2.dim_value = 3

        # Save final model
        final_path = Path("python_training/models/lightgbm_xauusd.onnx")
        onnx.save_model(onnx_model, str(final_path))
        temp_path.unlink()  # Delete temp file

        print(f"  Saved: {final_path.name}")

    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install onnxmltools")
        return False

    # Validate
    print("\nValidating ONNX...")
    try:
        onnx_model = onnx.load(str(final_path))
        onnx.checker.check_model(onnx_model)
        print("  Model is valid")

        # Check final structure
        print("\nFinal Model Structure:")
        print(f"  Opset: {onnx_model.opset_import[0].version}")
        for inp in onnx_model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  Input '{inp.name}': {shape}")
        for out in onnx_model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  Output '{out.name}': {shape}")

    except Exception as e:
        print(f"  Validation failed: {e}")
        return False

    # Test inference
    print("\nTesting ONNX inference...")
    sess = ort.InferenceSession(str(final_path))
    test_input = np.random.randn(1, 26).astype(np.float32)
    outputs = sess.run(None, {'input': test_input})

    if len(outputs) == 2:
        probs = outputs[1]
        print(f"  Output shape: {probs.shape}")
        print(f"  Probabilities: {probs[0]}")
        print(f"  Sum: {probs.sum():.6f}")

        if probs.shape == (1, 3) and 0.99 < probs.sum() < 1.01:
            print("  PASS: Output is correct!")
        else:
            print("  WARNING: Unexpected output format")

    # Copy to MT5 folders
    print("\nCopying to MT5 folders...")
    import shutil

    destinations = [
        Path("mt5_expert_advisor/Files/lightgbm_xauusd.onnx"),
        Path("MT5_XAUBOT/Files/lightgbm_xauusd.onnx")
    ]

    for dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(final_path), str(dest))
        print(f"  Copied to: {dest}")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print("\nFile size:", final_path.stat().st_size // 1024, "KB")
    print("\nNext: Copy to MT5 agent directory and restart MT5")

    return True


if __name__ == "__main__":
    success = export_lightgbm_mt5()
    exit(0 if success else 1)
