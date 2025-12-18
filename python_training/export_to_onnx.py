"""
Export LightGBM to ONNX format.
"""
import sys
import json
import numpy as np
import lightgbm as lgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from pathlib import Path


def main():
    print("=" * 70)
    print("Export LightGBM to ONNX")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    
    # Load model
    print("\nLoading LightGBM model...")
    model = lgb.Booster(model_file=str(project_root / 'python_training' / 'models' / 'lightgbm_xauusd.pkl'))
    
    # Load feature order
    with open(project_root / 'config' / 'features_order.json', 'r') as f:
        feature_cols = json.load(f)
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Trees: {model.num_trees()}")
    
    # Convert to ONNX
    print("\nConverting to ONNX...")
    initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]
    
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_types,
        target_opset=12
    )
    
    # Save ONNX
    onnx_path = project_root / 'python_training' / 'models' / 'lightgbm_xauusd.onnx'
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    file_size = onnx_path.stat().st_size / 1024
    print(f"  Saved: {onnx_path}")
    print(f"  Size: {file_size:.1f} KB")
    
    # Verify ONNX
    print("\nVerifying ONNX inference...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(str(onnx_path))
    
    # Test with random input
    test_input = np.random.randn(1, len(feature_cols)).astype(np.float32)
    
    # LightGBM prediction
    lgb_pred = model.predict(test_input)[0]
    
    # ONNX prediction
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    onnx_pred = session.run([label_name], {input_name: test_input})[0][0]
    
    # Compare
    max_diff = np.max(np.abs(lgb_pred - onnx_pred))
    print(f"  Max difference: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("  Parity check: PASSED")
    else:
        print("  Parity check: WARNING - difference > 1e-5")
    
    # Copy to MT5 folder
    mt5_path = project_root / 'mt5_expert_advisor' / 'Files' / 'lightgbm_xauusd.onnx'
    mt5_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(mt5_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"\n  Copied to MT5: {mt5_path}")
    
    print("\nONNX export complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
