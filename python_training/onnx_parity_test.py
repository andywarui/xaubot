"""
ONNX Parity Test: Compare Python ONNX inference with MT5-logged predictions.
Load MT5 feature log and prediction log, run ONNX in Python, compare outputs.
"""
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from typing import Tuple


def load_mt5_logs(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MT5 feature and prediction logs."""
    
    feature_log_path = project_root / "mt5_expert_advisor" / "feature_log.csv"
    prediction_log_path = project_root / "mt5_expert_advisor" / "prediction_log.csv"
    
    if not feature_log_path.exists():
        raise FileNotFoundError(f"Feature log not found: {feature_log_path}")
    
    if not prediction_log_path.exists():
        raise FileNotFoundError(f"Prediction log not found: {prediction_log_path}")
    
    # Load logs (MT5 uses semicolon separator)
    features_df = pd.read_csv(feature_log_path, sep=";")
    predictions_df = pd.read_csv(prediction_log_path, sep=";")
    
    # Convert time to datetime
    features_df["time"] = pd.to_datetime(features_df["time"])
    predictions_df["time"] = pd.to_datetime(predictions_df["time"])
    
    # Merge on time
    merged_df = features_df.merge(predictions_df, on="time", how="inner")
    
    return merged_df, features_df.columns[1:]  # Skip 'time' column


def run_onnx_inference(onnx_path: Path, features: np.ndarray) -> np.ndarray:
    """Run ONNX inference."""
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # Ensure float32
    features = features.astype(np.float32)
    
    # Run inference
    output = sess.run([output_name], {input_name: features})[0]
    return output


def main():
    print("=" * 70)
    print("ONNX PARITY TEST: Python vs MT5")
    print("=" * 70)
    print()
    
    project_root = Path(__file__).parent.parent
    
    # Load feature order contract
    with open(project_root / "config" / "features_order.json", "r") as f:
        feature_order = json.load(f)
    
    print(f"Feature contract: {len(feature_order)} features")
    print()
    
    # Load MT5 logs
    print("Loading MT5 logs...")
    try:
        merged_df, feature_cols_mt5 = load_mt5_logs(project_root)
        print(f"  ✓ Loaded {len(merged_df):,} rows from MT5 logs")
        print(f"  Time range: {merged_df['time'].min()} → {merged_df['time'].max()}")
    except FileNotFoundError as e:
        print(f"  ✗ ERROR: {e}")
        print()
        print("Run EA with EnableFeatureLog=true and EnablePredictionLog=true first")
        return False
    
    print()
    
    # Verify feature columns match contract
    print("Verifying feature order...")
    feature_cols_list = list(feature_cols_mt5)
    
    if feature_cols_list != feature_order:
        print("  ✗ ERROR: Feature order mismatch!")
        print(f"  Expected: {feature_order[:3]}...")
        print(f"  Got:      {feature_cols_list[:3]}...")
        return False
    
    print("  ✓ Feature order matches contract")
    print()
    
    # Load ONNX model
    onnx_path = project_root / "python_training" / "models" / "lightgbm_xauusd.onnx"
    print(f"Loading ONNX model: {onnx_path.name}")
    
    if not onnx_path.exists():
        print(f"  ✗ ERROR: ONNX model not found")
        print("  Run: python python_training/export_onnx_mt5.py")
        return False
    
    print("  ✓ ONNX model loaded")
    print()
    
    # Run ONNX inference on MT5 features
    print("Running Python ONNX inference...")
    features_array = merged_df[feature_order].values
    
    python_predictions = []
    for i in range(len(features_array)):
        feature_row = features_array[i:i+1]  # Keep 2D shape [1, 26]
        output = run_onnx_inference(onnx_path, feature_row)
        python_predictions.append(output[0])  # Extract [3] from [1, 3]
    
    python_predictions = np.array(python_predictions)
    print(f"  ✓ Completed {len(python_predictions):,} inferences")
    print()
    
    # Compare with MT5 predictions
    print("Comparing Python vs MT5 predictions...")
    
    mt5_probs = merged_df[["p_short", "p_hold", "p_long"]].values
    
    # Calculate differences
    prob_diff = np.abs(python_predictions - mt5_probs)
    max_diff = prob_diff.max()
    mean_diff = prob_diff.mean()
    
    # Class predictions
    python_classes = np.argmax(python_predictions, axis=1)
    mt5_classes = merged_df["best_class"].values
    
    class_match_rate = (python_classes == mt5_classes).mean() * 100
    
    print("Results:")
    print(f"  Max probability difference: {max_diff:.10f}")
    print(f"  Mean probability difference: {mean_diff:.10f}")
    print(f"  Class prediction match rate: {class_match_rate:.2f}%")
    print()
    
    # Check specific mismatches
    if max_diff > 1e-5:
        print("⚠ WARNING: Large probability differences detected")
        print()
        worst_idx = prob_diff.max(axis=1).argmax()
        print(f"Worst mismatch at row {worst_idx}:")
        print(f"  Time: {merged_df.iloc[worst_idx]['time']}")
        print(f"  Python: {python_predictions[worst_idx]}")
        print(f"  MT5:    {mt5_probs[worst_idx]}")
        print(f"  Diff:   {prob_diff[worst_idx]}")
        print()
    
    if class_match_rate < 99.0:
        print("⚠ WARNING: Class predictions don't match")
        print()
        mismatch_indices = np.where(python_classes != mt5_classes)[0]
        print(f"Mismatches: {len(mismatch_indices)} out of {len(python_classes)}")
        print()
        print("First 5 mismatches:")
        for idx in mismatch_indices[:5]:
            print(f"  Row {idx}: Python={python_classes[idx]}, MT5={mt5_classes[idx]}")
        print()
    
    # Final verdict
    print("=" * 70)
    if max_diff < 1e-5 and class_match_rate >= 99.9:
        print("✅ PARITY TEST PASSED")
        print("   Python and MT5 ONNX inferences match within tolerance")
    elif max_diff < 1e-4 and class_match_rate >= 99.0:
        print("⚠ PARITY TEST: ACCEPTABLE")
        print("   Minor differences detected but likely acceptable")
    else:
        print("❌ PARITY TEST FAILED")
        print("   Significant differences between Python and MT5")
        print()
        print("Possible causes:")
        print("  - Feature calculation differences")
        print("  - Different ONNX runtime versions")
        print("  - Floating-point precision issues")
        print("  - MT5 indicator settings mismatch")
    
    print("=" * 70)
    print()
    
    return max_diff < 1e-4 and class_match_rate >= 99.0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
