#!/usr/bin/env python3
"""
Export Scaler Parameters to JSON for MT5
Converts sklearn MinMaxScaler to JSON format for MQL5.
"""

import pickle
import json
import numpy as np
from pathlib import Path
import warnings

# Suppress version warning
warnings.filterwarnings('ignore', category=UserWarning)


def export_scaler_to_json(scaler_path: str, output_path: str, feature_names_path: str = None) -> dict:
    """
    Export MinMaxScaler parameters to JSON.
    
    For MinMaxScaler: X_scaled = (X - data_min) / data_range * scale + min_
    Simplified: X_scaled = X * scale_ + min_
    
    Args:
        scaler_path: Path to pickle file
        output_path: Path for JSON output
        feature_names_path: Optional path to feature names JSON
        
    Returns:
        dict with scaler info
    """
    print(f"Loading scaler from: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    scaler_type = type(scaler).__name__
    print(f"Scaler type: {scaler_type}")
    
    if scaler_type != 'MinMaxScaler':
        raise ValueError(f"Expected MinMaxScaler, got {scaler_type}")
    
    n_features = scaler.n_features_in_
    print(f"Features: {n_features}")
    print(f"Feature range: {scaler.feature_range}")
    
    # Get scaler parameters
    # MinMaxScaler formula: X_scaled = X * scale_ + min_
    scale = scaler.scale_.tolist()
    min_val = scaler.min_.tolist()
    data_min = scaler.data_min_.tolist()
    data_max = scaler.data_max_.tolist()
    data_range = scaler.data_range_.tolist()
    
    # Load feature names if available
    feature_names = None
    if feature_names_path and Path(feature_names_path).exists():
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        print(f"Loaded {len(feature_names)} feature names")
    
    # Create JSON structure
    scaler_json = {
        "scaler_type": scaler_type,
        "n_features": n_features,
        "feature_range": list(scaler.feature_range),
        "scale": scale,
        "min": min_val,
        "data_min": data_min,
        "data_max": data_max,
        "data_range": data_range,
        "usage": {
            "description": "To scale: X_scaled = X * scale[i] + min[i]",
            "inverse": "To unscale: X_original = (X_scaled - min[i]) / scale[i]"
        }
    }
    
    if feature_names:
        scaler_json["feature_names"] = feature_names
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(scaler_json, f, indent=2)
    
    print(f"Saved to: {output_path}")
    
    # Verify by testing scaling
    print("\n=== Verification ===")
    test_data = np.random.randn(1, n_features).astype(np.float32)
    
    # Scale with sklearn
    sklearn_scaled = scaler.transform(test_data)
    
    # Scale with our parameters
    scale_arr = np.array(scale)
    min_arr = np.array(min_val)
    manual_scaled = test_data * scale_arr + min_arr
    
    # Compare
    max_diff = np.max(np.abs(sklearn_scaled - manual_scaled))
    print(f"Max difference: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✓ Verification PASSED")
    else:
        print("✗ Verification FAILED")
        
    return {
        "n_features": n_features,
        "output_path": output_path,
        "max_diff": float(max_diff)
    }


def main():
    scaler_path = "python_training/models/multi_tf_scaler.pkl"
    output_path = "mt5_expert_advisor/Files/NeuralBot/scaler_params.json"
    
    # Optional: feature names from config
    feature_names_path = "config/features_order.json"
    
    try:
        info = export_scaler_to_json(scaler_path, output_path, feature_names_path)
        
        print(f"\n{'='*60}")
        print("SUCCESS: Scaler exported to JSON!")
        print(f"{'='*60}")
        print(f"Features: {info['n_features']}")
        print(f"Output: {info['output_path']}")
        print(f"Verification diff: {info['max_diff']:.2e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
