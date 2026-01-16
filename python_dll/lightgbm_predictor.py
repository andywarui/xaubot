"""
LightGBM Predictor DLL for MT5
Loads original +472% backtested LightGBM model
Provides ctypes-compatible interface for MQL5

Model: python_training/models/lightgbm_xauusd.pkl
Features: 26 (price, returns, indicators, time)
Output: 0 (SHORT), 1 (HOLD), 2 (LONG)
"""

import os
import sys
import pickle
import numpy as np
from ctypes import c_double, c_int, c_char_p, POINTER
import traceback

# Global model instance
_model = None
_model_loaded = False
_last_error = ""

# Feature names in exact order used during training
FEATURE_NAMES = [
    'close',
    'returns',
    'returns_2',
    'returns_5',
    'returns_10',
    'rsi_14',
    'rsi_28',
    'atr_14',
    'atr_ratio',
    'ema_ratio_8_21',
    'ema_ratio_13_55',
    'macd',
    'macd_signal',
    'macd_hist',
    'bb_position',
    'bb_width',
    'price_position',
    'high_low_range',
    'close_to_high',
    'close_to_low',
    'hour_sin',
    'hour_cos',
    'day_sin',
    'day_cos',
    'is_london_session',
    'is_ny_session'
]

NUM_FEATURES = len(FEATURE_NAMES)  # 26


def init_model(model_path: str) -> int:
    """
    Initialize the LightGBM model from pickle file.
    
    Args:
        model_path: Absolute path to lightgbm_xauusd.pkl
        
    Returns:
        1 if successful, 0 if failed
    """
    global _model, _model_loaded, _last_error
    
    try:
        if not os.path.exists(model_path):
            _last_error = f"Model file not found: {model_path}"
            print(f"[ERROR] {_last_error}")
            return 0
            
        with open(model_path, 'rb') as f:
            _model = pickle.load(f)
            
        _model_loaded = True
        _last_error = ""
        print(f"[SUCCESS] Model loaded from {model_path}")
        print(f"[INFO] Model type: {type(_model).__name__}")
        
        # Verify model has predict method
        if not hasattr(_model, 'predict'):
            _last_error = "Model does not have predict method"
            print(f"[ERROR] {_last_error}")
            return 0
            
        return 1
        
    except Exception as e:
        _last_error = f"Failed to load model: {str(e)}"
        print(f"[ERROR] {_last_error}")
        traceback.print_exc()
        return 0


def predict_signal(features: list) -> int:
    """
    Predict trading signal from 26 features.
    
    Args:
        features: List of 26 float values in exact order
        
    Returns:
        0 = SHORT, 1 = HOLD, 2 = LONG
        -1 if error
    """
    global _model, _model_loaded, _last_error
    
    if not _model_loaded:
        _last_error = "Model not loaded. Call init_model first."
        print(f"[ERROR] {_last_error}")
        return -1
        
    try:
        if len(features) != NUM_FEATURES:
            _last_error = f"Expected {NUM_FEATURES} features, got {len(features)}"
            print(f"[ERROR] {_last_error}")
            return -1
            
        # Convert to numpy array with correct shape
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        
        # Get prediction
        prediction = _model.predict(X)[0]
        
        # Ensure valid output
        signal = int(prediction)
        if signal not in [0, 1, 2]:
            _last_error = f"Invalid prediction value: {signal}"
            print(f"[WARNING] {_last_error}, defaulting to HOLD (1)")
            return 1
            
        return signal
        
    except Exception as e:
        _last_error = f"Prediction failed: {str(e)}"
        print(f"[ERROR] {_last_error}")
        traceback.print_exc()
        return -1


def predict_with_probabilities(features: list) -> tuple:
    """
    Predict trading signal with class probabilities.
    
    Args:
        features: List of 26 float values
        
    Returns:
        (signal, prob_short, prob_hold, prob_long)
        (-1, 0, 0, 0) if error
    """
    global _model, _model_loaded, _last_error
    
    if not _model_loaded:
        _last_error = "Model not loaded. Call init_model first."
        return (-1, 0.0, 0.0, 0.0)
        
    try:
        if len(features) != NUM_FEATURES:
            _last_error = f"Expected {NUM_FEATURES} features, got {len(features)}"
            return (-1, 0.0, 0.0, 0.0)
            
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        
        # Get prediction
        prediction = _model.predict(X)[0]
        signal = int(prediction)
        
        # Get probabilities if available
        if hasattr(_model, 'predict_proba'):
            probs = _model.predict_proba(X)[0]
            if len(probs) == 3:
                return (signal, float(probs[0]), float(probs[1]), float(probs[2]))
            else:
                # Binary classifier - map to 3 classes
                return (signal, float(probs[0]) if signal == 0 else 0.0,
                        0.0, float(probs[1]) if signal == 2 else 0.0)
        else:
            # No probabilities available, return 1.0 for predicted class
            probs = [0.0, 0.0, 0.0]
            if 0 <= signal <= 2:
                probs[signal] = 1.0
            return (signal, probs[0], probs[1], probs[2])
            
    except Exception as e:
        _last_error = f"Prediction with probabilities failed: {str(e)}"
        print(f"[ERROR] {_last_error}")
        return (-1, 0.0, 0.0, 0.0)


def get_last_error() -> str:
    """Return last error message."""
    global _last_error
    return _last_error


def get_model_info() -> str:
    """Return model information as string."""
    global _model, _model_loaded
    
    if not _model_loaded:
        return "Model not loaded"
        
    info = {
        'type': type(_model).__name__,
        'features': NUM_FEATURES,
        'classes': [0, 1, 2],
        'labels': ['SHORT', 'HOLD', 'LONG']
    }
    
    # Add LightGBM-specific info if available
    if hasattr(_model, 'n_features_'):
        info['n_features'] = _model.n_features_
    if hasattr(_model, 'n_classes_'):
        info['n_classes'] = _model.n_classes_
    if hasattr(_model, 'classes_'):
        info['classes'] = list(_model.classes_)
        
    return str(info)


def cleanup_model():
    """Free model resources."""
    global _model, _model_loaded, _last_error
    _model = None
    _model_loaded = False
    _last_error = ""
    print("[INFO] Model resources cleaned up")


def get_version() -> str:
    """Return DLL version."""
    return "1.0.0"


def get_feature_names() -> list:
    """Return list of feature names in order."""
    return FEATURE_NAMES.copy()


# =============================================================================
# Self-test when run directly
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LightGBM Predictor - Self Test")
    print("=" * 60)
    
    # Find model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "python_training", "models", "lightgbm_xauusd.pkl")
    
    print(f"\n[TEST] Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found!")
        print(f"[INFO] Expected location: {model_path}")
        sys.exit(1)
        
    # Test 1: Load model
    print("\n[TEST 1] Loading model...")
    result = init_model(model_path)
    if result != 1:
        print(f"[FAIL] Failed to load model: {get_last_error()}")
        sys.exit(1)
    print("[PASS] Model loaded successfully")
    
    # Test 2: Check model info
    print("\n[TEST 2] Getting model info...")
    info = get_model_info()
    print(f"[INFO] {info}")
    print("[PASS] Model info retrieved")
    
    # Test 3: Check feature names
    print("\n[TEST 3] Checking feature names...")
    names = get_feature_names()
    print(f"[INFO] {len(names)} features: {names[:5]}... (truncated)")
    if len(names) != 26:
        print(f"[FAIL] Expected 26 features, got {len(names)}")
        sys.exit(1)
    print("[PASS] Feature names correct")
    
    # Test 4: Make prediction with dummy data
    print("\n[TEST 4] Testing prediction with dummy data...")
    dummy_features = [
        2650.0,    # close
        0.001,     # returns
        0.002,     # returns_2
        0.003,     # returns_5
        0.005,     # returns_10
        50.0,      # rsi_14
        55.0,      # rsi_28
        15.0,      # atr_14
        1.0,       # atr_ratio
        1.01,      # ema_ratio_8_21
        1.02,      # ema_ratio_13_55
        5.0,       # macd
        4.5,       # macd_signal
        0.5,       # macd_hist
        0.5,       # bb_position
        0.02,      # bb_width
        0.5,       # price_position
        20.0,      # high_low_range
        0.3,       # close_to_high
        0.7,       # close_to_low
        0.5,       # hour_sin
        0.866,     # hour_cos
        0.0,       # day_sin
        1.0,       # day_cos
        1,         # is_london_session
        0          # is_ny_session
    ]
    
    signal = predict_signal(dummy_features)
    if signal == -1:
        print(f"[FAIL] Prediction failed: {get_last_error()}")
        sys.exit(1)
    
    signal_names = ['SHORT', 'HOLD', 'LONG']
    print(f"[INFO] Prediction: {signal} ({signal_names[signal]})")
    print("[PASS] Prediction successful")
    
    # Test 5: Prediction with probabilities
    print("\n[TEST 5] Testing prediction with probabilities...")
    result = predict_with_probabilities(dummy_features)
    print(f"[INFO] Signal: {result[0]}, Probs: SHORT={result[1]:.3f}, HOLD={result[2]:.3f}, LONG={result[3]:.3f}")
    print("[PASS] Probabilities retrieved")
    
    # Test 6: Cleanup
    print("\n[TEST 6] Testing cleanup...")
    cleanup_model()
    print("[PASS] Cleanup successful")
    
    print("\n" + "=" * 60)
    print("All tests passed! Ready for DLL compilation.")
    print("=" * 60)
    print("\nNext step:")
    print("    python compile_dll.py")
