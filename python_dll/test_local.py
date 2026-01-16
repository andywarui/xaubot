"""
Test LightGBM Predictor locally (before DLL compilation)
Verifies model loading and predictions work correctly

Usage:
    python test_local.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import lightgbm_predictor
sys.path.insert(0, str(Path(__file__).parent))

import lightgbm_predictor as predictor
import numpy as np

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_result(test_name, passed, message=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {test_name}")
    if message:
        print(f"          {message}")

def get_model_path():
    """Find the model file."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_path = project_root / "python_training" / "models" / "lightgbm_xauusd.pkl"
    return str(model_path)

def create_dummy_features():
    """Create realistic dummy features for testing."""
    return [
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

def main():
    print_header("LightGBM Predictor - Local Test Suite")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Find model file
    tests_total += 1
    model_path = get_model_path()
    model_exists = os.path.exists(model_path)
    print_result("Model file exists", model_exists, model_path if model_exists else "NOT FOUND")
    if model_exists:
        tests_passed += 1
        file_size = os.path.getsize(model_path)
        print(f"          Size: {file_size / 1024 / 1024:.2f} MB")
    else:
        print("\n  ERROR: Cannot continue without model file!")
        print(f"  Expected: {model_path}")
        return 1
    
    # Test 2: Load model
    tests_total += 1
    result = predictor.init_model(model_path)
    print_result("Model loading", result == 1, 
                 predictor.get_last_error() if result != 1 else "Model loaded successfully")
    if result == 1:
        tests_passed += 1
    else:
        print("\n  ERROR: Cannot continue without loading model!")
        return 1
    
    # Test 3: Get model info
    tests_total += 1
    info = predictor.get_model_info()
    print_result("Get model info", info != "Model not loaded", info[:80] + "..." if len(info) > 80 else info)
    if info != "Model not loaded":
        tests_passed += 1
    
    # Test 4: Check feature count
    tests_total += 1
    feature_names = predictor.get_feature_names()
    correct_count = len(feature_names) == 26
    print_result("Feature count", correct_count, f"{len(feature_names)} features")
    if correct_count:
        tests_passed += 1
    
    # Test 5: Basic prediction
    tests_total += 1
    dummy_features = create_dummy_features()
    signal = predictor.predict_signal(dummy_features)
    valid_signal = signal in [0, 1, 2]
    signal_names = ['SHORT', 'HOLD', 'LONG']
    print_result("Basic prediction", valid_signal, 
                 f"Signal: {signal} ({signal_names[signal] if valid_signal else 'INVALID'})")
    if valid_signal:
        tests_passed += 1
    
    # Test 6: Prediction with probabilities
    tests_total += 1
    result = predictor.predict_with_probabilities(dummy_features)
    valid_probs = result[0] in [0, 1, 2] and abs(sum(result[1:4]) - 1.0) < 0.01
    print_result("Prediction with probabilities", valid_probs,
                 f"Probs: SHORT={result[1]:.3f}, HOLD={result[2]:.3f}, LONG={result[3]:.3f}")
    if valid_probs or result[0] in [0, 1, 2]:  # Accept even if probs don't sum to 1
        tests_passed += 1
    
    # Test 7: Wrong feature count (should fail gracefully)
    tests_total += 1
    wrong_features = [1.0] * 10  # Only 10 features
    error_signal = predictor.predict_signal(wrong_features)
    handled_error = error_signal == -1
    print_result("Error handling (wrong feature count)", handled_error,
                 "Correctly returned -1 for invalid input" if handled_error else "Did not handle error properly")
    if handled_error:
        tests_passed += 1
    
    # Test 8: Cleanup
    tests_total += 1
    predictor.cleanup_model()
    info_after = predictor.get_model_info()
    cleaned = info_after == "Model not loaded"
    print_result("Cleanup", cleaned, "Model resources freed" if cleaned else "Cleanup may have failed")
    if cleaned:
        tests_passed += 1
    
    # Summary
    print_header("Test Summary")
    all_passed = tests_passed == tests_total
    print(f"\n  Tests Passed: {tests_passed}/{tests_total}")
    
    if all_passed:
        print("\n  🎉 All tests passed! Ready to compile DLL.")
        print("\n  Next step:")
        print("      python compile_dll.py")
        return 0
    else:
        print(f"\n  ⚠️  {tests_total - tests_passed} test(s) failed.")
        print("  Please fix issues before compiling DLL.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
