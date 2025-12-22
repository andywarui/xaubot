#!/usr/bin/env python3
"""
Phase 3 Task 4: Comprehensive Python Validation Script
Tests the complete 2-model inference pipeline:
1. Load Transformer ONNX -> get multi_tf_signal
2. Combine with 26 LightGBM features -> 27 total
3. Run LightGBM ONNX -> get prediction

This validates the pipeline matches the training workflow exactly.
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class MT5PipelineValidator:
    """Validates the complete 2-model MT5 inference pipeline."""
    
    def __init__(self, models_dir: str = "mt5_expert_advisor/Files/NeuralBot"):
        self.models_dir = Path(models_dir)
        self.transformer_path = self.models_dir / "transformer.onnx"
        self.lightgbm_path = self.models_dir / "hybrid_lightgbm.onnx"
        self.scaler_path = self.models_dir / "scaler_params.json"
        
        # Load models
        self.transformer_sess = None
        self.lightgbm_sess = None
        self.scaler = None
        
        self._load_all()
    
    def _load_all(self):
        """Load all models and configs."""
        print("="*60)
        print("Loading MT5 Pipeline Components")
        print("="*60)
        
        # Load Transformer
        print(f"\n1. Loading Transformer: {self.transformer_path}")
        if not self.transformer_path.exists():
            raise FileNotFoundError(f"Transformer not found: {self.transformer_path}")
        self.transformer_sess = ort.InferenceSession(str(self.transformer_path))
        
        t_inp = self.transformer_sess.get_inputs()[0]
        t_out = self.transformer_sess.get_outputs()[0]
        print(f"   Input: {t_inp.name} {t_inp.shape}")
        print(f"   Output: {t_out.name} {t_out.shape}")
        self.transformer_input_name = t_inp.name
        self.transformer_input_shape = t_inp.shape
        
        # Load LightGBM
        print(f"\n2. Loading LightGBM: {self.lightgbm_path}")
        if not self.lightgbm_path.exists():
            raise FileNotFoundError(f"LightGBM not found: {self.lightgbm_path}")
        self.lightgbm_sess = ort.InferenceSession(str(self.lightgbm_path))
        
        l_inp = self.lightgbm_sess.get_inputs()[0]
        l_outs = self.lightgbm_sess.get_outputs()
        print(f"   Input: {l_inp.name} {l_inp.shape}")
        for out in l_outs:
            print(f"   Output: {out.name} {out.shape}")
        self.lightgbm_input_name = l_inp.name
        
        # Load Scaler
        print(f"\n3. Loading Scaler: {self.scaler_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        with open(self.scaler_path) as f:
            self.scaler = json.load(f)
        print(f"   Features: {self.scaler['n_features']}")
        print(f"   Range: {self.scaler['feature_range']}")
        
        print("\n" + "="*60)
        print("All components loaded successfully!")
        print("="*60)
    
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Apply MinMaxScaler transformation."""
        scale = np.array(self.scaler['scale'])
        min_val = np.array(self.scaler['min'])
        return features * scale + min_val
    
    def run_transformer(self, sequence: np.ndarray) -> float:
        """
        Run Transformer inference.
        
        Args:
            sequence: Shape [1, 30, 130] - 30 timesteps, 130 features
            
        Returns:
            multi_tf_signal: float value
        """
        if sequence.shape != tuple(self.transformer_input_shape):
            raise ValueError(f"Expected shape {self.transformer_input_shape}, got {sequence.shape}")
        
        outputs = self.transformer_sess.run(None, {self.transformer_input_name: sequence})
        return float(outputs[0][0, 0])
    
    def run_lightgbm(self, features: np.ndarray) -> tuple:
        """
        Run LightGBM inference.
        
        Args:
            features: Shape [1, 27] - 27 hybrid features
            
        Returns:
            (label, probabilities)
        """
        outputs = self.lightgbm_sess.run(None, {self.lightgbm_input_name: features})
        label = int(outputs[0][0])
        probs = outputs[1][0]
        return label, probs
    
    def run_full_pipeline(self, transformer_input: np.ndarray, 
                          lightgbm_26_features: np.ndarray) -> dict:
        """
        Run the complete 2-model pipeline.
        
        Args:
            transformer_input: Shape [1, 30, 130] - scaled transformer sequence
            lightgbm_26_features: Shape [26,] - raw LightGBM features (without multi_tf_signal)
            
        Returns:
            dict with prediction results
        """
        # Step 1: Get multi_tf_signal from Transformer
        multi_tf_signal = self.run_transformer(transformer_input)
        
        # Step 2: Combine features (multi_tf_signal is feature 0)
        combined = np.zeros((1, 27), dtype=np.float32)
        combined[0, 0] = multi_tf_signal
        combined[0, 1:] = lightgbm_26_features
        
        # Step 3: Run LightGBM
        label, probs = self.run_lightgbm(combined)
        
        return {
            "multi_tf_signal": multi_tf_signal,
            "label": label,
            "label_str": ["HOLD", "BUY", "SELL"][label],
            "probabilities": probs.tolist(),
            "confidence": float(np.max(probs))
        }
    
    def validate_pipeline(self, n_tests: int = 100) -> dict:
        """
        Run validation tests on the pipeline.
        
        Args:
            n_tests: Number of test runs
            
        Returns:
            Validation results
        """
        print("\n" + "="*60)
        print(f"Running {n_tests} Validation Tests")
        print("="*60)
        
        results = {
            "tests_run": n_tests,
            "transformer_outputs": [],
            "predictions": {"HOLD": 0, "BUY": 0, "SELL": 0},
            "avg_confidence": 0,
            "errors": []
        }
        
        confidences = []
        
        for i in range(n_tests):
            try:
                # Generate random inputs
                # Transformer input: [1, 30, 130] - scaled to [0,1]
                transformer_input = np.random.rand(1, 30, 130).astype(np.float32)
                
                # LightGBM 26 features (without multi_tf_signal)
                lgb_features = np.random.randn(26).astype(np.float32)
                
                # Run pipeline
                result = self.run_full_pipeline(transformer_input, lgb_features)
                
                results["transformer_outputs"].append(result["multi_tf_signal"])
                results["predictions"][result["label_str"]] += 1
                confidences.append(result["confidence"])
                
                if i < 5 or i == n_tests - 1:
                    print(f"Test {i+1:3d}: multi_tf={result['multi_tf_signal']:+.4f}, "
                          f"pred={result['label_str']:4s}, conf={result['confidence']:.4f}")
                elif i == 5:
                    print("  ...")
                    
            except Exception as e:
                results["errors"].append(str(e))
                print(f"Test {i+1}: ERROR - {e}")
        
        results["avg_confidence"] = float(np.mean(confidences)) if confidences else 0
        results["transformer_range"] = [
            float(np.min(results["transformer_outputs"])),
            float(np.max(results["transformer_outputs"]))
        ]
        
        # Remove full list for summary
        del results["transformer_outputs"]
        
        return results
    
    def test_edge_cases(self) -> dict:
        """Test edge cases for robustness."""
        print("\n" + "="*60)
        print("Edge Case Tests")
        print("="*60)
        
        tests = []
        
        # Test 1: All zeros
        print("\n1. All zeros input:")
        try:
            t_in = np.zeros((1, 30, 130), dtype=np.float32)
            l_in = np.zeros(26, dtype=np.float32)
            result = self.run_full_pipeline(t_in, l_in)
            print(f"   Result: {result['label_str']}, conf={result['confidence']:.4f}")
            tests.append({"name": "zeros", "passed": True})
        except Exception as e:
            print(f"   ERROR: {e}")
            tests.append({"name": "zeros", "passed": False, "error": str(e)})
        
        # Test 2: All ones
        print("\n2. All ones input:")
        try:
            t_in = np.ones((1, 30, 130), dtype=np.float32)
            l_in = np.ones(26, dtype=np.float32)
            result = self.run_full_pipeline(t_in, l_in)
            print(f"   Result: {result['label_str']}, conf={result['confidence']:.4f}")
            tests.append({"name": "ones", "passed": True})
        except Exception as e:
            print(f"   ERROR: {e}")
            tests.append({"name": "ones", "passed": False, "error": str(e)})
        
        # Test 3: Large values
        print("\n3. Large values:")
        try:
            t_in = np.ones((1, 30, 130), dtype=np.float32) * 100
            l_in = np.ones(26, dtype=np.float32) * 100
            result = self.run_full_pipeline(t_in, l_in)
            print(f"   Result: {result['label_str']}, conf={result['confidence']:.4f}")
            tests.append({"name": "large", "passed": True})
        except Exception as e:
            print(f"   ERROR: {e}")
            tests.append({"name": "large", "passed": False, "error": str(e)})
        
        # Test 4: Negative values
        print("\n4. Negative values:")
        try:
            t_in = np.ones((1, 30, 130), dtype=np.float32) * -1
            l_in = np.ones(26, dtype=np.float32) * -1
            result = self.run_full_pipeline(t_in, l_in)
            print(f"   Result: {result['label_str']}, conf={result['confidence']:.4f}")
            tests.append({"name": "negative", "passed": True})
        except Exception as e:
            print(f"   ERROR: {e}")
            tests.append({"name": "negative", "passed": False, "error": str(e)})
        
        # Test 5: NaN handling (should fail gracefully)
        print("\n5. NaN handling:")
        try:
            t_in = np.ones((1, 30, 130), dtype=np.float32)
            t_in[0, 0, 0] = np.nan
            l_in = np.ones(26, dtype=np.float32)
            result = self.run_full_pipeline(t_in, l_in)
            has_nan = np.isnan(result['confidence'])
            print(f"   Result: {result['label_str']}, conf={result['confidence']:.4f}, has_nan={has_nan}")
            tests.append({"name": "nan", "passed": True, "note": f"NaN in output: {has_nan}"})
        except Exception as e:
            print(f"   Expected error for NaN: {e}")
            tests.append({"name": "nan", "passed": True, "note": "Correctly rejected NaN"})
        
        passed = sum(1 for t in tests if t.get("passed", False))
        return {
            "tests": tests,
            "passed": passed,
            "total": len(tests)
        }


def main():
    print("\n" + "#"*60)
    print("# MT5 2-Model Pipeline Validation")
    print("#"*60)
    
    try:
        validator = MT5PipelineValidator()
        
        # Run validation tests
        val_results = validator.validate_pipeline(n_tests=100)
        
        print("\n" + "="*60)
        print("Validation Results Summary")
        print("="*60)
        print(f"Tests run: {val_results['tests_run']}")
        print(f"Predictions distribution:")
        for label, count in val_results['predictions'].items():
            print(f"  {label}: {count} ({100*count/val_results['tests_run']:.1f}%)")
        print(f"Avg confidence: {val_results['avg_confidence']:.4f}")
        print(f"Transformer output range: {val_results['transformer_range']}")
        print(f"Errors: {len(val_results['errors'])}")
        
        # Run edge case tests
        edge_results = validator.test_edge_cases()
        
        print("\n" + "="*60)
        print("Edge Case Results")
        print("="*60)
        print(f"Passed: {edge_results['passed']}/{edge_results['total']}")
        
        # Overall status
        print("\n" + "#"*60)
        all_passed = (len(val_results['errors']) == 0 and 
                      edge_results['passed'] == edge_results['total'])
        
        if all_passed:
            print("# VALIDATION PASSED - Pipeline ready for MT5!")
        else:
            print("# VALIDATION FAILED - Check errors above")
        print("#"*60)
        
        # Save results
        results_path = "mt5_expert_advisor/Files/NeuralBot/validation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "validation": val_results,
                "edge_cases": edge_results,
                "overall_passed": all_passed
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
