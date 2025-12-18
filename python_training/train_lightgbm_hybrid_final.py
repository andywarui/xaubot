"""
Hybrid LightGBM Training: Uses transformer signals + base features.

Input: 30 features from hybrid_features_{train,val,test}.parquet
  - 26 base features (ATR, RSI, higher TF context, etc.)
  - 3 transformer probability outputs (p_short, p_hold, p_long)
  - 1 label column

Output: ONNX model for MT5 integration

Usage:
    python train_lightgbm_hybrid_final.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import pickle
import json

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path("/workspace/xaubot/data/processed")
MODEL_DIR = Path("/workspace/xaubot/python_training/models")
MODEL_DIR.mkdir(exist_ok=True)

# =============================================================================
# Load Data
# =============================================================================
def load_hybrid_data():
    """Load the hybrid feature parquet files."""
    print("=" * 70)
    print("🚀 Hybrid LightGBM Training")
    print("=" * 70)
    
    print("\n[1/4] Loading hybrid features...")
    
    train_df = pd.read_parquet(DATA_DIR / "hybrid_features_train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "hybrid_features_val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "hybrid_features_test.parquet")
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    # Separate features and labels (exclude non-numeric columns)
    label_col = 'label'
    exclude_cols = ['label', 'time', 'close']  # time is Timestamp, close is target-related
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols[:5]}...{feature_cols[-3:]}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values.astype(int)
    
    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values.astype(int)
    
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values.astype(int)
    
    print(f"\n  Label distribution (train):")
    for label in [0, 1, 2]:
        count = (y_train == label).sum()
        pct = count / len(y_train) * 100
        name = ['SHORT', 'HOLD', 'LONG'][label]
        print(f"    {label} ({name}): {count:,} ({pct:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

# =============================================================================
# Train LightGBM
# =============================================================================
def train_lightgbm(X_train, y_train, X_val, y_val, feature_cols):
    """Train LightGBM classifier."""
    print("\n[2/4] Training LightGBM...")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)
    
    # LightGBM parameters optimized for trading
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,            # Moderate complexity
        'max_depth': 7,              # Limit depth for speed
        'learning_rate': 0.1,        # Fast learning
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'min_child_samples': 1000,   # Higher = faster (less splits)
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': 1,                # Show some output
        'seed': 42,
        'n_jobs': -1,                # Use all CPU cores
        'max_bin': 127,              # Fewer bins = faster
    }
    
    # Train with early stopping - show progress every 10 rounds
    print("  Starting training (this may take 5-10 minutes)...")
    print("  Progress will show every 10 rounds...")
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,  # Reduced for faster training
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=2)  # More frequent logging
        ]
    )
    
    print(f"\n  Best iteration: {model.best_iteration}")
    
    return model

# =============================================================================
# Evaluate Model
# =============================================================================
def evaluate_model(model, X, y, split_name="Test"):
    """Evaluate model performance."""
    # Get predictions
    y_prob = model.predict(X)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n  {split_name} Accuracy: {accuracy:.2%}")
    print(f"\n  Classification Report ({split_name}):")
    print(classification_report(y, y_pred, target_names=['SHORT', 'HOLD', 'LONG']))
    
    print(f"  Confusion Matrix ({split_name}):")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    return accuracy, y_pred, y_prob

# =============================================================================
# Export to ONNX
# =============================================================================
def export_to_onnx(model, feature_cols):
    """Export LightGBM model to ONNX format."""
    print("\n[4/4] Exporting to ONNX...")
    
    try:
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        # Define input shape
        initial_type = [('input', FloatTensorType([None, len(feature_cols)]))]
        
        # Convert to ONNX
        onnx_model = convert_lightgbm(model, initial_types=initial_type)
        
        # Save ONNX model
        onnx_path = MODEL_DIR / "hybrid_lightgbm.onnx"
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"  ✅ Saved ONNX model: {onnx_path}")
        
        # Validate ONNX model
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"  Input name: {input_name}, shape: {session.get_inputs()[0].shape}")
        print(f"  Output name: {output_name}, shape: {session.get_outputs()[0].shape}")
        
        return onnx_path
        
    except ImportError as e:
        print(f"  ⚠️ ONNX export skipped (missing dependency): {e}")
        print("  Install with: pip install onnxmltools onnxruntime")
        return None

# =============================================================================
# Main
# =============================================================================
def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_hybrid_data()
    
    # Train model
    model = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
    
    # Evaluate
    print("\n[3/4] Evaluating model...")
    val_acc, _, _ = evaluate_model(model, X_val, y_val, "Validation")
    test_acc, _, _ = evaluate_model(model, X_test, y_test, "Test")
    
    # Save model
    model_path = MODEL_DIR / "hybrid_lightgbm.txt"
    model.save_model(str(model_path))
    print(f"\n  ✅ Saved LightGBM model: {model_path}")
    
    # Save feature names
    config = {
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'best_iteration': model.best_iteration
    }
    config_path = MODEL_DIR / "hybrid_lightgbm_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✅ Saved config: {config_path}")
    
    # Export to ONNX
    export_to_onnx(model, feature_cols)
    
    # Feature importance
    print("\n" + "=" * 70)
    print("📊 Top 10 Feature Importances:")
    print("=" * 70)
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        print(f"  {feat:25s}: {imp:,.0f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 Training Complete!")
    print("=" * 70)
    print(f"  Validation Accuracy: {val_acc:.2%}")
    print(f"  Test Accuracy:       {test_acc:.2%}")
    print(f"  Best Iteration:      {model.best_iteration}")
    print(f"\nFiles saved:")
    print(f"  • {model_path}")
    print(f"  • {config_path}")
    print(f"  • {MODEL_DIR / 'hybrid_lightgbm.onnx'}")

if __name__ == "__main__":
    main()