"""
Train LightGBM with Class Balancing for improved HOLD/LONG recall.
CPU-optimized with progress display.

Strategies:
1. Class weights (inverse frequency)
2. Per-class threshold optimization

Target: All class recalls > 55%
"""
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "python_training" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_hybrid_features(split):
    """Load hybrid features with multi_tf_signal."""
    path = DATA_DIR / f"hybrid_features_{split}.parquet"
    df = pd.read_parquet(path)
    
    feature_cols = [c for c in df.columns if c not in ["time", "close", "label"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    
    return X, y, feature_cols


def compute_sample_weights(y, strategy="balanced"):
    """Compute sample weights for class balancing."""
    classes = np.unique(y)
    
    if strategy == "balanced":
        class_weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[label] for label in y])
        
    elif strategy == "sqrt_balanced":
        counts = np.bincount(y)
        weights = np.sqrt(len(y) / (len(classes) * counts))
        weight_dict = dict(zip(classes, weights))
        sample_weights = np.array([weight_dict[label] for label in y])
    
    else:
        sample_weights = np.ones(len(y))
    
    return sample_weights


def optimize_thresholds(y_true, y_proba):
    """Find optimal probability thresholds per class with progress bar."""
    n_classes = y_proba.shape[1]
    threshold_range = np.arange(0.20, 0.55, 0.025)
    
    best_score = 0
    best_thresh = [0.33, 0.33, 0.33]
    
    total_iterations = len(threshold_range) ** 3
    
    print(f"    Searching {total_iterations:,} threshold combinations...")
    
    with tqdm(total=total_iterations, desc="    Threshold Search", unit="combo") as pbar:
        for short_t in threshold_range:
            for hold_t in threshold_range:
                for long_t in threshold_range:
                    thresholds = [short_t, hold_t, long_t]
                    
                    adjusted_proba = y_proba.copy()
                    for i in range(n_classes):
                        adjusted_proba[:, i] = adjusted_proba[:, i] / thresholds[i]
                    
                    y_pred = np.argmax(adjusted_proba, axis=1)
                    score = recall_score(y_true, y_pred, average='macro')
                    
                    if score > best_score:
                        best_score = score
                        best_thresh = thresholds
                    
                    pbar.update(1)
    
    print(f"\n    Best thresholds: SHORT={best_thresh[0]:.3f}, HOLD={best_thresh[1]:.3f}, LONG={best_thresh[2]:.3f}")
    print(f"    Best macro recall: {best_score:.4f}")
    
    return best_thresh


def apply_thresholds(y_proba, thresholds):
    """Apply thresholds to get predictions."""
    adjusted_proba = y_proba.copy()
    for i in range(len(thresholds)):
        adjusted_proba[:, i] = adjusted_proba[:, i] / thresholds[i]
    return np.argmax(adjusted_proba, axis=1)


class ProgressCallback:
    """Custom callback for LightGBM progress display."""
    def __init__(self, total_rounds):
        self.total_rounds = total_rounds
        self.pbar = None
        self.start_time = None
        
    def __call__(self, env):
        if self.pbar is None:
            self.pbar = tqdm(total=self.total_rounds, desc="    Training", unit="round")
            self.start_time = time.time()
        
        self.pbar.update(1)
        
        if env.evaluation_result_list:
            val_loss = env.evaluation_result_list[-1][2]
            elapsed = time.time() - self.start_time
            rate = (env.iteration + 1) / elapsed if elapsed > 0 else 0
            eta = (self.total_rounds - env.iteration - 1) / rate if rate > 0 else 0
            self.pbar.set_postfix({
                'val_loss': f'{val_loss:.4f}',
                'ETA': f'{eta:.0f}s'
            })
        
        if env.iteration == self.total_rounds - 1:
            self.pbar.close()


def train_lightgbm_balanced(X_train, y_train, X_val, y_val, feature_cols, 
                            weight_strategy="balanced"):
    """Train LightGBM with class balancing."""
    
    print(f"  Computing sample weights (strategy: {weight_strategy})...")
    sample_weights = compute_sample_weights(y_train, weight_strategy)
    
    for i, name in enumerate(['SHORT', 'HOLD', 'LONG']):
        mask = y_train == i
        avg_weight = sample_weights[mask].mean()
        print(f"    {name} avg weight: {avg_weight:.3f}")
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                             feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data,
                           feature_name=feature_cols)
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
    }
    
    num_rounds = 500
    
    print(f"  Training LightGBM ({num_rounds} rounds max, CPU)...")
    
    progress_cb = ProgressCallback(num_rounds)
    
    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            progress_cb
        ]
    )
    
    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best iteration: {model.best_iteration}")
    
    return model


def evaluate_model(model, X, y, thresholds=None, set_name="Test"):
    """Evaluate model with optional threshold adjustment."""
    print(f"\n  Predicting on {set_name} set...")
    y_proba = model.predict(X)
    
    if thresholds:
        y_pred = apply_thresholds(y_proba, thresholds)
    else:
        y_pred = np.argmax(y_proba, axis=1)
    
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n  {set_name} Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=['SHORT', 'HOLD', 'LONG']))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    print(f"\n  Per-Class Recall:")
    for i, name in enumerate(['SHORT', 'HOLD', 'LONG']):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"    {name}: {recall:.2%}")
    
    return y_pred, y_proba, accuracy


def main():
    total_start = time.time()
    
    print("=" * 70)
    print("Balanced LightGBM Training - Phase 1 Model Optimization")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading hybrid features...")
    load_start = time.time()
    
    X_train, y_train, feature_cols = load_hybrid_features("train")
    X_val, y_val, _ = load_hybrid_features("val")
    X_test, y_test, _ = load_hybrid_features("test")
    
    print(f"  Loaded in {time.time() - load_start:.1f}s")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Features: {len(feature_cols)}")
    
    print("\n  Class Distribution (Train):")
    for i, name in enumerate(['SHORT', 'HOLD', 'LONG']):
        count = (y_train == i).sum()
        pct = count / len(y_train) * 100
        print(f"    {name}: {count:,} ({pct:.1f}%)")
    
    # Train with balanced weights
    print("\n[2/5] Training with class balancing...")
    model = train_lightgbm_balanced(
        X_train, y_train, X_val, y_val, feature_cols,
        weight_strategy="balanced"
    )
    
    # Evaluate before threshold optimization
    print("\n[3/5] Evaluating (before threshold optimization)...")
    y_pred_val, y_proba_val, acc_before = evaluate_model(
        model, X_val, y_val, thresholds=None, set_name="Validation"
    )
    
    # Optimize thresholds
    print("\n[4/5] Optimizing per-class thresholds...")
    optimal_thresholds = optimize_thresholds(y_val, y_proba_val)
    
    # Evaluate after threshold optimization
    print("\n[5/5] Final Evaluation (with optimized thresholds)...")
    
    print("\n" + "=" * 70)
    print("VALIDATION SET (with thresholds)")
    print("=" * 70)
    y_pred_val_opt, _, acc_val = evaluate_model(
        model, X_val, y_val, thresholds=optimal_thresholds, set_name="Validation"
    )
    
    print("\n" + "=" * 70)
    print("TEST SET (with thresholds)")
    print("=" * 70)
    y_pred_test, y_proba_test, acc_test = evaluate_model(
        model, X_test, y_test, thresholds=optimal_thresholds, set_name="Test"
    )
    
    # Save model and config
    print("\n" + "=" * 70)
    print("Saving Model & Config")
    print("=" * 70)
    
    model_path = MODEL_DIR / "lightgbm_balanced.txt"
    model.save_model(str(model_path))
    print(f"  Model saved: {model_path}")
    
    config = {
        "feature_cols": feature_cols,
        "optimal_thresholds": optimal_thresholds,
        "val_accuracy": float(acc_val),
        "test_accuracy": float(acc_test),
        "weight_strategy": "balanced",
        "class_names": ["SHORT", "HOLD", "LONG"],
        "best_iteration": model.best_iteration
    }
    
    config_path = MODEL_DIR / "lightgbm_balanced_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("Top 10 Feature Importances")
    print("=" * 70)
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp[:10]:
        print(f"  {name:25s}: {imp:,.0f}")
    
    # Summary & Success Check
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Validation Accuracy: {acc_val:.2%}")
    print(f"  Test Accuracy:       {acc_test:.2%}")
    print(f"  Optimal Thresholds:  SHORT={optimal_thresholds[0]:.3f}, "
          f"HOLD={optimal_thresholds[1]:.3f}, LONG={optimal_thresholds[2]:.3f}")
    
    y_pred_test_final = apply_thresholds(y_proba_test, optimal_thresholds)
    
    print("\n  Per-Class Recall (Target: >55%):")
    all_passed = True
    for i, name in enumerate(['SHORT', 'HOLD', 'LONG']):
        recall = ((y_test == i) & (y_pred_test_final == i)).sum() / (y_test == i).sum()
        status = "PASS" if recall >= 0.55 else "FAIL"
        if recall < 0.55:
            all_passed = False
        print(f"    {name}: {recall:.2%} [{status}]")
    
    if all_passed:
        print("\n  SUCCESS: All class recalls > 55%!")
    else:
        print("\n  WARNING: Some recalls below 55% - may need further tuning")
    
    total_elapsed = time.time() - total_start
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    print("\n" + "=" * 70)
    print("Phase 1 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()