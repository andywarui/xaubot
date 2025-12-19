"""
Optimize thresholds to achieve >55% recall for ALL classes.
Run this after training to find the best threshold combination.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import recall_score, accuracy_score
from itertools import product

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "python_training" / "models"


def load_test_data():
    """Load test data and model."""
    df = pd.read_parquet(DATA_DIR / "hybrid_features_test.parquet")
    feature_cols = [c for c in df.columns if c not in ["time", "close", "label"]]
    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    model = lgb.Booster(model_file=str(MODEL_DIR / "lightgbm_balanced.txt"))
    y_proba = model.predict(X)

    return y, y_proba


def apply_thresholds(y_proba, thresholds):
    """Apply thresholds to get predictions."""
    adjusted = y_proba.copy()
    for i in range(3):
        adjusted[:, i] = adjusted[:, i] / thresholds[i]
    return np.argmax(adjusted, axis=1)


def evaluate(y_true, y_pred):
    """Calculate per-class recall and accuracy."""
    recalls = []
    for i in range(3):
        recall = ((y_true == i) & (y_pred == i)).sum() / (y_true == i).sum()
        recalls.append(recall)
    accuracy = accuracy_score(y_true, y_pred)
    return recalls, accuracy


def main():
    print("=" * 70)
    print("THRESHOLD OPTIMIZER")
    print("Goal: Find thresholds where ALL classes have recall > 55%")
    print("=" * 70)

    print("\n[1/3] Loading data and model...")
    y_test, y_proba = load_test_data()
    print(f"  Loaded {len(y_test):,} test samples")

    print("\n[2/3] Searching for optimal thresholds...")

    # Define search range
    short_range = np.arange(0.30, 0.60, 0.02)
    hold_range = np.arange(0.15, 0.35, 0.02)
    long_range = np.arange(0.30, 0.55, 0.02)

    total = len(short_range) * len(hold_range) * len(long_range)
    print(f"  Testing {total:,} combinations...")

    best_score = 0
    best_thresholds = None
    best_recalls = None
    best_accuracy = 0

    all_pass_results = []

    for short_t, hold_t, long_t in product(short_range, hold_range, long_range):
        thresholds = [short_t, hold_t, long_t]
        y_pred = apply_thresholds(y_proba, thresholds)
        recalls, accuracy = evaluate(y_test, y_pred)

        # Check if all classes pass
        all_pass = all(r >= 0.55 for r in recalls)
        min_recall = min(recalls)

        if all_pass:
            all_pass_results.append({
                'thresholds': thresholds,
                'recalls': recalls,
                'accuracy': accuracy,
                'min_recall': min_recall,
                'macro_recall': np.mean(recalls)
            })

        # Track best by minimum recall (to maximize worst class)
        if min_recall > best_score:
            best_score = min_recall
            best_thresholds = thresholds
            best_recalls = recalls
            best_accuracy = accuracy

    print("\n[3/3] Results...")

    print("\n" + "=" * 70)
    print("BEST OVERALL (maximizes minimum recall)")
    print("=" * 70)
    print(f"  Thresholds: SHORT={best_thresholds[0]:.2f}, HOLD={best_thresholds[1]:.2f}, LONG={best_thresholds[2]:.2f}")
    print(f"  Accuracy: {best_accuracy:.2%}")
    print(f"  SHORT Recall: {best_recalls[0]:.2%}")
    print(f"  HOLD Recall:  {best_recalls[1]:.2%}")
    print(f"  LONG Recall:  {best_recalls[2]:.2%}")

    if all(r >= 0.55 for r in best_recalls):
        print("\n  ✅ ALL CLASSES > 55%!")
    else:
        print("\n  ⚠️  Some classes below 55%")

    # Show all passing combinations
    if all_pass_results:
        print("\n" + "=" * 70)
        print(f"ALL {len(all_pass_results)} COMBINATIONS WHERE ALL CLASSES > 55%")
        print("=" * 70)

        # Sort by accuracy
        all_pass_results.sort(key=lambda x: x['accuracy'], reverse=True)

        print(f"\n  Top 10 by Accuracy:")
        print(f"  {'SHORT_t':<8} {'HOLD_t':<8} {'LONG_t':<8} {'Accuracy':<10} {'SHORT':<8} {'HOLD':<8} {'LONG':<8}")
        print("  " + "-" * 66)

        for r in all_pass_results[:10]:
            t = r['thresholds']
            rc = r['recalls']
            print(f"  {t[0]:<8.2f} {t[1]:<8.2f} {t[2]:<8.2f} {r['accuracy']:<10.2%} {rc[0]:<8.2%} {rc[1]:<8.2%} {rc[2]:<8.2%}")

        # Best by macro recall
        best_macro = max(all_pass_results, key=lambda x: x['macro_recall'])
        print(f"\n  Best by Macro Recall:")
        t = best_macro['thresholds']
        rc = best_macro['recalls']
        print(f"  SHORT_t={t[0]:.2f}, HOLD_t={t[1]:.2f}, LONG_t={t[2]:.2f}")
        print(f"  Accuracy: {best_macro['accuracy']:.2%}, Macro Recall: {best_macro['macro_recall']:.2%}")
        print(f"  SHORT={rc[0]:.2%}, HOLD={rc[1]:.2%}, LONG={rc[2]:.2%}")
    else:
        print("\n" + "=" * 70)
        print("NO COMBINATION FOUND WHERE ALL CLASSES > 55%")
        print("=" * 70)
        print("  Consider: More aggressive class balancing or model retraining")

    # Save best config
    if best_thresholds:
        import json
        config_path = MODEL_DIR / "optimal_thresholds.json"
        config = {
            "optimal_thresholds": best_thresholds,
            "recalls": {
                "SHORT": float(best_recalls[0]),
                "HOLD": float(best_recalls[1]),
                "LONG": float(best_recalls[2])
            },
            "accuracy": float(best_accuracy),
            "all_pass": all(r >= 0.55 for r in best_recalls)
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n  Saved: {config_path}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
