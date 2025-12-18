"""
Walk-Forward Optimization (WFO) for XAUUSD Neural Trading Bot.

Implements 7-fold WFO across 2019-2024 data to validate model robustness
and prevent overfitting. Each fold uses expanding/rolling window training
with out-of-sample testing.

Phase 2.2 Implementation per XAUBOT_DEVELOPMENT_PLAN.md
"""
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class WalkForwardOptimizer:
    """Walk-Forward Optimization for time-series trading models."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data" / "processed"
        self.models_dir = project_root / "python_training" / "models"
        self.results_dir = project_root / "python_training" / "backtesting" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model config
        with open(self.models_dir / "lightgbm_balanced_config.json", "r") as f:
            self.config = json.load(f)
        
        self.feature_cols = self.config["feature_cols"]
        self.thresholds = self.config["thresholds"]
        self.class_names = ["SHORT", "HOLD", "LONG"]
        
        # WFO fold definitions (year, half)
        # Format: (train_start, train_end, test_start, test_end)
        self.folds = [
            ("2019-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),  # Fold 1
            ("2019-01-01", "2021-06-30", "2021-07-01", "2021-12-31"),  # Fold 2
            ("2019-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),  # Fold 3
            ("2020-01-01", "2022-06-30", "2022-07-01", "2022-12-31"),  # Fold 4
            ("2020-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),  # Fold 5
            ("2021-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),  # Fold 6
            ("2021-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),  # Fold 7
        ]
        
    def load_all_data(self) -> pd.DataFrame:
        """Load and combine all hybrid feature data with timestamps."""
        print("📥 Loading hybrid feature data...")
        
        dfs = []
        for split in ["train", "val", "test"]:
            path = self.data_dir / f"hybrid_features_{split}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df["split"] = split
                dfs.append(df)
                print(f"   {split}: {len(df):,} rows")
        
        df_all = pd.concat(dfs, ignore_index=True)
        
        # Ensure time column is datetime
        if "time" in df_all.columns:
            df_all["time"] = pd.to_datetime(df_all["time"])
        elif "close_time" in df_all.columns:
            df_all["time"] = pd.to_datetime(df_all["close_time"])
        else:
            # If no time column, we need to load from M1 data
            print("   ⚠️ No time column found, loading from M1 parquet...")
            m1_path = self.data_dir / "xauusd_M1.parquet"
            if m1_path.exists():
                m1_df = pd.read_parquet(m1_path)
                if len(m1_df) >= len(df_all):
                    df_all["time"] = pd.to_datetime(m1_df["time"].iloc[:len(df_all)].values)
        
        df_all = df_all.sort_values("time").reset_index(drop=True)
        print(f"\n   Total: {len(df_all):,} rows")
        print(f"   Period: {df_all['time'].min()} → {df_all['time'].max()}")
        
        return df_all
    
    def get_fold_data(self, df: pd.DataFrame, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data for a specific WFO fold."""
        train_start, train_end, test_start, test_end = self.folds[fold_idx]
        
        train_mask = (df["time"] >= train_start) & (df["time"] <= train_end)
        test_mask = (df["time"] >= test_start) & (df["time"] <= test_end)
        
        return df[train_mask].copy(), df[test_mask].copy()
    
    def train_fold_model(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> lgb.Booster:
        """Train LightGBM model for a single fold."""
        X_train = df_train[self.feature_cols].values
        y_train = df_train["label"].values.astype(int)
        
        # Calculate class weights
        class_counts = np.bincount(y_train, minlength=3)
        total = len(y_train)
        class_weights = {i: total / (3 * count) if count > 0 else 1.0 
                        for i, count in enumerate(class_counts)}
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        
        val_data = None
        if df_val is not None and len(df_val) > 0:
            X_val = df_val[self.feature_cols].values
            y_val = df_val["label"].values.astype(int)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 100,
            "verbose": -1,
            "seed": 42,
        }
        
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data] if val_data else None,
            num_boost_round=200,
            callbacks=callbacks if val_data else None,
        )
        
        return model
    
    def predict_with_thresholds(self, model: lgb.Booster, X: np.ndarray) -> np.ndarray:
        """Predict using optimized per-class thresholds."""
        proba = model.predict(X)
        
        # Apply thresholds
        short_thresh = self.thresholds.get("SHORT", 0.48)
        hold_thresh = self.thresholds.get("HOLD", 0.20)
        long_thresh = self.thresholds.get("LONG", 0.40)
        
        predictions = []
        for p in proba:
            if p[0] >= short_thresh and p[0] >= p[2]:
                predictions.append(0)  # SHORT
            elif p[2] >= long_thresh and p[2] > p[0]:
                predictions.append(2)  # LONG
            elif p[1] >= hold_thresh:
                predictions.append(1)  # HOLD
            else:
                predictions.append(np.argmax(p))
        
        return np.array(predictions)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics for a fold."""
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        # Per-class metrics
        metrics = {
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "per_class": {}
        }
        
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics["per_class"][class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(cm[i, :].sum())
            }
        
        return metrics
    
    def run_wfo(self) -> Dict:
        """Execute full Walk-Forward Optimization."""
        print("=" * 70)
        print("WALK-FORWARD OPTIMIZATION")
        print("=" * 70)
        print()
        
        df_all = self.load_all_data()
        
        fold_results = []
        all_test_preds = []
        all_test_true = []
        
        print("\n" + "=" * 70)
        print("FOLD-BY-FOLD ANALYSIS")
        print("=" * 70)
        
        for fold_idx in range(len(self.folds)):
            fold_def = self.folds[fold_idx]
            print(f"\n{'─' * 70}")
            print(f"FOLD {fold_idx + 1}: Train [{fold_def[0]} → {fold_def[1]}]")
            print(f"         Test  [{fold_def[2]} → {fold_def[3]}]")
            print("─" * 70)
            
            df_train, df_test = self.get_fold_data(df_all, fold_idx)
            
            if len(df_train) == 0 or len(df_test) == 0:
                print(f"   ⚠️ Insufficient data for fold {fold_idx + 1}")
                print(f"      Train: {len(df_train)}, Test: {len(df_test)}")
                continue
            
            print(f"   Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")
            
            # Split some train for validation
            val_size = int(len(df_train) * 0.1)
            df_val = df_train.iloc[-val_size:]
            df_train_actual = df_train.iloc[:-val_size]
            
            # Train model
            print("   Training model...")
            model = self.train_fold_model(df_train_actual, df_val)
            
            # Predict on test
            X_test = df_test[self.feature_cols].values
            y_test = df_test["label"].values.astype(int)
            y_pred = self.predict_with_thresholds(model, X_test)
            
            # Collect for aggregate metrics
            all_test_preds.extend(y_pred)
            all_test_true.extend(y_test)
            
            # Calculate fold metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            print(f"\n   📊 Fold {fold_idx + 1} Results:")
            print(f"      Accuracy: {metrics['accuracy']*100:.1f}%")
            for class_name, class_metrics in metrics["per_class"].items():
                print(f"      {class_name}: P={class_metrics['precision']*100:.0f}% "
                      f"R={class_metrics['recall']*100:.0f}% "
                      f"F1={class_metrics['f1']*100:.0f}%")
            
            fold_results.append({
                "fold": fold_idx + 1,
                "train_period": f"{fold_def[0]} → {fold_def[1]}",
                "test_period": f"{fold_def[2]} → {fold_def[3]}",
                "train_size": len(df_train),
                "test_size": len(df_test),
                "metrics": metrics
            })
        
        # Aggregate results
        print("\n" + "=" * 70)
        print("AGGREGATE WFO RESULTS")
        print("=" * 70)
        
        if len(fold_results) == 0:
            print("❌ No valid folds completed!")
            return {"error": "No valid folds"}
        
        # Overall metrics
        all_test_preds = np.array(all_test_preds)
        all_test_true = np.array(all_test_true)
        aggregate_metrics = self.calculate_metrics(all_test_true, all_test_preds)
        
        # Per-fold statistics
        accuracies = [f["metrics"]["accuracy"] for f in fold_results]
        
        results = {
            "run_date": datetime.now().isoformat(),
            "num_folds": len(fold_results),
            "fold_results": fold_results,
            "aggregate_metrics": aggregate_metrics,
            "wfo_score": {
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "min_accuracy": float(np.min(accuracies)),
                "max_accuracy": float(np.max(accuracies)),
            },
            "thresholds_used": self.thresholds,
        }
        
        print(f"\n📈 WFO Score: {results['wfo_score']['mean_accuracy']*100:.1f}% "
              f"± {results['wfo_score']['std_accuracy']*100:.1f}%")
        print(f"   Range: [{results['wfo_score']['min_accuracy']*100:.1f}%, "
              f"{results['wfo_score']['max_accuracy']*100:.1f}%]")
        
        print(f"\n📊 Aggregate Metrics (all OOS predictions):")
        print(f"   Accuracy: {aggregate_metrics['accuracy']*100:.1f}%")
        for class_name, class_metrics in aggregate_metrics["per_class"].items():
            print(f"   {class_name}: P={class_metrics['precision']*100:.0f}% "
                  f"R={class_metrics['recall']*100:.0f}% "
                  f"F1={class_metrics['f1']*100:.0f}%")
        
        # Save results
        output_path = self.results_dir / "wfo_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Check success criteria
        self._check_success_criteria(results)
        
        return results
    
    def _check_success_criteria(self, results: Dict):
        """Check if WFO results meet Phase 2 criteria."""
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 70)
        
        criteria = {
            "WFO stability (std < 10%)": results["wfo_score"]["std_accuracy"] < 0.10,
            "Min fold accuracy > 50%": results["wfo_score"]["min_accuracy"] > 0.50,
            "Mean accuracy > 55%": results["wfo_score"]["mean_accuracy"] > 0.55,
        }
        
        # Check per-class recalls from aggregate
        agg = results["aggregate_metrics"]
        for class_name in self.class_names:
            recall = agg["per_class"][class_name]["recall"]
            criteria[f"{class_name} recall > 50%"] = recall > 0.50
        
        all_pass = True
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
            if not passed:
                all_pass = False
        
        print()
        if all_pass:
            print("🎉 ALL WFO CRITERIA PASSED!")
        else:
            print("⚠️ Some criteria not met - review results")


def main():
    project_root = Path(__file__).parent.parent.parent
    wfo = WalkForwardOptimizer(project_root)
    results = wfo.run_wfo()
    return results


if __name__ == "__main__":
    main()
