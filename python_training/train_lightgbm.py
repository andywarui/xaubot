"""
Train LightGBM on M1 features.
"""
import sys
import json
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_meta():
    config_path = Path(__file__).parent.parent / "config" / "model_meta.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("LightGBM Training on M1 Features")
    print("=" * 70)
    
    config = load_config()
    model_meta = load_model_meta()
    project_root = Path(__file__).parent.parent
    
    # Load data
    print("\nLoading M1 feature data...")
    train_df = pd.read_parquet(project_root / 'data' / 'processed' / 'features_m1_train.parquet')
    val_df = pd.read_parquet(project_root / 'data' / 'processed' / 'features_m1_val.parquet')
    
    print(f"  Train: {len(train_df):,} M1 bars")
    print(f"  Val:   {len(val_df):,} M1 bars")
    
    # Prepare features and labels
    feature_cols = [c for c in train_df.columns if c not in ['time', 'label']]
    print(f"  Features: {len(feature_cols)}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    # Class distribution
    print("\nClass distribution:")
    for cls in [0, 1, 2]:
        train_count = (y_train == cls).sum()
        val_count = (y_val == cls).sum()
        print(f"  Class {cls}: Train {train_count:,} ({train_count/len(y_train)*100:.1f}%), Val {val_count:,} ({val_count/len(y_val)*100:.1f}%)")
    
    # Train LightGBM
    print("\nTraining LightGBM...")
    params = model_meta['training_params']
    params['objective'] = 'multiclass'
    params['num_class'] = 3
    params['metric'] = 'multi_logloss'
    params['verbose'] = -1
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=params['n_estimators'],
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50)
        ]
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    
    # Predictions
    print("\nEvaluating...")
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['SHORT', 'HOLD', 'LONG'], digits=3))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print("           Predicted")
    print("           SHORT  HOLD  LONG")
    print(f"Actual SHORT  {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}")
    print(f"       HOLD   {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}")
    print(f"       LONG   {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}")
    
    # Feature importance
    print("\nTop 10 Features:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:8.0f}")
    
    # Save model
    model_dir = project_root / 'python_training' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'lightgbm_xauusd.pkl'
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")
    
    # Save metadata
    metadata = {
        'num_features': len(feature_cols),
        'num_classes': 3,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'accuracy': float(accuracy),
        'best_iteration': model.best_iteration
    }
    
    with open(model_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nTraining complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
