"""
Retrain LightGBM with SMC quality filtering
Only use SMC score 3-4 samples (high-probability setups)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score
import json

print("=" * 70)
print("LIGHTGBM RETRAIN - SMC FILTERED (SCORE 3-4)")
print("=" * 70)
print()

# Load labeled data
print("📥 Loading labeled data...")
df = pd.read_csv('data/processed/xauusd_labeled.csv')
df['time'] = pd.to_datetime(df['time'])

print(f"   Original rows: {len(df):,}")
print()

# Filter for high-quality SMC setups (score 3-4)
print("🔧 Filtering for SMC Quality Score 3-4...")
df_filtered = df[df['smc_quality_score'].isin([3, 4])].copy()

print(f"   After filtering: {len(df_filtered):,} rows")
print(f"   Retained: {len(df_filtered)/len(df)*100:.1f}%")
print()

# SMC score distribution in filtered data
print("📊 SMC Score Distribution (Filtered):")
smc_dist = df_filtered['smc_quality_score'].value_counts().sort_index()
for score, count in smc_dist.items():
    pct = count / len(df_filtered) * 100
    print(f"   Score {int(score)}: {count:,} ({pct:.1f}%)")
print()

# Feature columns
exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 
                'forward_return_15m', 'label']
feature_cols = [col for col in df_filtered.columns if col not in exclude_cols]

# Prepare data
X = df_filtered[feature_cols].values
y = (df_filtered['label'].values + 1)  # Map to 0,1,2

# Train/test split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"🔪 Train/Test Split:")
print(f"   Train: {len(X_train):,} samples")
print(f"   Test:  {len(X_test):,} samples")
print()

# Class distribution
print("📊 Class Distribution:")
for dataset_name, y_data in [('Train', y_train), ('Test', y_test)]:
    print(f"\n   {dataset_name}:")
    for cls in [0, 1, 2]:
        count = (y_data == cls).sum()
        pct = count / len(y_data) * 100
        label_name = ['SHORT', 'HOLD', 'LONG'][cls]
        print(f"      {label_name}: {count:,} ({pct:.1f}%)")
print()

# LightGBM parameters (slightly tuned for filtered data)
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,  # Increased (less features)
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'seed': 42
}

# Train
print("🚀 Training filtered model...")
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=20)
    ]
)

print()
print("✅ Training complete!")
print(f"   Best iteration: {model.best_iteration}")
print()

# Predictions
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred_proba, axis=1)
max_proba = np.max(y_pred_proba, axis=1)

# Overall performance
accuracy = accuracy_score(y_test, y_pred)

print("=" * 70)
print("📈 SMC-FILTERED MODEL PERFORMANCE")
print("=" * 70)
print()
print(f"✅ Overall Accuracy: {accuracy:.2%}")
print()

# Classification report
target_names = ['SHORT', 'HOLD', 'LONG']
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

# Confidence analysis
print("\n📊 High-Confidence Performance:")
for threshold in [0.50, 0.55, 0.60]:
    conf_mask = max_proba >= threshold
    if conf_mask.sum() > 0:
        y_test_conf = y_test[conf_mask]
        y_pred_conf = y_pred[conf_mask]
        conf_acc = accuracy_score(y_test_conf, y_pred_conf)
        
        # LONG accuracy
        long_mask = (y_test_conf == 2)
        if long_mask.sum() > 0:
            long_acc = ((y_pred_conf == 2) & (y_test_conf == 2)).sum() / long_mask.sum()
        else:
            long_acc = 0
        
        print(f"   Confidence ≥{threshold:.0%}: {conf_mask.sum():,} samples, "
              f"Accuracy: {conf_acc:.2%}, LONG: {long_acc:.2%}")
print()

# Save model
model_path = 'python_training/models/lightgbm_xauusd_smc_filtered.txt'
print(f"💾 Saving SMC-filtered model to: {model_path}")
model.save_model(model_path)

# Save metadata
metadata = {
    'model_type': 'LightGBM-SMC-Filtered',
    'smc_filter': 'score >= 3',
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': float(accuracy),
    'best_iteration': model.best_iteration
}

with open('python_training/models/model_metadata_smc.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print()
print("=" * 70)
print("✅ SMC-FILTERED MODEL COMPLETE!")
print("=" * 70)
print()
print("📊 Comparison:")
print(f"   Baseline (all data): 47.50%")
print(f"   SMC-filtered (3-4):  {accuracy:.2%}")
print(f"   Improvement: +{(accuracy - 0.475)*100:.1f} percentage points")
print()
print("🎯 Next Step:")
print("   Export best model to ONNX: python src/export_to_onnx.py")