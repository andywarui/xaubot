"""
Train LightGBM classifier for XAUUSD trading
3-class classification: SHORT (-1), HOLD (0), LONG (1)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
from datetime import datetime

print("=" * 70)
print("LIGHTGBM MODEL TRAINING - XAUUSD AI BOT")
print("=" * 70)
print()

# Load labeled data
print("📥 Loading labeled data...")
df = pd.read_csv('data/processed/xauusd_labeled.csv')
df['time'] = pd.to_datetime(df['time'])

print(f"   Rows: {len(df):,}")
print(f"   Date range: {df['time'].min().date()} → {df['time'].max().date()}")
print()

# Define feature columns (exclude target and metadata)
exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 
                'forward_return_15m', 'label']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"📊 Features: {len(feature_cols)}")
print(f"   Target: label (-1: SHORT, 0: HOLD, 1: LONG)")
print()

# Prepare data
X = df[feature_cols].values
y = df['label'].values

# Map labels: -1→0, 0→1, 1→2 (LightGBM requires 0-indexed labels)
y_mapped = y + 1  # Now: 0=SHORT, 1=HOLD, 2=LONG

print("🔪 Train/Test Split (80/20 temporal)...")
split_idx = int(0.8 * len(X))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_mapped[:split_idx], y_mapped[split_idx:]

train_dates = df['time'].iloc[:split_idx]
test_dates = df['time'].iloc[split_idx:]

print(f"   Train: {len(X_train):,} samples ({train_dates.min().date()} → {train_dates.max().date()})")
print(f"   Test:  {len(X_test):,} samples ({test_dates.min().date()} → {test_dates.max().date()})")
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

# LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

print("🔧 LightGBM Parameters:")
for key, val in params.items():
    print(f"   {key}: {val}")
print()

# Train model
print("🚀 Training LightGBM model...")
print("   (This may take 2-5 minutes...)")
print()

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
print(f"   Best score: {model.best_score['valid']['multi_logloss']:.4f}")
print()

# Predictions
print("🔮 Making predictions...")
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred_proba, axis=1)

# Evaluate
print("\n" + "=" * 70)
print("📈 MODEL PERFORMANCE")
print("=" * 70)
print()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Overall Accuracy: {accuracy:.2%}")
print()

# Classification report
print("📊 Classification Report:")
target_names = ['SHORT (0)', 'HOLD (1)', 'LONG (2)']
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

# Confusion matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("        Predicted")
print("           SHORT  HOLD  LONG")
print(f"Actual SHORT  {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}")
print(f"       HOLD   {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}")
print(f"       LONG   {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}")
print()

# Per-class accuracy
print("📊 Per-Class Accuracy:")
for i, name in enumerate(['SHORT', 'HOLD', 'LONG']):
    class_acc = cm[i][i] / cm[i].sum()
    print(f"   {name}: {class_acc:.2%}")
print()

# Feature importance
print("🔝 Top 20 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(20).iterrows():
    print(f"   {row['feature']:25s}: {row['importance']:8.0f}")
print()

# Trading signals analysis
print("📊 Trading Signals on Test Set:")
trading_signals = {
    'SHORT': (y_pred == 0).sum(),
    'HOLD': (y_pred == 1).sum(),
    'LONG': (y_pred == 2).sum()
}

total_signals = sum(trading_signals.values())
for signal, count in trading_signals.items():
    pct = count / total_signals * 100
    print(f"   {signal:5s}: {count:,} ({pct:.1f}%)")
print()

# Confidence analysis
print("📊 Prediction Confidence Analysis:")
max_proba = np.max(y_pred_proba, axis=1)
for threshold in [0.50, 0.60, 0.70, 0.80]:
    high_conf = (max_proba >= threshold).sum()
    pct = high_conf / len(max_proba) * 100
    print(f"   Confidence >= {threshold:.0%}: {high_conf:,} predictions ({pct:.1f}%)")
print()

# Save model
model_dir = 'python_training/models'
import os
os.makedirs(model_dir, exist_ok=True)

model_path = f'{model_dir}/lightgbm_xauusd_v1.txt'
print(f"💾 Saving model to: {model_path}")
model.save_model(model_path)

# Save feature list
feature_list_path = f'{model_dir}/feature_list.json'
with open(feature_list_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"   Feature list: {feature_list_path}")

# Save metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'model_type': 'LightGBM',
    'num_features': len(feature_cols),
    'num_classes': 3,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': float(accuracy),
    'best_iteration': model.best_iteration,
    'params': params
}

metadata_path = f'{model_dir}/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   Metadata: {metadata_path}")

print()
print("=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
print()
print("📊 Summary:")
print(f"   Model: LightGBM (200 rounds, early stopping)")
print(f"   Accuracy: {accuracy:.2%}")
print(f"   Classes: 3 (SHORT, HOLD, LONG)")
print(f"   Files saved: {model_dir}/")
print()
print("🎯 Next Steps:")
print("   1. Review classification report above")
print("   2. Check feature importance")
print("   3. Proceed to ONNX export: python export_to_onnx.py")