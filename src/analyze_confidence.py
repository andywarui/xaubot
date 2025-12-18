"""
Analyze model predictions by confidence level
Find optimal confidence threshold for trading
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import json

print("=" * 70)
print("CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 70)
print()

# Load data and model
print("📥 Loading data and model...")
df = pd.read_csv('data/processed/xauusd_labeled.csv')
df['time'] = pd.to_datetime(df['time'])

with open('python_training/models/feature_list.json', 'r') as f:
    feature_cols = json.load(f)

model = lgb.Booster(model_file='python_training/models/lightgbm_xauusd_v1.txt')

# Split data
split_idx = int(0.8 * len(df))
X_test = df[feature_cols].iloc[split_idx:].values
y_test = (df['label'].iloc[split_idx:] + 1).values  # Map to 0,1,2

print(f"   Test samples: {len(X_test):,}")
print()

# Get predictions and confidence
print("🔮 Computing predictions and confidence...")
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
max_proba = np.max(y_pred_proba, axis=1)

print(f"   ✓ Predictions computed")
print()

# Analyze by confidence threshold
print("=" * 70)
print("📊 PERFORMANCE BY CONFIDENCE THRESHOLD")
print("=" * 70)
print()

thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

results = []

for threshold in thresholds:
    # Filter high-confidence predictions
    conf_mask = max_proba >= threshold
    
    if conf_mask.sum() == 0:
        continue
    
    y_test_conf = y_test[conf_mask]
    y_pred_conf = y_pred[conf_mask]
    
    # Calculate metrics
    n_samples = len(y_test_conf)
    pct_total = n_samples / len(y_test) * 100
    accuracy = accuracy_score(y_test_conf, y_pred_conf)
    
    # Per-class accuracy
    short_acc = ((y_pred_conf == 0) & (y_test_conf == 0)).sum() / max((y_test_conf == 0).sum(), 1)
    hold_acc = ((y_pred_conf == 1) & (y_test_conf == 1)).sum() / max((y_test_conf == 1).sum(), 1)
    long_acc = ((y_pred_conf == 2) & (y_test_conf == 2)).sum() / max((y_test_conf == 2).sum(), 1)
    
    results.append({
        'threshold': threshold,
        'n_samples': n_samples,
        'pct_total': pct_total,
        'accuracy': accuracy,
        'short_acc': short_acc,
        'hold_acc': hold_acc,
        'long_acc': long_acc
    })
    
    print(f"Confidence ≥ {threshold:.0%}:")
    print(f"   Samples: {n_samples:,} ({pct_total:.1f}% of test set)")
    print(f"   Overall Accuracy: {accuracy:.2%}")
    print(f"   SHORT: {short_acc:.2%} | HOLD: {hold_acc:.2%} | LONG: {long_acc:.2%}")
    print()

# Find optimal threshold (target: 60%+ accuracy with reasonable sample size)
print("=" * 70)
print("🎯 RECOMMENDED TRADING THRESHOLD")
print("=" * 70)
print()

# Look for threshold with >60% accuracy and >1% sample retention
optimal = None
for r in results:
    if r['accuracy'] >= 0.60 and r['pct_total'] >= 1.0:
        optimal = r
        break

if optimal:
    print(f"✅ Optimal Threshold: {optimal['threshold']:.0%}")
    print(f"   Expected Accuracy: {optimal['accuracy']:.2%}")
    print(f"   Trade Frequency: {optimal['pct_total']:.1f}% of signals")
    print(f"   Estimated Trades/Day: {optimal['n_samples'] / (len(y_test) / 240) * 0.25:.1f}")
    print()
    print(f"📊 Per-Class Performance:")
    print(f"   SHORT: {optimal['short_acc']:.2%}")
    print(f"   HOLD:  {optimal['hold_acc']:.2%}")
    print(f"   LONG:  {optimal['long_acc']:.2%}")
else:
    print("⚠️ No threshold meets 60% accuracy + 1% retention")
    print("   Using threshold with highest accuracy:")
    best = max(results, key=lambda x: x['accuracy'])
    print(f"   Threshold: {best['threshold']:.0%}")
    print(f"   Accuracy: {best['accuracy']:.2%}")

print()

# Detailed analysis of optimal threshold
if optimal:
    threshold = optimal['threshold']
    conf_mask = max_proba >= threshold
    
    print("=" * 70)
    print(f"DETAILED ANALYSIS AT {threshold:.0%} CONFIDENCE")
    print("=" * 70)
    print()
    
    y_test_conf = y_test[conf_mask]
    y_pred_conf = y_pred[conf_mask]
    
    # Classification report
    target_names = ['SHORT', 'HOLD', 'LONG']
    print("📊 Classification Report:")
    print(classification_report(y_test_conf, y_pred_conf, target_names=target_names, digits=3))
    
    # Signal distribution
    print("📊 Signal Distribution:")
    for cls, name in enumerate(target_names):
        count = (y_pred_conf == cls).sum()
        pct = count / len(y_pred_conf) * 100
        print(f"   {name}: {count:,} ({pct:.1f}%)")

print()
print("=" * 70)
print("✅ CONFIDENCE ANALYSIS COMPLETE")
print("=" * 70)
print()
print("🎯 Key Takeaway:")
print(f"   Use {optimal['threshold']:.0%} confidence threshold for live trading")
print(f"   Expected win rate: {optimal['accuracy']:.1%}")
print()
print("📝 Next Step:")
print("   Run: python src/retrain_filtered.py")