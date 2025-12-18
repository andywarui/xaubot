"""
Create training labels for XAUUSD trading bot
Labels: -1 (short), 0 (hold), +1 (long)
Based on 15-minute forward returns
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("CREATE TRAINING LABELS")
print("=" * 70)
print()

# Load features
print("📥 Loading feature data...")
df = pd.read_csv('data/processed/xauusd_features.csv')
df['time'] = pd.to_datetime(df['time'])

print(f"   Rows: {len(df):,}")
print(f"   Features: {len([c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']])}")
print()

# Compute forward returns
print("🔧 Computing 15-minute forward returns...")
df['forward_return_15m'] = df['close'].pct_change(15).shift(-15)

print(f"   ✓ Forward return computed")
print()

# Create labels based on thresholds
print("🏷️ Creating labels...")

# Define thresholds (0.1% = 10 pips on XAUUSD ~$2000)
# Adjust based on ATR
threshold_buy = 0.0005   # 0.05% (buy if return > this)
threshold_sell = -0.0005  # -0.05% (sell if return < this)

df['label'] = np.where(
    df['forward_return_15m'] > threshold_buy, 
    1,  # Long
    np.where(
        df['forward_return_15m'] < threshold_sell,
        -1,  # Short
        0   # Hold
    )
)

# Remove rows with NaN labels (last 15 bars)
df = df.dropna(subset=['forward_return_15m', 'label'])

print(f"   Threshold (buy): {threshold_buy*100:.2f}%")
print(f"   Threshold (sell): {threshold_sell*100:.2f}%")
print()

# Label distribution
print("📊 Label Distribution:")
label_counts = df['label'].value_counts()
for label in [-1, 0, 1]:
    count = label_counts.get(label, 0)
    pct = count / len(df) * 100
    label_name = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}[label]
    print(f"   {label_name:5s} ({label:2d}): {count:,} samples ({pct:.1f}%)")

print()

# Check label balance
short_pct = (df['label'] == -1).sum() / len(df) * 100
long_pct = (df['label'] == 1).sum() / len(df) * 100
hold_pct = (df['label'] == 0).sum() / len(df) * 100

print("⚖️ Label Balance:")
if 30 <= short_pct <= 40 and 30 <= long_pct <= 40:
    print("   ✅ Well-balanced (30-40% short, 30-40% long)")
elif hold_pct > 50:
    print("   ⚠️ Too many HOLD labels - consider adjusting thresholds")
else:
    print("   ✓ Acceptable balance")
print()

# Save labeled data
output_file = 'data/processed/xauusd_labeled.csv'
print(f"💾 Saving labeled data to: {output_file}")

df.to_csv(output_file, index=False)

print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")
print()

# Show sample
print("📋 Sample labeled data:")
sample_cols = ['time', 'close', 'smc_quality_score', 'forward_return_15m', 'label']
print(df[sample_cols].head(10).to_string(index=False))

print()
print("=" * 70)
print("✅ LABELS CREATED!")
print("=" * 70)
print()
print("📊 Dataset Summary:")
print(f"   Total samples: {len(df):,}")
print(f"   Features: {len([c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume', 'forward_return_15m', 'label']])}")
print(f"   Labels: 3 classes (SHORT, HOLD, LONG)")
print(f"   File: {output_file}")
print()
print("🎯 Next Step:")
print("   Train baseline model: python train_lightgbm.py")
