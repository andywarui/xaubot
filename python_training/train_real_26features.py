"""
Train LightGBM model on REAL XAUUSD data from Kaggle
Uses 26 features (no Transformer dependency)

Data: Kaggle XAUUSD 2022-2024 (1.06M bars)
Source: https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

print("="*70)
print("TRAINING LIGHTGBM ON REAL XAUUSD DATA (26 FEATURES)")
print("="*70)
print()

# Paths
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'raw' / 'XAU_1m_data.csv'
model_dir = project_root / 'python_training' / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Data source: {data_path}")
print(f"Output directory: {model_dir}")
print()

# Load and prepare data
print("Step 1: Loading real XAUUSD data...")
df = pd.read_csv(data_path, sep=';', parse_dates=['Date'])
df = df.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
print(f"✓ Loaded {len(df):,} rows")
print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
print()

# Filter for 2022-2024 (matching backtest period)
print("Step 2: Filtering for 2022-2024...")
df = df[(df['time'] >= '2022-01-01') & (df['time'] <= '2024-12-31')].copy()
df = df.sort_values('time').reset_index(drop=True)
print(f"✓ Filtered to {len(df):,} rows")
print()

# Calculate technical indicators
print("Step 3: Calculating technical indicators...")

# True Range & ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
)
df['atr_14'] = df['tr'].rolling(window=14, min_periods=1).mean()

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
rs = gain / (loss + 1e-10)
df['rsi_14'] = 100 - (100 / (1 + rs))

# EMAs
df['ema_10'] = df['close'].ewm(span=10, adjust=False, min_periods=1).mean()
df['ema_20'] = df['close'].ewm(span=20, adjust=False, min_periods=1).mean()
df['ema_50'] = df['close'].ewm(span=50, adjust=False, min_periods=1).mean()

print("✓ Technical indicators calculated")
print()

# Calculate 26 features
print("Step 4: Building feature matrix (26 features)...")

def calculate_features(df, idx):
    """Calculate 26 features for each bar"""
    if idx < 60:
        return None

    row = df.iloc[idx]
    features = {}

    # Price features (4)
    body = row['close'] - row['open']
    features['body'] = body
    features['body_abs'] = abs(body)
    features['candle_range'] = row['high'] - row['low']
    features['close_position'] = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-8)

    # Returns (4)
    features['return_1'] = (df.iloc[idx]['close'] / df.iloc[idx-1]['close']) - 1.0
    features['return_5'] = (df.iloc[idx]['close'] / df.iloc[idx-5]['close']) - 1.0
    features['return_15'] = (df.iloc[idx]['close'] / df.iloc[idx-15]['close']) - 1.0
    features['return_60'] = (df.iloc[idx]['close'] / df.iloc[idx-60]['close']) - 1.0

    # Technical indicators (6)
    features['tr'] = row.get('tr', 0.0)
    features['atr_14'] = row.get('atr_14', 0.0)
    features['rsi_14'] = row.get('rsi_14', 50.0)
    features['ema_10'] = row.get('ema_10', row['close'])
    features['ema_20'] = row.get('ema_20', row['close'])
    features['ema_50'] = row.get('ema_50', row['close'])

    # Time features (2)
    hour = row['time'].hour
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Multi-timeframe placeholders (10) - set to 0 for now
    for i in range(10):
        features[f'mtf_{i}'] = 0.0

    return features

# Build feature matrix
feature_list = []
labels = []
valid_indices = []

for idx in range(100, len(df) - 30):  # Leave room for lookahead
    features = calculate_features(df, idx)
    if features is None:
        continue

    # Create target: Will this trade be profitable?
    # Look ahead 30 bars and check if TP hit before SL
    entry_price = df.iloc[idx]['close']
    tp_usd = 8.0  # Take profit
    sl_usd = 4.0  # Stop loss

    label = 1  # HOLD by default
    for future_idx in range(idx + 1, min(idx + 31, len(df))):
        future_high = df.iloc[future_idx]['high']
        future_low = df.iloc[future_idx]['low']

        # Check LONG
        if future_high >= entry_price + tp_usd:
            label = 2  # LONG (profitable)
            break
        if future_low <= entry_price - sl_usd:
            label = 1  # HOLD (would lose)
            break

        # Check SHORT
        if future_low <= entry_price - tp_usd:
            label = 0  # SHORT (profitable)
            break
        if future_high >= entry_price + sl_usd:
            label = 1  # HOLD (would lose)
            break

    feature_list.append(list(features.values()))
    labels.append(label)
    valid_indices.append(idx)

    if len(feature_list) % 100000 == 0:
        print(f"  Processed {len(feature_list):,} samples...")

X = np.array(feature_list, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"✓ Feature matrix created: {X.shape}")
print(f"  Features: {len(features)} per sample")
print(f"  Total samples: {len(X):,}")
print()

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print("Target distribution:")
for cls, count in zip(unique, counts):
    cls_name = ['SHORT', 'HOLD', 'LONG'][cls]
    print(f"  {cls_name}: {count:,} ({count/len(y)*100:.1f}%)")
print()

# Train/test split (80/20, chronological)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train):,} samples")
print(f"Test:  {len(X_test):,} samples")
print()

# Train LightGBM
print("Step 5: Training LightGBM model...")
print()

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'max_depth': 6,
    'min_data_in_leaf': 100
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
)

print()
print(f"✓ Training complete")
print(f"  Best iteration: {model.best_iteration}")
print()

# Evaluate
print("Step 6: Evaluating model...")
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_class = np.argmax(y_pred, axis=1)

accuracy = (y_pred_class == y_test).mean()
print(f"Test Accuracy: {accuracy*100:.2f}%")
print()

print("Per-class accuracy:")
for cls in range(3):
    cls_name = ['SHORT', 'HOLD', 'LONG'][cls]
    mask = y_test == cls
    if mask.sum() > 0:
        cls_acc = (y_pred_class[mask] == cls).mean()
        print(f"  {cls_name}: {cls_acc*100:.1f}%")
print()

# Save model
print("Step 7: Saving model...")
model_path = model_dir / 'lightgbm_real_26features.txt'
model.save_model(str(model_path))
print(f"✓ Saved to {model_path.name}")
print()

# Save config
config = {
    'num_features': 26,
    'num_classes': 3,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': float(accuracy),
    'best_iteration': model.best_iteration,
    'data_source': 'Kaggle XAUUSD 2022-2024',
    'data_url': 'https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024',
    'target': 'Profitable trades (TP=$8, SL=$4, lookahead=30 bars)',
    'class_names': ['SHORT', 'HOLD', 'LONG']
}

config_path = model_dir / 'lightgbm_real_26features_config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✓ Config saved to {config_path.name}")
print()

print("="*70)
print("TRAINING COMPLETE")
print("="*70)
print()
print(f"Model: {model_path.name}")
print(f"Features: 26")
print(f"Accuracy: {accuracy*100:.1f}%")
print(f"Training samples: {len(X_train):,}")
print()
print("Next steps:")
print("1. Convert to ONNX: python python_training/export_real_model_onnx.py")
print("2. Run backtest: python python_backtesting/run_backtest.py")
print()
