"""
Train LightGBM Model on Synthetic XAUUSD Data

This script trains a LightGBM classifier on the synthetic data we generated,
then exports it to ONNX format for backtesting.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LIGHTGBM TRAINING ON SYNTHETIC DATA")
print("="*70)
print()

# Load synthetic data
print("Step 1: Loading synthetic data...")
data_path = Path(__file__).parent.parent / 'python_backtesting' / 'xauusd_m1_backtest.parquet'

if not data_path.exists():
    print(f"❌ ERROR: Data file not found at {data_path}")
    print("   Please run: python python_backtesting/prepare_data.py")
    exit(1)

df = pd.read_parquet(data_path)
print(f"✓ Loaded {len(df):,} bars")
print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
print()

# Calculate features (same 26 features as backtest engine)
print("Step 2: Calculating 26 features...")

def calculate_features(data, idx):
    """Calculate 26 features for a given bar"""
    if idx < 60:
        return None

    row = data.iloc[idx]
    features = {}

    # Price features
    body = row['close'] - row['open']
    features['body'] = body
    features['body_abs'] = abs(body)
    features['candle_range'] = row['high'] - row['low']
    features['close_position'] = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-8)

    # Returns
    features['return_1'] = (data.iloc[idx]['close'] / data.iloc[idx-1]['close']) - 1.0
    features['return_5'] = (data.iloc[idx]['close'] / data.iloc[idx-5]['close']) - 1.0
    features['return_15'] = (data.iloc[idx]['close'] / data.iloc[idx-15]['close']) - 1.0
    features['return_60'] = (data.iloc[idx]['close'] / data.iloc[idx-60]['close']) - 1.0

    # Technical indicators
    features['tr'] = row.get('tr', 0.0)
    features['atr_14'] = row.get('atr_14', 0.0)
    features['rsi_14'] = row.get('rsi_14', 50.0)
    features['ema_10'] = row.get('ema_10', row['close'])
    features['ema_20'] = row.get('ema_20', row['close'])
    features['ema_50'] = row.get('ema_50', row['close'])

    # Time features
    hour = row['time'].hour if 'time' in row.index else 0
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Multi-timeframe (simplified - using same timeframe for now)
    features['M5_trend'] = 0.0
    features['M5_position'] = 0.5
    features['M15_trend'] = 0.0
    features['M15_position'] = 0.5
    features['H1_trend'] = 0.0
    features['H1_position'] = 0.5
    features['H4_trend'] = 0.0
    features['H4_position'] = 0.5
    features['D1_trend'] = 0.0
    features['D1_position'] = 0.5

    return features

def create_target(data, idx, lookahead=15):
    """
    Create target label based on future price movement

    Labels:
    0 = SHORT (price will go down)
    1 = HOLD (no significant movement)
    2 = LONG (price will go up)
    """
    if idx + lookahead >= len(data):
        return None

    current_price = data.iloc[idx]['close']
    future_price = data.iloc[idx + lookahead]['close']

    # Calculate future return
    future_return = (future_price - current_price) / current_price

    # Thresholds for XAUUSD (in percentage)
    long_threshold = 0.0015   # 0.15% up (~$3 on $2000)
    short_threshold = -0.0015  # 0.15% down

    if future_return > long_threshold:
        return 2  # LONG
    elif future_return < short_threshold:
        return 0  # SHORT
    else:
        return 1  # HOLD

# Sample data (use every 5th bar to speed up training)
sample_interval = 5
indices = range(100, len(df) - 20, sample_interval)

print(f"  Using every {sample_interval}th bar for training")
print(f"  Total training samples: {len(indices):,}")
print()

features_list = []
targets_list = []

print("Extracting features and targets...")
for i, idx in enumerate(indices):
    features = calculate_features(df, idx)
    target = create_target(df, idx, lookahead=15)

    if features is not None and target is not None:
        features_list.append(features)
        targets_list.append(target)

    if (i + 1) % 50000 == 0:
        print(f"  Processed {i+1:,} / {len(indices):,} bars...")

print(f"✓ Extracted {len(features_list):,} samples")
print()

# Convert to DataFrame
X = pd.DataFrame(features_list)
y = pd.Series(targets_list)

# Check class distribution
print("Target distribution:")
print(y.value_counts().sort_index())
print(f"  0 (SHORT): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
print(f"  1 (HOLD):  {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
print(f"  2 (LONG):  {(y == 2).sum():,} ({(y == 2).mean()*100:.1f}%)")
print()

# Split data
print("Step 3: Splitting data (70% train, 15% val, 15% test)...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)  # 0.176 * 0.85 ≈ 0.15

print(f"  Train: {len(X_train):,} samples")
print(f"  Val:   {len(X_val):,} samples")
print(f"  Test:  {len(X_test):,} samples")
print()

# Train LightGBM
print("Step 4: Training LightGBM classifier...")
print()

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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
    'verbose': 1,
    'max_depth': 6,
    'min_data_in_leaf': 100,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1
}

print("Training parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")
print()

model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=20)
    ]
)

print()
print("✓ Training complete!")
print(f"  Best iteration: {model.best_iteration}")
print(f"  Best score: {model.best_score['valid']['multi_logloss']:.4f}")
print()

# Evaluate
print("Step 5: Evaluating model...")
y_pred_train = model.predict(X_train).argmax(axis=1)
y_pred_val = model.predict(X_val).argmax(axis=1)
y_pred_test = model.predict(X_test).argmax(axis=1)

train_acc = (y_pred_train == y_train).mean()
val_acc = (y_pred_val == y_val).mean()
test_acc = (y_pred_test == y_test).mean()

print(f"  Train accuracy: {train_acc*100:.2f}%")
print(f"  Val accuracy:   {val_acc*100:.2f}%")
print(f"  Test accuracy:  {test_acc*100:.2f}%")
print()

print("Test set classification report:")
print(classification_report(y_test, y_pred_test, target_names=['SHORT', 'HOLD', 'LONG']))

print("Confusion Matrix (Test):")
cm = confusion_matrix(y_test, y_pred_test)
print("             Predicted")
print("              S    H    L")
print(f"Actual SHORT  {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}")
print(f"       HOLD   {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}")
print(f"       LONG   {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}")
print()

# Save model
print("Step 6: Saving LightGBM model...")
model_dir = Path(__file__).parent / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

# Save native LightGBM model
lgb_model_path = model_dir / 'lightgbm_synthetic.txt'
model.save_model(str(lgb_model_path))
print(f"✓ Saved native model: {lgb_model_path.name}")

# Export to ONNX
print()
print("Step 7: Exporting to ONNX...")

try:
    import onnxmltools
    from onnxmltools.convert import convert_lightgbm
    from onnxconverter_common.data_types import FloatTensorType

    # Define input type
    initial_types = [('input', FloatTensorType([None, 26]))]

    # Convert to ONNX
    onnx_model = convert_lightgbm(
        model,
        initial_types=initial_types,
        target_opset=14
    )

    # Save ONNX model
    onnx_path = model_dir / 'lightgbm_synthetic.onnx'
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print(f"✓ Saved ONNX model: {onnx_path.name}")
    print(f"  File size: {onnx_path.stat().st_size / 1024:.2f} KB")

except ImportError:
    print("⚠️  onnxmltools not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'onnxmltools', '--quiet'])
    print("✓ Installed onnxmltools. Please re-run this script.")
    exit(0)

print()
print("="*70)
print("TRAINING COMPLETE!")
print("="*70)
print()
print("Model files created:")
print(f"  - {lgb_model_path}")
print(f"  - {onnx_path}")
print()
print("Next step: Run backtest")
print("  python python_backtesting/run_backtest.py")
print()
