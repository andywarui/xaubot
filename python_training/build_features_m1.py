"""
M1-based feature engineering with higher TF context.
Each row = one M1 bar with M5/M15/H1/H4/D1 context.
"""
import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_meta():
    config_path = Path(__file__).parent.parent / "config" / "model_meta.json"
    with open(config_path, 'r') as f:
        return json.load(f)


# M1 Features (simplified for speed)
def add_m1_features(df):
    # Price
    df['body'] = df['close'] - df['open']
    df['body_abs'] = abs(df['body'])
    df['candle_range'] = df['high'] - df['low']
    df['close_position'] = (df['close'] - df['low']) / (df['candle_range'] + 1e-10)
    
    # Returns
    for p in [1, 5, 15, 60]:  # 1m, 5m, 15m, 1h in M1 bars
        df[f'return_{p}'] = df['close'].pct_change(p)
    
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = df['tr'].rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # EMAs
    for period in [10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Session
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


# Higher TF merge
def merge_higher_tf_features(df_m1, project_root):
    df = df_m1.copy()
    
    for tf_name in ['M5', 'M15', 'H1', 'H4', 'D1']:
        tf_path = project_root / 'data' / 'processed' / f'xauusd_{tf_name}.parquet'
        if not tf_path.exists():
            continue
        
        tf_df = pd.read_parquet(tf_path).drop(columns=['tf'], errors='ignore')
        
        # Compute higher TF features
        tf_df[f'{tf_name}_ema20'] = tf_df['close'].ewm(span=20, adjust=False).mean()
        tf_df[f'{tf_name}_trend'] = np.sign(tf_df['close'] - tf_df[f'{tf_name}_ema20'])
        tf_df[f'{tf_name}_range'] = tf_df['high'] - tf_df['low']
        tf_df[f'{tf_name}_position'] = (tf_df['close'] - tf_df['low']) / (tf_df[f'{tf_name}_range'] + 1e-10)
        
        # Nearest-past join
        merge_cols = ['time', f'{tf_name}_trend', f'{tf_name}_position']
        df = pd.merge_asof(df.sort_values('time'), tf_df[merge_cols].sort_values('time'),
                          on='time', direction='backward')
    
    return df


# Labels - per M1 bar
def create_labels(df, tp_pips=80, sl_pips=40, forward_bars=12):
    tp_move = tp_pips * 0.01
    sl_move = sl_pips * 0.01
    labels = []
    
    for i in range(len(df) - forward_bars):
        current_close = df['close'].iloc[i]
        future_highs = df['high'].iloc[i+1:i+forward_bars+1].values
        future_lows = df['low'].iloc[i+1:i+forward_bars+1].values
        
        long_tp_idx = np.where(future_highs >= current_close + tp_move)[0]
        long_sl_idx = np.where(future_lows <= current_close - sl_move)[0]
        
        long_tp_first = len(long_tp_idx) > 0 and (len(long_sl_idx) == 0 or long_tp_idx[0] < long_sl_idx[0])
        long_sl_first = len(long_sl_idx) > 0 and (len(long_tp_idx) == 0 or long_sl_idx[0] < long_tp_idx[0])
        
        if long_tp_first:
            labels.append(2)
        elif long_sl_first:
            labels.append(0)
        else:
            labels.append(1)
    
    labels.extend([np.nan] * forward_bars)
    df['label'] = labels
    return df


def main():
    print("=" * 70)
    print("M1-Based Feature Engineering with Higher TF Context")
    print("=" * 70)
    
    config = load_config()
    model_meta = load_model_meta()
    project_root = Path(__file__).parent.parent
    
    # Load M1 parquet
    m1_parquet = project_root / 'data' / 'processed' / 'xauusd_M1.parquet'
    if not m1_parquet.exists():
        print(f"ERROR: {m1_parquet} not found")
        sys.exit(1)
    
    print("\nLoading M1 data...")
    df = pd.read_parquet(m1_parquet)
    df = df.drop(columns=['tf'], errors='ignore')
    
    # Add time features
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['dayofweek'] = df['time'].dt.dayofweek
    
    print(f"  Loaded {len(df):,} M1 bars")
    
    # Compute M1 features
    print("\nComputing M1 features...")
    df = add_m1_features(df)
    
    # Merge higher TF features
    print("\nMerging higher timeframe context...")
    df = merge_higher_tf_features(df, project_root)
    
    # Create labels (per M1 bar)
    print("\nCreating labels (per M1 bar)...")
    label_params = model_meta['label_params']
    df = create_labels(df, label_params['tp_pips'], label_params['sl_pips'], label_params['forward_bars'])
    
    # Remove NaN
    print("\nCleaning NaN...")
    initial = len(df)
    df = df.dropna()
    print(f"  Removed {initial - len(df):,} rows, remaining: {len(df):,}")
    
    # Get feature columns
    exclude_cols = ['time', 'label', 'open', 'high', 'low', 'close', 'volume', 
                   'year', 'month', 'day', 'hour', 'minute', 'dayofweek']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"  Total features: {len(feature_cols)}")
    
    # Split by date
    print("\nSplitting data...")
    split_config = config['train_test_split']
    n = len(df)
    train_end = int(n * split_config['train_ratio'])
    val_end = int(n * (split_config['train_ratio'] + split_config['val_ratio']))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"  Train: {len(train_df):,} M1 bars")
    print(f"  Val:   {len(val_df):,} M1 bars")
    print(f"  Test:  {len(test_df):,} M1 bars")
    
    # Save
    output_dir = project_root / 'data' / 'processed'
    save_cols = ['time'] + feature_cols + ['label']
    
    train_df[save_cols].to_parquet(output_dir / 'features_m1_train.parquet', index=False)
    val_df[save_cols].to_parquet(output_dir / 'features_m1_val.parquet', index=False)
    test_df[save_cols].to_parquet(output_dir / 'features_m1_test.parquet', index=False)
    
    # Save feature order
    features_order_path = project_root / 'config' / 'features_order.json'
    with open(features_order_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    print(f"\nSaved features_m1_*.parquet and features_order.json")
    print("\nFeature engineering complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
