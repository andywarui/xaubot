"""
Build 26 features for each timeframe (M1, M5, M15, H1, D1).
Creates features_{tf}_train/val/test.parquet for multi-TF Transformer.

NOTE: M1 features already exist! This script adds 'close' column and
      builds features for M5/M15/H1/D1.

Features (26 per TF):
  body, body_abs, candle_range, close_position,
  return_1, return_5, return_15, return_60,
  tr, atr_14, rsi_14,
  ema_10, ema_20, ema_50,
  hour_sin, hour_cos,
  M5_trend, M5_position,
  M15_trend, M15_position,
  H1_trend, H1_position,
  H4_trend, H4_position,
  D1_trend, D1_position

Usage:
    python build_features_all_tf.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


TIMEFRAMES = ['M1', 'M5', 'M15', 'H1', 'D1']
FEATURE_COLS_26 = [
    'body', 'body_abs', 'candle_range', 'close_position',
    'return_1', 'return_5', 'return_15', 'return_60',
    'tr', 'atr_14', 'rsi_14',
    'ema_10', 'ema_20', 'ema_50',
    'hour_sin', 'hour_cos',
    'M5_trend', 'M5_position',
    'M15_trend', 'M15_position',
    'H1_trend', 'H1_position',
    'H4_trend', 'H4_position',
    'D1_trend', 'D1_position'
]


def add_base_features(df):
    """Add 16 base features to any timeframe DataFrame."""
    df = df.copy()
    
    # Price features
    df['body'] = df['close'] - df['open']
    df['body_abs'] = abs(df['body'])
    df['candle_range'] = df['high'] - df['low']
    df['close_position'] = (df['close'] - df['low']) / (df['candle_range'] + 1e-10)
    
    # Returns (in terms of bars, not minutes)
    for p in [1, 5, 15, 60]:
        df[f'return_{p}'] = df['close'].pct_change(p)
    
    # True Range & ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # EMAs (normalized to current close)
    for period in [10, 20, 50]:
        ema = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}'] = (df['close'] - ema) / (ema + 1e-10)  # Relative position
    
    # Session (hour from time)
    if 'hour' not in df.columns:
        df['hour'] = df['time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


def add_higher_tf_features(df, project_root):
    """Add 10 higher TF context features (M5, M15, H1, H4, D1 trend/position)."""
    df = df.copy()
    
    for tf_name in ['M5', 'M15', 'H1', 'H4', 'D1']:
        tf_path = project_root / 'data' / 'processed' / f'xauusd_{tf_name}.parquet'
        
        if not tf_path.exists():
            print(f"  WARNING: {tf_path} not found, filling with 0")
            df[f'{tf_name}_trend'] = 0
            df[f'{tf_name}_position'] = 0.5
            continue
        
        tf_df = pd.read_parquet(tf_path).drop(columns=['tf'], errors='ignore')
        
        # Compute higher TF features
        ema20 = tf_df['close'].ewm(span=20, adjust=False).mean()
        tf_df[f'{tf_name}_trend'] = np.sign(tf_df['close'] - ema20)
        tf_range = tf_df['high'] - tf_df['low']
        tf_df[f'{tf_name}_position'] = (tf_df['close'] - tf_df['low']) / (tf_range + 1e-10)
        
        # Merge using nearest-past join
        merge_cols = ['time', f'{tf_name}_trend', f'{tf_name}_position']
        df = pd.merge_asof(
            df.sort_values('time'),
            tf_df[merge_cols].sort_values('time'),
            on='time',
            direction='backward'
        )
    
    return df


def create_labels(df, tp_pips=80, sl_pips=40, forward_bars=12):
    """Create labels based on TP/SL outcomes - VECTORIZED for speed."""
    tp_move = tp_pips * 0.01
    sl_move = sl_pips * 0.01
    n = len(df)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    labels = np.ones(n, dtype=np.int32)  # Default to HOLD
    
    # Process in batches for memory efficiency
    batch_size = 100000
    num_batches = (n - forward_bars) // batch_size + 1
    
    print(f"    Creating labels for {n:,} rows ({num_batches} batches)...")
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, n - forward_bars)
        
        for i in range(start, end):
            current_close = close[i]
            
            # Check future bars
            long_tp_hit = False
            long_sl_hit = False
            short_tp_hit = False
            short_sl_hit = False
            long_tp_bar = 999
            long_sl_bar = 999
            short_tp_bar = 999
            short_sl_bar = 999
            
            for j in range(1, forward_bars + 1):
                idx = i + j
                if idx >= n:
                    break
                
                # Long TP/SL
                if not long_tp_hit and high[idx] >= current_close + tp_move:
                    long_tp_hit = True
                    long_tp_bar = j
                if not long_sl_hit and low[idx] <= current_close - sl_move:
                    long_sl_hit = True
                    long_sl_bar = j
                
                # Short TP/SL
                if not short_tp_hit and low[idx] <= current_close - tp_move:
                    short_tp_hit = True
                    short_tp_bar = j
                if not short_sl_hit and high[idx] >= current_close + sl_move:
                    short_sl_hit = True
                    short_sl_bar = j
            
            long_win = long_tp_bar < long_sl_bar
            short_win = short_tp_bar < short_sl_bar
            
            if long_win and not short_win:
                labels[i] = 2  # LONG
            elif short_win and not long_win:
                labels[i] = 0  # SHORT
            # else: remains HOLD (1)
        
        if batch_idx % 10 == 0:
            print(f"      Batch {batch_idx+1}/{num_batches} done")
    
    return labels


def build_features_for_tf(tf_name: str, project_root: Path, skip_labels: bool = False):
    """Build 26 features for a single timeframe."""
    print(f"\n  Building features for {tf_name}...")
    
    # Load raw data
    tf_path = project_root / 'data' / 'processed' / f'xauusd_{tf_name}.parquet'
    if not tf_path.exists():
        print(f"    ERROR: {tf_path} not found!")
        return None
    
    df = pd.read_parquet(tf_path).drop(columns=['tf'], errors='ignore')
    print(f"    Loaded {len(df):,} bars")
    
    # Add hour if not present
    if 'hour' not in df.columns:
        df['hour'] = df['time'].dt.hour
    
    # Add base features (16)
    df = add_base_features(df)
    
    # Add higher TF features (10)
    df = add_higher_tf_features(df, project_root)
    
    # Create labels (skip for higher TFs if we'll use M1 labels)
    if not skip_labels:
        df['label'] = create_labels(df)
    else:
        df['label'] = 1  # Placeholder, will use M1 labels
    
    # Drop NaN rows
    df = df.dropna(subset=FEATURE_COLS_26)
    print(f"    After dropna: {len(df):,} bars")
    
    return df


def update_m1_with_close(project_root: Path):
    """Add close prices to existing M1 features."""
    print("\n  Adding close prices to existing M1 features...")
    
    data_dir = project_root / 'data' / 'processed'
    m1_raw = pd.read_parquet(data_dir / 'xauusd_M1.parquet')
    m1_raw = m1_raw[['time', 'close']]
    
    for split in ['train', 'val', 'test']:
        path = data_dir / f'features_m1_{split}.parquet'
        df = pd.read_parquet(path)
        
        if 'close' not in df.columns:
            df = df.merge(m1_raw, on='time', how='left')
            df.to_parquet(path, index=False)
            print(f"    Updated features_m1_{split}.parquet with close")
        else:
            print(f"    features_m1_{split}.parquet already has close")
    
    return True


def main():
    print("=" * 70)
    print("Build Features for All Timeframes (Multi-TF Transformer)")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'data' / 'processed'
    
    # Step 1: Update M1 features with close prices
    update_m1_with_close(project_root)
    
    # Step 2: Build features for M5/M15/H1/D1 only (M1 already exists)
    higher_tfs = ['M5', 'M15', 'H1', 'D1']
    all_dfs = {}
    
    for tf in higher_tfs:
        # Skip label creation for higher TFs (much faster)
        df = build_features_for_tf(tf, project_root, skip_labels=True)
        if df is not None:
            all_dfs[tf] = df
    
    print(f"\n  Built features for {len(all_dfs)} higher timeframes")
    
    # Split each TF into train/val/test (70/15/15 chronologically)
    print("\n  Splitting into train/val/test...")
    
    for tf, df in all_dfs.items():
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Keep only required columns
        keep_cols = ['time', 'close'] + FEATURE_COLS_26 + ['label']
        train_df = train_df[[c for c in keep_cols if c in train_df.columns]]
        val_df = val_df[[c for c in keep_cols if c in val_df.columns]]
        test_df = test_df[[c for c in keep_cols if c in test_df.columns]]
        
        # Save
        tf_lower = tf.lower()
        train_df.to_parquet(output_dir / f'features_{tf_lower}_train.parquet', index=False)
        val_df.to_parquet(output_dir / f'features_{tf_lower}_val.parquet', index=False)
        test_df.to_parquet(output_dir / f'features_{tf_lower}_test.parquet', index=False)
        
        print(f"    {tf}: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    
    # Save feature order
    with open(project_root / 'config' / 'features_order_26.json', 'w') as f:
        json.dump(FEATURE_COLS_26, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Feature building complete!")
    print("=" * 70)
    print("\nFiles created:")
    for tf in TIMEFRAMES:
        tf_lower = tf.lower()
        print(f"  data/processed/features_{tf_lower}_train.parquet")
        print(f"  data/processed/features_{tf_lower}_val.parquet")
        print(f"  data/processed/features_{tf_lower}_test.parquet")
    
    print("\nNext: Run train_multi_tf_transformer.py")


if __name__ == "__main__":
    main()
