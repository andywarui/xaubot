"""
Process real XAUUSD M1 data from Kaggle for backtesting

Source: https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024
File: XAU_1m_data.csv (6.7M rows, 2004-2024)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("="*70)
print("PREPARING REAL XAUUSD M1 DATA FOR BACKTESTING")
print("="*70)
print()

# Paths
project_root = Path(__file__).parent.parent
raw_data_path = project_root / 'data' / 'raw' / 'XAU_1m_data.csv'
output_path = project_root / 'python_backtesting' / 'xauusd_m1_real_backtest.parquet'

print(f"Input:  {raw_data_path}")
print(f"Output: {output_path}")
print()

# Load data
print("Step 1: Loading raw data...")
df = pd.read_csv(raw_data_path, sep=';', parse_dates=['Date'])
print(f"✓ Loaded {len(df):,} rows")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  Columns: {list(df.columns)}")
print()

# Rename columns to match our format
print("Step 2: Renaming columns...")
df = df.rename(columns={
    'Date': 'time',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})
print("✓ Columns renamed")
print()

# Filter for 2022-2024 period (matching our synthetic backtest)
print("Step 3: Filtering for 2022-2024 period...")
df = df[(df['time'] >= '2022-01-01') & (df['time'] <= '2024-12-31')].copy()
print(f"✓ Filtered to {len(df):,} rows")
print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
print()

# Sort by time
print("Step 4: Sorting by time...")
df = df.sort_values('time').reset_index(drop=True)
print("✓ Data sorted")
print()

# Calculate technical indicators
print("Step 5: Calculating technical indicators...")

# True Range
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)

# ATR (14-period)
df['atr_14'] = df['tr'].rolling(window=14, min_periods=1).mean()

# RSI (14-period)
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
rs = gain / (loss + 1e-10)
df['rsi_14'] = 100 - (100 / (1 + rs))

# EMAs
df['ema_10'] = df['close'].ewm(span=10, adjust=False, min_periods=1).mean()
df['ema_20'] = df['close'].ewm(span=20, adjust=False, min_periods=1).mean()
df['ema_50'] = df['close'].ewm(span=50, adjust=False, min_periods=1).mean()

# MACD
ema_12 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
ema_26 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# ADX (simplified - using 14 period)
plus_dm = df['high'].diff()
minus_dm = -df['low'].diff()
plus_dm[plus_dm < 0] = 0
minus_dm[minus_dm < 0] = 0

tr_14 = df['tr'].rolling(window=14, min_periods=1).mean()
plus_di = 100 * (plus_dm.rolling(window=14, min_periods=1).mean() / (tr_14 + 1e-10))
minus_di = 100 * (minus_dm.rolling(window=14, min_periods=1).mean() / (tr_14 + 1e-10))

dx = 100 * abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-10)
df['adx'] = dx.rolling(window=14, min_periods=1).mean()

# Spread (estimate as high-low, or use fixed value)
df['spread'] = df['high'] - df['low']
df['spread'] = df['spread'].clip(lower=0.1, upper=5.0)  # Realistic spread range

print("✓ Technical indicators calculated:")
print(f"  - ATR, RSI, EMA (10, 20, 50)")
print(f"  - MACD, ADX")
print(f"  - Spread")
print()

# Handle NaN values
print("Step 6: Handling missing values...")
initial_nans = df.isna().sum().sum()
df = df.ffill().bfill()
final_nans = df.isna().sum().sum()
print(f"✓ NaN values: {initial_nans} → {final_nans}")
print()

# Save to parquet
print("Step 7: Saving to parquet...")
df.to_parquet(output_path, index=False, compression='snappy')
file_size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"✓ Saved to {output_path.name}")
print(f"  File size: {file_size_mb:.1f} MB")
print()

# Summary statistics
print("="*70)
print("DATA SUMMARY")
print("="*70)
print(f"Total bars: {len(df):,}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
print(f"Average ATR: ${df['atr_14'].mean():.2f}")
print(f"Average RSI: {df['rsi_14'].mean():.1f}")
print()

print("Sample data (first 5 rows):")
print(df[['time', 'open', 'high', 'low', 'close', 'atr_14', 'rsi_14']].head())
print()

print("="*70)
print("PREPARATION COMPLETE")
print("="*70)
print()
print("✓ Real XAUUSD M1 data ready for backtesting")
print(f"✓ Run: python python_backtesting/run_backtest.py")
print()
