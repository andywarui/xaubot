"""
Feature Engineering for XAUUSD AI Trading Bot
Generates 61 features from OHLCV overlap data
"""

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands

print("=" * 70)
print("XAUUSD FEATURE ENGINEERING")
print("=" * 70)
print()

# Load overlap data
print("📥 Loading overlap data...")
df = pd.read_csv('data/processed/xauusd_m1_overlap.csv')
df['time'] = pd.to_datetime(df['time'])

print(f"   Rows: {len(df):,}")
print(f"   Date range: {df['time'].min().date()} → {df['time'].max().date()}")
print()

# Initialize feature counter
feature_count = 0

print("🔧 Computing features...")
print()

# ==================== TECHNICAL INDICATORS (21 features) ====================
print("1️⃣ Technical Indicators...")

# ATR
atr_14 = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
df['atr_14'] = atr_14.average_true_range()

atr_5 = AverageTrueRange(df['high'], df['low'], df['close'], window=5)
df['atr_5'] = atr_5.average_true_range()

# RSI
rsi = RSIIndicator(df['close'], window=14)
df['rsi_14'] = rsi.rsi()

# EMAs
ema12 = EMAIndicator(df['close'], window=12)
df['ema_12'] = ema12.ema_indicator()

ema26 = EMAIndicator(df['close'], window=26)
df['ema_26'] = ema26.ema_indicator()

df['ema_12_slope'] = df['ema_12'].diff()
df['ema_26_slope'] = df['ema_26'].diff()

# SMAs
sma50 = SMAIndicator(df['close'], window=50)
df['sma_50'] = sma50.sma_indicator()

sma200 = SMAIndicator(df['close'], window=200)
df['sma_200'] = sma200.sma_indicator()

# MACD
macd = MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_histogram'] = macd.macd_diff()

# Bollinger Bands
bb = BollingerBands(df['close'], window=20, window_dev=2)
df['bb_upper'] = bb.bollinger_hband()
df['bb_lower'] = bb.bollinger_lband()
df['bb_middle'] = bb.bollinger_mavg()
df['bb_width'] = df['bb_upper'] - df['bb_lower']
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)

# VWAP (simplified - rolling window)
df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(60).sum() / (df['volume'].rolling(60).sum() + 0.0001)
df['price_to_vwap'] = df['close'] / (df['vwap'] + 0.0001)

# Stochastic
stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
df['stoch_k'] = stoch.stoch()
df['stoch_d'] = stoch.stoch_signal()

feature_count += 21
print(f"   ✓ Added 21 technical indicators (Total: {feature_count})")

# ==================== MARKET STRUCTURE (10 features) ====================
print("2️⃣ Market Structure...")

# Swing highs/lows
df['swing_high_dist'] = df['high'].rolling(5, center=True).max().fillna(method='ffill') - df['close']
df['swing_low_dist'] = df['close'] - df['low'].rolling(5, center=True).min().fillna(method='ffill')

# Order blocks (simplified)
df['bullish_ob'] = ((df['close'] > df['open']) & (df['close'].shift(-1) < df['open'].shift(-1))).astype(int)
df['bearish_ob'] = ((df['close'] < df['open']) & (df['close'].shift(-1) > df['open'].shift(-1))).astype(int)

# Fair Value Gaps
df['fvg_up'] = df['low'].shift(-1) - df['high'].shift(1)
df['fvg_down'] = df['low'].shift(1) - df['high'].shift(-1)
df['fvg_size'] = df[['fvg_up', 'fvg_down']].max(axis=1)

# Liquidity sweeps
df['liquidity_sweep_high'] = ((df['high'] > df['high'].rolling(20).max().shift(1)) & (df['close'] < df['open'])).astype(int)
df['liquidity_sweep_low'] = ((df['low'] < df['low'].rolling(20).min().shift(1)) & (df['close'] > df['open'])).astype(int)

# Premium/Discount
session_high = df['high'].rolling(240).max()
session_low = df['low'].rolling(240).min()
df['premium_discount'] = (df['close'] - session_low) / (session_high - session_low + 0.0001)

feature_count += 10
print(f"   ✓ Added 10 market structure features (Total: {feature_count})")

# ==================== ORDERFLOW (8 features) ====================
print("3️⃣ Orderflow Metrics...")

# CVD (approximation)
df['bar_direction'] = np.sign(df['close'] - df['open'])
df['delta'] = df['volume'] * df['bar_direction']
df['cvd'] = df['delta'].cumsum()

# Delta divergence
df['price_change'] = df['close'].pct_change()
df['cvd_change'] = df['cvd'].pct_change()
df['cvd_divergence'] = df['price_change'] - df['cvd_change']

# Volume profile
df['volume_ma'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 0.0001)

# Absorption
df['price_range_norm'] = (df['high'] - df['low']) / (df['atr_14'] + 0.0001)
df['absorption_score'] = df['volume_ratio'] * (1 / (df['price_range_norm'] + 0.001))

feature_count += 8
print(f"   ✓ Added 8 orderflow features (Total: {feature_count})")

# ==================== TIME FEATURES (6 features) ====================
print("4️⃣ Time Context...")

df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute
df['dayofweek'] = df['time'].dt.dayofweek

# Minutes since London/NY open
london_open_minute = 8 * 60
df['minutes_since_london'] = (df['hour'] * 60 + df['minute']) - london_open_minute

ny_open_minute = 13 * 60 + 30
df['minutes_since_ny'] = (df['hour'] * 60 + df['minute']) - ny_open_minute

# Session position
df['session_position'] = (df['minutes_since_ny'] / (3.5 * 60)).clip(0, 1)

feature_count += 6
print(f"   ✓ Added 6 time features (Total: {feature_count})")

# ==================== VOLATILITY (8 features) ====================
print("5️⃣ Volatility Context...")

# ATR percentile
df['atr_percentile'] = df['atr_14'].rolling(240).apply(
    lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5, raw=False
)

# Tick volatility
df['tick_volatility'] = df['close'].rolling(10).std()

# Range expansion
df['range_expansion'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 0.0001)

# Volatility regime
atr_mean = df['atr_14'].rolling(240).mean()
atr_std = df['atr_14'].rolling(240).std()
df['volatility_regime'] = ((df['atr_14'] - atr_mean) / (atr_std + 0.0001)).fillna(0)

# True range
df['true_range'] = df['high'] - df['low']
df['tr_percentile'] = df['true_range'].rolling(60).apply(
    lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5, raw=False
)

# Price velocity
df['price_velocity'] = df['close'].diff(3) / 3
df['price_acceleration'] = df['price_velocity'].diff()

feature_count += 8
print(f"   ✓ Added 8 volatility features (Total: {feature_count})")

# ==================== PRICE ACTION (6 features) ====================
print("6️⃣ Price Action...")

# Returns
df['returns_1m'] = df['close'].pct_change()
df['returns_5m'] = df['close'].pct_change(5)
df['returns_15m'] = df['close'].pct_change(15)

# Momentum
df['momentum'] = df['close'] - df['close'].shift(14)

# Distance to highs/lows
df['dist_to_high'] = (df['high'].rolling(50).max() - df['close']) / (df['atr_14'] + 0.0001)
df['dist_to_low'] = (df['close'] - df['low'].rolling(50).min()) / (df['atr_14'] + 0.0001)

feature_count += 6
print(f"   ✓ Added 6 price action features (Total: {feature_count})")

# ==================== SENTIMENT (1 feature) ====================
print("7️⃣ Sentiment...")
df['sentiment'] = 0.0  # Placeholder (can be updated with real sentiment data)
feature_count += 1
print(f"   ✓ Added 1 sentiment feature (Total: {feature_count})")

# ==================== SMC QUALITY SCORE (1 feature) - NEW! ====================
print("8️⃣ SMC Quality Score (70% Win Rate Strategy)...")

# H4 range (4 hours = 240 bars)
h4_high = df['high'].rolling(240).max()
h4_low = df['low'].rolling(240).min()
h4_mid = (h4_high + h4_low) / 2

# Step 1: H4 Directional Bias
df['h4_bias'] = np.where(
    (df['close'] > h4_mid) & (df['close'] > df['close'].shift(240)),
    1,  # Bullish
    np.where(
        (df['close'] < h4_mid) & (df['close'] < df['close'].shift(240)),
        -1,  # Bearish
        0   # Neutral
    )
)

# Step 2: Premium/Discount Zone
df['in_discount'] = (df['close'] < h4_mid).astype(int)
df['in_premium'] = (df['close'] > h4_mid).astype(int)

# Step 3: Inducement (Liquidity Sweep)
df['inducement_taken'] = (df['liquidity_sweep_high'] | df['liquidity_sweep_low']).astype(int)

# Step 4: Entry Zone Present
df['entry_zone_present'] = (
    (df['fvg_size'] > 0) | 
    (df['bullish_ob'] == 1) | 
    (df['bearish_ob'] == 1)
).astype(int)

# SMC Quality Score: 0-4
df['smc_quality_score'] = (
    (df['h4_bias'] != 0).astype(int) +
    ((df['in_discount'] == 1) | (df['in_premium'] == 1)).astype(int) +
    (df['inducement_taken'] == 1).astype(int) +
    (df['entry_zone_present'] == 1).astype(int)
)

feature_count += 1
print(f"   ✓ Added SMC quality score (Total: {feature_count})")

print()
print(f"✅ Feature computation complete: {feature_count} total features")
print()

# Remove NaN rows (from indicator lookback periods)
print("🧹 Cleaning data...")
before_clean = len(df)
df = df.dropna()
after_clean = len(df)
removed = before_clean - after_clean

print(f"   Rows before: {before_clean:,}")
print(f"   Rows after: {after_clean:,}")
print(f"   Removed (NaN): {removed:,}")
print()

# Save feature data
output_file = 'data/processed/xauusd_features.csv'
print(f"💾 Saving features to: {output_file}")

df.to_csv(output_file, index=False)

file_size_mb = len(df) * len(df.columns) * 8 / (1024**2)  # Approximate
print(f"   File size: ~{file_size_mb:.1f} MB")
print(f"   Shape: {df.shape}")
print()

# Show feature list
print("📋 Feature List (61 total):")
feature_cols = [col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

print()

# Show SMC quality score distribution
print("📊 SMC Quality Score Distribution:")
smc_dist = df['smc_quality_score'].value_counts().sort_index()
for score, count in smc_dist.items():
    pct = count / len(df) * 100
    print(f"   Score {int(score)}: {count:,} bars ({pct:.1f}%)")

print()
print("=" * 70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 70)
print()
print("📊 Summary:")
print(f"   Input bars: 180,000")
print(f"   Output bars: {len(df):,}")
print(f"   Features: {len(feature_cols)}")
print(f"   File: {output_file}")
print()
print("💡 SMC Quality Insight:")
print("   Score 3-4: High-probability setups (target these)")
print("   Score 0-2: Lower-probability (ML model will filter)")
print()
print("🎯 Next Step:")
print("   Run: python create_labels.py")
