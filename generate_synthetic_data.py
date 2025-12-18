"""
Generate Synthetic XAUUSD Data for Testing/Development
=======================================================
This script creates realistic synthetic gold price data for all timeframes.
Use this for testing the training pipeline without downloading the full ~1.6GB dataset.

WARNING: This is synthetic data for TESTING ONLY. 
         For production models, use real market data from data/processed/ (via Git LFS).

Usage:
    python generate_synthetic_data.py

Output:
    - data/processed/xauusd_{M1,M5,M15,H1,D1}.parquet (price data)
    - Note: Feature files must still be generated using build_features_all_tf.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create project structure
directories = [
    'data/processed',
    'python_training/models',
    'trained_models'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✅ Created directory: {directory}")

# Generate synthetic XAUUSD price data
def generate_synthetic_xauusd_data(timeframe='M1', days=365):
    """Generate realistic synthetic gold price data"""
    
    # Timeframe settings
    intervals = {
        'M1': 1,      # 1 minute
        'M5': 5,      # 5 minutes
        'M15': 15,    # 15 minutes
        'H1': 60,     # 1 hour
        'D1': 1440    # 1 day
    }
    
    interval_minutes = intervals[timeframe]
    total_minutes = days * 24 * 60
    num_bars = total_minutes // interval_minutes
    
    # Start with realistic gold price around $1800-2000
    base_price = 1900.0
    
    # Generate time series
    start_time = datetime.now() - timedelta(days=days)
    times = [start_time + timedelta(minutes=i * interval_minutes) for i in range(num_bars)]
    
    # Generate realistic price movements
    prices = []
    current_price = base_price
    
    for i in range(num_bars):
        # Add trend component (slight upward bias for gold)
        trend = 0.0001 * np.sin(i / 1000) if timeframe == 'D1' else 0.0001 * np.sin(i / 100)
        
        # Add volatility based on timeframe
        volatility = {
            'M1': 0.002,
            'M5': 0.004, 
            'M15': 0.008,
            'H1': 0.015,
            'D1': 0.03
        }[timeframe]
        
        # Random walk with mean reversion
        change = np.random.normal(trend, volatility)
        current_price *= (1 + change)
        
        # Keep price in realistic range
        current_price = max(1500, min(2500, current_price))
        prices.append(current_price)
    
    # Generate OHLCV data
    ohlc_data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        spread = price * 0.001  # 0.1% spread
        high = price + np.random.uniform(0, spread)
        low = price - np.random.uniform(0, spread)
        open_price = prices[i-1] if i > 0 else price
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        volume = np.random.randint(100, 1000) * (10 if timeframe == 'M1' else 100)
        
        ohlc_data.append({
            'time': times[i],
            'open': round(open_price, 2),
            'high': round(high, 2), 
            'low': round(low, 2),
            'close': round(price, 2),
            'volume': volume
        })
    
    return pd.DataFrame(ohlc_data)

if __name__ == "__main__":
    # Generate data for all timeframes
    timeframes = ['M1', 'M5', 'M15', 'H1', 'D1']
    
    print("=" * 60)
    print("GENERATING SYNTHETIC XAUUSD DATA FOR TESTING")
    print("=" * 60)
    print("\n⚠️  WARNING: This is SYNTHETIC data for testing only!")
    print("   For production, use real data from Git LFS.\n")
    
    for tf in timeframes:
        print(f"Generating {tf} data...")
        df = generate_synthetic_xauusd_data(tf, days=365)
        
        # Save price data
        output_path = f'data/processed/xauusd_{tf}.parquet'
        df.to_parquet(output_path, index=False)
        print(f"  ✅ Saved: {output_path} ({len(df):,} bars)")
    
    print(f"\n✅ Generated synthetic XAUUSD data for all timeframes")
    print(f"\nNext steps:")
    print(f"  1. Run: python python_training/build_features_all_tf.py")
    print(f"     (generates feature files from price data)")
    print(f"  2. Then proceed with transformer training")
    print(f"\n✅ Environment setup complete!")
