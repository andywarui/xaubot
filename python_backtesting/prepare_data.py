"""
Prepare XAUUSD M1 data with technical indicators for backtesting

This script generates realistic XAUUSD M1 data or loads existing data,
then calculates all required technical indicators matching the MT5 EA.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators needed for backtesting

    Indicators:
    - TR (True Range)
    - ATR(14)
    - RSI(14)
    - EMA(10, 20, 50)
    - MACD(12, 26, 9)
    - ADX(14)
    - Spread (simulated)
    """
    print("Calculating technical indicators...")

    # True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )

    # ATR(14)
    df['atr_14'] = df['tr'].rolling(window=14).mean()

    # RSI(14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # EMAs
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # MACD(12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # ADX(14) - Simplified calculation
    # Calculate +DI and -DI
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = -df['low'].diff()

    df['plus_dm'] = np.where(
        (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
        df['high_diff'],
        0
    )
    df['minus_dm'] = np.where(
        (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
        df['low_diff'],
        0
    )

    # Smooth DMs and TR
    atr_14 = df['tr'].rolling(window=14).mean()
    plus_di = 100 * (df['plus_dm'].rolling(window=14).mean() / (atr_14 + 1e-8))
    minus_di = 100 * (df['minus_dm'].rolling(window=14).mean() / (atr_14 + 1e-8))

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(window=14).mean()

    # Simulate spread (typical XAUUSD spread is $0.30-$0.80)
    df['spread'] = np.random.uniform(0.30, 0.80, len(df))

    # Multi-timeframe EMAs (simplified - using same data)
    # In full implementation, these would be calculated from aggregated timeframes
    df['ema_20_m15'] = df['ema_20']  # Placeholder
    df['ema_20_h1'] = df['ema_20']   # Placeholder

    # Clean up temporary columns
    df = df.drop(columns=['prev_close', 'high_diff', 'low_diff', 'plus_dm', 'minus_dm'])

    print(f"✓ Indicators calculated for {len(df)} bars")
    return df

def generate_realistic_xauusd_data(
    start_date: str = '2022-01-01',
    end_date: str = '2024-12-31',
    base_price: float = 1900.0
) -> pd.DataFrame:
    """
    Generate realistic XAUUSD M1 data for backtesting

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        base_price: Base gold price

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Generating XAUUSD M1 data from {start_date} to {end_date}...")

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Generate minute-by-minute timestamps
    # Skip weekends (Forex markets closed)
    timestamps = []
    current = start
    while current <= end:
        # Skip Saturday (5) and Sunday (6)
        if current.weekday() < 5:
            # Trading hours: 00:00 to 23:59 (24/5 market)
            for hour in range(24):
                for minute in range(60):
                    timestamps.append(current + timedelta(hours=hour, minutes=minute))
        current += timedelta(days=1)

    print(f"  Generated {len(timestamps):,} timestamps")

    # Generate realistic price data
    # Use geometric Brownian motion with realistic parameters
    n_bars = len(timestamps)

    # XAUUSD typical parameters
    volatility = 0.015  # ~1.5% daily volatility
    drift = 0.0001      # Slight upward drift
    dt = 1.0 / (252 * 24 * 60)  # 1 minute timestep

    # Generate random returns
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_bars)

    # Generate price series
    prices = base_price * np.exp(np.cumsum(returns))

    # Add intraday patterns (higher volatility during London/NY overlap)
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        # Increase volatility during active trading hours (8:00-17:00 UTC)
        if 8 <= hour <= 17:
            prices[i] *= (1 + np.random.normal(0, 0.001))

    # Generate OHLC from close prices
    ohlc_data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Simulate realistic intrabar price movement
        bar_volatility = np.random.uniform(0.5, 2.0)  # $0.50-$2.00 range

        open_price = close + np.random.uniform(-0.3, 0.3)
        high = max(open_price, close) + abs(np.random.normal(0, bar_volatility * 0.5))
        low = min(open_price, close) - abs(np.random.normal(0, bar_volatility * 0.5))

        # Simulate volume (arbitrary units)
        volume = np.random.randint(100, 1000)

        ohlc_data.append({
            'time': ts,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })

    df = pd.DataFrame(ohlc_data)
    print(f"✓ Generated {len(df):,} M1 bars")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

    return df

def prepare_backtest_data(
    start_date: str = '2022-01-01',
    end_date: str = '2024-12-31',
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare complete dataset for backtesting

    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        output_file: Optional path to save prepared data

    Returns:
        DataFrame ready for backtesting
    """
    # Generate data
    df = generate_realistic_xauusd_data(start_date, end_date)

    # Calculate indicators
    df = calculate_indicators(df)

    # Forward fill any NaN values (from indicator calculation)
    df = df.ffill().bfill()

    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        print(f"\n✓ Data saved to {output_file}")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return df

if __name__ == "__main__":
    # Prepare 3 years of data for backtesting
    print("="*70)
    print("XAUUSD M1 DATA PREPARATION")
    print("="*70)
    print()

    df = prepare_backtest_data(
        start_date='2022-01-01',
        end_date='2024-12-31',
        output_file='python_backtesting/xauusd_m1_backtest.parquet'
    )

    print("\nData preparation complete!")
    print("Ready for backtesting with:")
    print(f"  - {len(df):,} M1 bars")
    print(f"  - {len(df.columns)} features/indicators")
    print(f"  - Date range: {df['time'].min()} to {df['time'].max()}")
