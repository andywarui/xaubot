"""
Validate and standardize merged XAUUSD data
- Fix column names (Gmt time → time, uppercase → lowercase)
- Parse timestamps
- Validate date range
- Check for flat price periods (market closed)
"""

import pandas as pd
import numpy as np

def validate_and_standardize():
    """
    Load merged file, standardize format, and validate quality.
    """
    
    print("=" * 70)
    print("VALIDATE & STANDARDIZE MERGED DATA")
    print("=" * 70)
    print()
    
    # Load merged file
    input_file = 'data/raw/xauusd_m1_2023_2025_merged.csv'
    print(f"📥 Loading: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print()
    
    # Standardize column names
    print("🔧 Standardizing column names...")
    column_mapping = {
        'Gmt time': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    df = df.rename(columns=column_mapping)
    print(f"   New columns: {list(df.columns)}")
    print()
    
    # Parse timestamps
    print("⏰ Parsing timestamps...")
    df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M:%S.%f')
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"   Start: {df['time'].min()}")
    print(f"   End: {df['time'].max()}")
    print(f"   Duration: {(df['time'].max() - df['time'].min()).days} days")
    print()
    
    # Extract time components
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['dayofweek'] = df['time'].dt.dayofweek
    
    # Data quality checks
    print("✅ Data Quality Checks:")
    
    # 1. Check for flat prices (market closed)
    df['price_change'] = df['close'].diff().abs()
    flat_bars = (df['price_change'] == 0).sum()
    flat_pct = (flat_bars / len(df)) * 100
    print(f"   Flat price bars: {flat_bars:,} ({flat_pct:.1f}%)")
    
    # 2. OHLC consistency
    high_valid = (df['high'] >= df['open']).all() and (df['high'] >= df['close']).all()
    low_valid = (df['low'] <= df['open']).all() and (df['low'] <= df['close']).all()
    hl_valid = (df['high'] >= df['low']).all()
    
    if high_valid and low_valid and hl_valid:
        print(f"   OHLC consistency: ✓ PASS")
    else:
        print(f"   OHLC consistency: ✗ FAIL")
        if not high_valid:
            print(f"      Issue: High not always >= Open/Close")
        if not low_valid:
            print(f"      Issue: Low not always <= Open/Close")
        if not hl_valid:
            print(f"      Issue: High not always >= Low")
    
    # 3. Price range
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # 4. Volume
    print(f"   Volume range: {df['volume'].min():.5f} - {df['volume'].max():.5f}")
    zero_volume = (df['volume'] == 0).sum()
    print(f"   Zero volume bars: {zero_volume:,} ({zero_volume/len(df)*100:.1f}%)")
    
    # 5. Time gaps
    df['time_diff'] = df['time'].diff().dt.total_seconds() / 60
    gaps = df[df['time_diff'] > 1]
    print(f"   Time gaps (>1 min): {len(gaps):,}")
    
    if len(gaps) > 0:
        print(f"   Largest gap: {gaps['time_diff'].max():.0f} minutes")
    
    print()
    
    # Session analysis
    print("📊 Session Distribution:")
    session_counts = df.groupby('hour').size()
    
    tokyo_hours = range(0, 9)
    london_hours = range(8, 17)
    ny_hours = range(13, 22)
    overlap_hours = range(13, 17)
    
    tokyo_bars = session_counts[session_counts.index.isin(tokyo_hours)].sum()
    london_bars = session_counts[session_counts.index.isin(london_hours)].sum()
    ny_bars = session_counts[session_counts.index.isin(ny_hours)].sum()
    overlap_bars = session_counts[session_counts.index.isin(overlap_hours)].sum()
    
    print(f"   Tokyo (00-09h): {tokyo_bars:,} bars ({tokyo_bars/len(df)*100:.1f}%)")
    print(f"   London (08-17h): {london_bars:,} bars ({london_bars/len(df)*100:.1f}%)")
    print(f"   NY (13-22h): {ny_bars:,} bars ({ny_bars/len(df)*100:.1f}%)")
    print(f"   Overlap (13-17h): {overlap_bars:,} bars ({overlap_bars/len(df)*100:.1f}%)")
    print()
    
    # Save standardized file
    output_file = 'data/raw/xauusd_m1_standardized.csv'
    print(f"💾 Saving standardized file...")
    
    # Keep only essential columns for next step
    df_clean = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_clean.to_csv(output_file, index=False)
    
    print(f"   Output: {output_file}")
    print(f"   Rows: {len(df_clean):,}")
    print()
    
    # Summary
    print("=" * 70)
    print("✅ VALIDATION COMPLETE!")
    print("=" * 70)
    print()
    print("📋 Summary:")
    print(f"   Total bars: {len(df_clean):,}")
    print(f"   Date range: {df_clean['time'].min().date()} → {df_clean['time'].max().date()}")
    print(f"   Duration: {(df_clean['time'].max() - df_clean['time'].min()).days} days (~3 years)")
    print(f"   Overlap bars: {overlap_bars:,} (target for training)")
    print(f"   File: {output_file}")
    print()
    print("🎯 Next Step:")
    print("   Run: python filter_overlap.py")
    print("   This will extract only London-NY overlap hours (13:00-16:59 UTC)")
    
    return True

if __name__ == "__main__":
    validate_and_standardize()
