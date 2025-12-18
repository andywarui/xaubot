"""
Filter XAUUSD data for London-NY overlap window (13:00-16:59 UTC)
Extracts only the highest liquidity trading hours
"""

import pandas as pd
import numpy as np

def filter_london_ny_overlap():
    """
    Filter standardized data for London-NY overlap hours.
    """
    
    print("=" * 70)
    print("FILTER FOR LONDON-NY OVERLAP (13:00-16:59 UTC)")
    print("=" * 70)
    print()
    
    # Load standardized file
    input_file = 'data/raw/xauusd_m1_standardized.csv'
    print(f"📥 Loading: {input_file}")
    
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['time'].min().date()} → {df['time'].max().date()}")
    print()
    
    # Extract hour
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    
    # Filter for overlap window (13:00-16:59 UTC)
    print("🔧 Applying overlap filter...")
    print("   Target hours: 13:00-16:59 UTC")
    print("   Target days: Monday-Friday (0-4)")
    print()
    
    overlap_df = df[
        (df['hour'] >= 13) & (df['hour'] < 17) &  # 13:00-16:59 UTC
        (df['dayofweek'] < 5)  # Monday-Friday only
    ].copy()
    
    # Remove weekend data
    original_count = len(overlap_df)
    overlap_df = overlap_df[overlap_df['dayofweek'] < 5]
    weekends_removed = original_count - len(overlap_df)
    
    print(f"📊 Filtering Results:")
    print(f"   Original bars: {len(df):,}")
    print(f"   Overlap bars: {len(overlap_df):,}")
    print(f"   Reduction: {(1 - len(overlap_df)/len(df))*100:.1f}%")
    print(f"   Weekends removed: {weekends_removed:,}")
    print()
    
    # Verify hour distribution
    print("⏰ Hour Distribution (should only be 13-16):")
    hour_counts = overlap_df['hour'].value_counts().sort_index()
    for hour, count in hour_counts.items():
        print(f"   {hour:02d}:00 - {count:,} bars")
    print()
    
    # Calculate bars per day
    total_days = (overlap_df['time'].max() - overlap_df['time'].min()).days
    avg_bars_per_day = len(overlap_df) / total_days
    print(f"📈 Statistics:")
    print(f"   Trading days: ~{total_days}")
    print(f"   Avg bars/day: {avg_bars_per_day:.0f} (expected: 240)")
    print()
    
    # Session volatility analysis
    print("💹 Volatility Analysis:")
    overlap_df['price_range'] = overlap_df['high'] - overlap_df['low']
    
    print(f"   Avg 1-min range: ${overlap_df['price_range'].mean():.2f}")
    print(f"   Max 1-min range: ${overlap_df['price_range'].max():.2f}")
    print(f"   Avg volume: {overlap_df['volume'].mean():.5f}")
    print()
    
    # Data quality checks on filtered data
    print("✅ Filtered Data Quality:")
    
    # Check for zeros (market closed during overlap = suspicious)
    zero_volume = (overlap_df['volume'] == 0).sum()
    flat_prices = (overlap_df['price_range'] == 0).sum()
    
    print(f"   Zero volume bars: {zero_volume:,} ({zero_volume/len(overlap_df)*100:.1f}%)")
    print(f"   Flat price bars: {flat_prices:,} ({flat_prices/len(overlap_df)*100:.1f}%)")
    
    if zero_volume / len(overlap_df) > 0.10:
        print(f"   ⚠️  High zero-volume rate (>10%) - some holidays included")
    
    print()
    
    # Save filtered data
    output_file = 'data/processed/xauusd_m1_overlap.csv'
    print(f"💾 Saving filtered data...")
    
    # Keep only essential columns
    overlap_df = overlap_df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    overlap_df.to_csv(output_file, index=False)
    
    print(f"   Output: {output_file}")
    print(f"   Rows: {len(overlap_df):,}")
    print(f"   Size: {overlap_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    print()
    
    # Show sample
    print("📋 Sample filtered data (first 10 rows):")
    print(overlap_df.head(10).to_string(index=False))
    print()
    
    # Summary
    print("=" * 70)
    print("✅ OVERLAP FILTERING COMPLETE!")
    print("=" * 70)
    print()
    print("📊 Final Dataset:")
    print(f"   Total bars: {len(overlap_df):,}")
    print(f"   Date range: {overlap_df['time'].min().date()} → {overlap_df['time'].max().date()}")
    print(f"   Hours: 13:00-16:59 UTC (London-NY overlap)")
    print(f"   Days: Monday-Friday only")
    print(f"   File: {output_file}")
    print()
    print("🎯 Next Step:")
    print("   Run: python feature_engineering.py")
    print("   This will compute 60 features from overlap data")
    
    return True

if __name__ == "__main__":
    filter_london_ny_overlap()
