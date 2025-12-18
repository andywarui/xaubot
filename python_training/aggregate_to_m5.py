"""
M5 preparation with session filtering.
"""
import sys
import yaml
import pandas as pd
from pathlib import Path


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("=" * 70)
    print("M5 Preparation with Session Filtering")
    print("=" * 70)
    
    config = load_config()
    project_root = Path(__file__).parent.parent
    
    # Load M5 parquet
    m5_parquet = project_root / 'data' / 'processed' / 'xauusd_M5.parquet'
    if not m5_parquet.exists():
        print(f"ERROR: {m5_parquet} not found. Run download_or_import_data.py first.")
        sys.exit(1)
    
    print("\nLoading M5 data...")
    df = pd.read_parquet(m5_parquet)
    df = df.drop(columns=['tf'], errors='ignore')
    print(f"  Loaded {len(df):,} rows")
    
    # Add time features
    print("\nAdding time features...")
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['dayofweek'] = df['time'].dt.dayofweek
    
    # Add session flag
    session_config = config['trading_session']
    df['session_flag'] = (
        (df['hour'] >= session_config['start_hour']) & 
        (df['hour'] < session_config['end_hour'])
    ).astype(int)
    print(f"  Session bars: {df['session_flag'].sum():,} / {len(df):,}")
    
    # Save full M5
    output_dir = project_root / 'data' / 'processed'
    m5_csv = output_dir / 'xauusd_m5_clean.csv'
    df.to_csv(m5_csv, index=False)
    print(f"\nSaved full M5: {m5_csv.name}")
    
    # Save session-only
    df_session = df[df['session_flag'] == 1].copy()
    session_csv = output_dir / 'xauusd_m5_session.csv'
    df_session.to_csv(session_csv, index=False)
    print(f"Saved session M5: {session_csv.name} ({len(df_session):,} rows)")
    
    print("\nM5 preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
