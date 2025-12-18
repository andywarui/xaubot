"""
Multi-timeframe JSONL import to Parquet.
"""
import sys
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from json_io import load_all_timeframes


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("=" * 70)
    print("Multi-Timeframe JSONL Import")
    print("=" * 70)
    
    config = load_config()
    project_root = Path(__file__).parent.parent
    
    # Load all timeframes
    print("\nLoading JSONL files...")
    tf_data = load_all_timeframes(config['raw_files'], config['raw_date_format'])
    
    if not tf_data:
        print("ERROR: No valid JSONL files found")
        sys.exit(1)
    
    # Save each timeframe as parquet
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving Parquet files...")
    for tf_name, df in tf_data.items():
        output_path = output_dir / f'xauusd_{tf_name}.parquet'
        df.to_parquet(output_path, index=False)
        print(f"  {tf_name:4s}: {len(df):8,} rows -> {output_path.name}")
    
    print("\nImport complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
