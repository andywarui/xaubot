"""
JSONL I/O utilities for multi-timeframe data.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict


def load_jsonl_to_df(filepath: Path) -> pd.DataFrame:
    """Load JSONL file to DataFrame with standardized columns."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Normalize column names
    col_map = {
        'Date': 'time', 'date': 'time', 'datetime': 'time',
        'Open': 'open', 'OPEN': 'open',
        'High': 'high', 'HIGH': 'high',
        'Low': 'low', 'LOW': 'low',
        'Close': 'close', 'CLOSE': 'close',
        'Volume': 'volume', 'VOLUME': 'volume'
    }
    df = df.rename(columns=col_map)
    
    # Select and order columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def standardize_time(df: pd.DataFrame, date_format: str) -> pd.DataFrame:
    """Standardize time column to datetime64[ns]."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format=date_format)
    
    # Remove invalid data
    df = df.dropna()
    df = df[df['volume'] > 0]
    df = df[(df['high'] >= df['open']) & (df['high'] >= df['close']) & 
            (df['high'] >= df['low']) & (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])]
    df = df[df['time'].dt.dayofweek < 5]  # Weekdays only
    
    # Sort and deduplicate
    df = df.sort_values('time').drop_duplicates(subset=['time'], keep='first')
    df = df.reset_index(drop=True)
    
    return df


def load_all_timeframes(raw_files: Dict[str, str], date_format: str) -> Dict[str, pd.DataFrame]:
    """Load all timeframe JSONL files and return dict of DataFrames."""
    tf_map = {
        'm1': 'M1', 'm5': 'M5', 'm15': 'M15', 'm30': 'M30',
        'h1': 'H1', 'h4': 'H4', 'd1': 'D1', 'w1': 'W1'
    }
    
    result = {}
    for key, tf_name in tf_map.items():
        filepath = Path(raw_files.get(key, ''))
        if not filepath.exists():
            continue
        
        df = load_jsonl_to_df(filepath)
        if df.empty:
            continue
        
        df = standardize_time(df, date_format)
        df['tf'] = tf_name
        result[tf_name] = df
    
    return result
