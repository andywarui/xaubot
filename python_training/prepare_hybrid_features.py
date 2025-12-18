"""
Prepare hybrid features by adding Transformer predictions to the original 26 features.

This script:
1. Loads the trained Transformer model
2. Generates price change predictions for each sequence
3. Adds 'transformer_signal' as feature #27
4. Saves new parquet files for hybrid LightGBM training

Usage:
    python prepare_hybrid_features.py
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from train_transformer_price import TimeSeriesTransformer, ForexSequenceDataset


def load_transformer_model(model_dir: Path, device: str):
    """Load trained Transformer model and its config."""
    
    # Load config
    with open(model_dir / 'transformer_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = TimeSeriesTransformer(
        feature_size=config['feature_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        seq_length=config['seq_length']
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_dir / 'transformer_price.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, config


def generate_predictions(model, data_scaled, close_prices, config, device):
    """
    Generate Transformer predictions for all sequences in the data.
    Returns array of predictions aligned with the data.
    """
    seq_length = config['seq_length']
    pred_bars = config['pred_bars']
    
    # Create dataset
    dataset = ForexSequenceDataset(data_scaled, close_prices, seq_length, pred_bars)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            predictions.extend(output.cpu().numpy().flatten())
    
    # Predictions are for indices [seq_length-1 : len(data) - pred_bars]
    # Pad beginning and end with NaN
    full_predictions = np.full(len(data_scaled), np.nan)
    
    # Predictions start at index seq_length - 1 (after we have seq_length bars)
    start_idx = seq_length - 1
    end_idx = start_idx + len(predictions)
    full_predictions[start_idx:end_idx] = predictions
    
    return full_predictions


def main():
    print("=" * 70)
    print("Prepare Hybrid Features (26 + Transformer Signal)")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'python_training' / 'models'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # -------------------------------------------------------------------------
    # Load Transformer Model
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading Transformer model...")
    
    if not (model_dir / 'transformer_price.pth').exists():
        print("ERROR: Transformer model not found!")
        print("Please run train_transformer_price.py first.")
        sys.exit(1)
    
    model, config = load_transformer_model(model_dir, device)
    print(f"  Loaded model with {config['feature_size']} features")
    print(f"  Sequence length: {config['seq_length']}")
    print(f"  Direction accuracy: {config['direction_accuracy']:.2%}")
    
    # Load scaler
    with open(model_dir / 'transformer_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # -------------------------------------------------------------------------
    # Load Feature Data
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading feature data...")
    
    train_df = pd.read_parquet(project_root / 'data' / 'processed' / 'features_m1_train.parquet')
    val_df = pd.read_parquet(project_root / 'data' / 'processed' / 'features_m1_val.parquet')
    test_df = pd.read_parquet(project_root / 'data' / 'processed' / 'features_m1_test.parquet')
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    
    feature_cols = config['feature_cols']
    
    # -------------------------------------------------------------------------
    # Load Close Prices
    # -------------------------------------------------------------------------
    print("\n[3/5] Loading close prices...")
    
    m1_parquet = project_root / 'data' / 'processed' / 'xauusd_M1.parquet'
    m1_df = pd.read_parquet(m1_parquet)
    m1_df = m1_df.drop(columns=['tf'], errors='ignore')
    
    # Merge close prices
    train_df = train_df.merge(m1_df[['time', 'close']], on='time', how='left')
    val_df = val_df.merge(m1_df[['time', 'close']], on='time', how='left')
    test_df = test_df.merge(m1_df[['time', 'close']], on='time', how='left')
    
    # -------------------------------------------------------------------------
    # Generate Predictions
    # -------------------------------------------------------------------------
    print("\n[4/5] Generating Transformer predictions...")
    
    # Train set
    print("  Processing train set...")
    X_train = train_df[feature_cols].values
    X_train_scaled = scaler.transform(X_train)
    train_preds = generate_predictions(model, X_train_scaled, train_df['close'].values, config, device)
    train_df['transformer_signal'] = train_preds
    
    # Validation set
    print("  Processing val set...")
    X_val = val_df[feature_cols].values
    X_val_scaled = scaler.transform(X_val)
    val_preds = generate_predictions(model, X_val_scaled, val_df['close'].values, config, device)
    val_df['transformer_signal'] = val_preds
    
    # Test set
    print("  Processing test set...")
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    test_preds = generate_predictions(model, X_test_scaled, test_df['close'].values, config, device)
    test_df['transformer_signal'] = test_preds
    
    # -------------------------------------------------------------------------
    # Clean and Save
    # -------------------------------------------------------------------------
    print("\n[5/5] Cleaning and saving hybrid features...")
    
    # Drop rows where transformer_signal is NaN (edges)
    train_before = len(train_df)
    val_before = len(val_df)
    test_before = len(test_df)
    
    train_df = train_df.dropna(subset=['transformer_signal'])
    val_df = val_df.dropna(subset=['transformer_signal'])
    test_df = test_df.dropna(subset=['transformer_signal'])
    
    print(f"  Train: {train_before:,} -> {len(train_df):,} (dropped {train_before - len(train_df):,} edge rows)")
    print(f"  Val:   {val_before:,} -> {len(val_df):,} (dropped {val_before - len(val_df):,} edge rows)")
    print(f"  Test:  {test_before:,} -> {len(test_df):,} (dropped {test_before - len(test_df):,} edge rows)")
    
    # Remove close column (not a feature)
    train_df = train_df.drop(columns=['close'], errors='ignore')
    val_df = val_df.drop(columns=['close'], errors='ignore')
    test_df = test_df.drop(columns=['close'], errors='ignore')
    
    # Save hybrid features
    output_dir = project_root / 'data' / 'processed'
    
    train_df.to_parquet(output_dir / 'features_m1_hybrid_train.parquet', index=False)
    val_df.to_parquet(output_dir / 'features_m1_hybrid_val.parquet', index=False)
    test_df.to_parquet(output_dir / 'features_m1_hybrid_test.parquet', index=False)
    
    print(f"\n  Saved: features_m1_hybrid_train.parquet")
    print(f"  Saved: features_m1_hybrid_val.parquet")
    print(f"  Saved: features_m1_hybrid_test.parquet")
    
    # Save updated feature order (27 features)
    hybrid_feature_cols = feature_cols + ['transformer_signal']
    with open(project_root / 'config' / 'features_order_hybrid.json', 'w') as f:
        json.dump(hybrid_feature_cols, f, indent=2)
    
    print(f"  Saved: config/features_order_hybrid.json")
    
    # Summary statistics
    print("\n" + "-" * 70)
    print("Transformer Signal Statistics:")
    print("-" * 70)
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        sig = df['transformer_signal']
        print(f"  {name}:")
        print(f"    Mean: {sig.mean():.4f}")
        print(f"    Std:  {sig.std():.4f}")
        print(f"    Min:  {sig.min():.4f}")
        print(f"    Max:  {sig.max():.4f}")
    
    print("\n" + "=" * 70)
    print("Hybrid features ready!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run train_lightgbm_hybrid.py to train with 27 features")
    print("  2. Run evaluate_hybrid.py to compare performance")


if __name__ == "__main__":
    main()
