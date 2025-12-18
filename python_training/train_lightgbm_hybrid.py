"""
Multi-TF Transformer: Uses 5 timeframes (M1, M5, M15, H1, D1) = 130 features.

Architecture:
- Input: 130 features (26×5 TFs) over 30 timesteps
- Output: Predicted % price change over next 5 bars (regression)

This creates a "multi_tf_signal" feature for hybrid LightGBM.

Usage:
    python train_multi_tf_transformer.py
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from torch.amp import autocast, GradScaler
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
TIMEFRAMES = ['m1', 'm5', 'm15', 'h1', 'd1']
SEQ_LENGTH = 30          # Number of past bars to look at
PRED_BARS = 5            # Predict price change over next N bars
BATCH_SIZE = 1024        # Increased for better GPU utilization
EPOCHS = 50              # More epochs for larger model
LEARNING_RATE = 5e-4     # Slightly lower LR for stability
D_MODEL = 128            # Larger for 130 features
N_HEADS = 8              # Attention heads
N_LAYERS = 3             # More layers for complexity
DIM_FEEDFORWARD = 512    # Larger FFN
DROPOUT = 0.15

# 26 features per TF
FEATURE_COLS_26 = [
    'body', 'body_abs', 'candle_range', 'close_position',
    'return_1', 'return_5', 'return_15', 'return_60',
    'tr', 'atr_14', 'rsi_14',
    'ema_10', 'ema_20', 'ema_50',
    'hour_sin', 'hour_cos',
    'M5_trend', 'M5_position',
    'M15_trend', 'M15_position',
    'H1_trend', 'H1_position',
    'H4_trend', 'H4_position',
    'D1_trend', 'D1_position'
]

# =============================================================================
# Dataset
# =============================================================================
class MultiTFSequenceDataset(Dataset):
    """
    Creates sequences from 130-feature data (26×5 TFs).
    Target: % price change over next PRED_BARS bars (using M1 close).
    """
    def __init__(self, data: np.ndarray, close_prices: np.ndarray,
                 seq_length: int = SEQ_LENGTH, pred_bars: int = PRED_BARS):
        self.data = data  # Shape: [N, 130]
        self.close_prices = close_prices  # M1 close prices for target
        self.seq_length = seq_length
        self.pred_bars = pred_bars

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_bars + 1

    def __getitem__(self, idx):
        # Input: seq_length bars of 130 features
        x = self.data[idx: idx + self.seq_length]

        # Target: % price change from M1 close
        current_close = self.close_prices[idx + self.seq_length - 1]
        future_close = self.close_prices[idx + self.seq_length + self.pred_bars - 1]

        # % change (scaled)
        y = ((future_close - current_close) / current_close) * 100

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32)
        )

# =============================================================================
# Transformer Model (Multi-TF)
# =============================================================================
class MultiTFTransformer(nn.Module):
    """
    Transformer encoder for multi-timeframe price prediction.

    Input: [batch, seq_length=30, features=130]
    Output: [batch, 1] (predicted % price change)
    """
    def __init__(
        self,
        feature_size: int = 130,  # 26 × 5 TFs
        d_model: int = D_MODEL,
        nhead: int = N_HEADS,
        num_layers: int = N_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        seq_length: int = SEQ_LENGTH
    ):
        super().__init__()

        self.feature_size = feature_size
        self.d_model = d_model
        self.seq_length = seq_length

        # Project 130 features to d_model
        self.input_fc = nn.Linear(feature_size, d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU for better gradients
            batch_first=True    # [batch, seq, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, src):
        """
        Args:
            src: [batch_size, seq_length, 130]
        Returns:
            prediction: [batch_size, 1]
        """
        # Project features: [B, S, 130] -> [B, S, d_model]
        src = self.input_fc(src)

        # Add positional embedding
        src = src + self.pos_embedding

        # Transformer encode (batch_first=True)
        encoded = self.transformer_encoder(src)  # [B, S, d_model]

        # Layer norm
        encoded = self.layer_norm(encoded)

        # Use last timestep for prediction
        last_step = encoded[:, -1, :]  # [B, d_model]

        # Output
        out = self.fc_out(last_step)  # [B, 1]
        return out

# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            output = model(x_batch)
            loss = criterion(output, y_batch)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)

def calculate_direction_accuracy(model, loader, device):
    """Calculate how often model predicts correct direction."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            output = model(x_batch)

            # Direction accuracy
            pred_direction = (output > 0).float()
            actual_direction = (y_batch > 0).float()

            correct += (pred_direction == actual_direction).sum().item()
            total += y_batch.size(0)

    return correct / total if total > 0 else 0.0

# =============================================================================
# Data Loading
# =============================================================================
def load_and_merge_all_tfs(project_root, split='train'):
    """
    Load features from all 5 TFs and merge by M1 timestamps.
    Returns: X (130 features), close_prices, labels
    """
    data_dir = project_root / 'data' / 'processed'

    # Load M1 as base (determines timestamps)
    m1_path = data_dir / f'features_m1_{split}.parquet'
    if not m1_path.exists():
        raise FileNotFoundError(f"M1 features not found: {m1_path}")

    m1_df = pd.read_parquet(m1_path)
    print(f"    M1 {split}: {len(m1_df):,} rows")

    # Get available feature columns (some may be missing)
    available_cols = [c for c in FEATURE_COLS_26 if c in m1_df.columns]

    # Start with M1 features
    X_m1 = m1_df[available_cols].values
    close_prices = m1_df['close'].values if 'close' in m1_df.columns else None
    labels = m1_df['label'].values if 'label' in m1_df.columns else None
    times = m1_df['time'].values

    # Load other TFs and align to M1 times
    all_X = [X_m1]

    for tf in ['m5', 'm15', 'h1', 'd1']:
        tf_path = data_dir / f'features_{tf}_{split}.parquet'

        if not tf_path.exists():
            print(f"    WARNING: {tf} features not found, padding with zeros")
            X_tf = np.zeros((len(m1_df), len(available_cols)))
        else:
            tf_df = pd.read_parquet(tf_path)
            print(f"    {tf.upper()} {split}: {len(tf_df):,} rows")

            # Merge by nearest-past time
            m1_times = pd.DataFrame({'time': times})
            tf_features = tf_df[['time'] + available_cols]

            merged = pd.merge_asof(
                m1_times.sort_values('time'),
                tf_features.sort_values('time'),
                on='time',
                direction='backward'
            )

            X_tf = merged[available_cols].fillna(0).values

        all_X.append(X_tf)

    # Concatenate: [N, 26] × 5 = [N, 130]
    X_combined = np.concatenate(all_X, axis=1)
    print(f"    Combined shape: {X_combined.shape}")

    return X_combined, close_prices, labels, len(available_cols)

# =============================================================================
# Main Training
# =============================================================================
def main():
    print("=" * 70)
    print("🚀 Multi-TF Transformer Training (130 Features)")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Check if feature files exist
    # -------------------------------------------------------------------------
    print("\n[0/6] Checking feature files...")

    missing_files = []
    for tf in TIMEFRAMES:
        for split in ['train', 'val']:
            path = project_root / 'data' / 'processed' / f'features_{tf}_{split}.parquet'
            if not path.exists():
                missing_files.append(path.name)

    if missing_files:
        print(f"\n  ❌ Missing files: {missing_files}")
        print("  Please run build_features_all_tf.py first!")
        print("\n  Command: python python_training/build_features_all_tf.py")
        sys.exit(1)

    print("  ✅ All feature files found!")

    # -------------------------------------------------------------------------
    # Load Multi-TF Data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading 5-TF feature data...")

    X_train, close_train, labels_train, n_features = load_and_merge_all_tfs(project_root, 'train')
    X_val, close_val, labels_val, _ = load_and_merge_all_tfs(project_root, 'val')

    total_features = X_train.shape[1]
    print(f"\n  Total features: {total_features} (26 × {len(TIMEFRAMES)} TFs)")

    # -------------------------------------------------------------------------
    # Scale Features
    # -------------------------------------------------------------------------
    print("\n[2/6] Scaling features...")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Handle NaN/Inf
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=1.0, neginf=0.0)

    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Val shape:   {X_val_scaled.shape}")

    # -------------------------------------------------------------------------
    # Create Datasets & Loaders
    # -------------------------------------------------------------------------
    print("\n[3/6] Creating sequence datasets...")

    train_dataset = MultiTFSequenceDataset(X_train_scaled, close_train, SEQ_LENGTH, PRED_BARS)
    val_dataset = MultiTFSequenceDataset(X_val_scaled, close_val, SEQ_LENGTH, PRED_BARS)

    # Tuned for a big Lightning GPU machine
    NUM_WORKERS_TRAIN = 8
    NUM_WORKERS_VAL = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences:   {len(val_dataset):,}")
    print(f"  Sequence length: {SEQ_LENGTH}")
    print(f"  Prediction horizon: {PRED_BARS} bars")

    # -------------------------------------------------------------------------
    # Initialize Model
    # -------------------------------------------------------------------------
    print("\n[4/6] Initializing Multi-TF Transformer...")

    model = MultiTFTransformer(
        feature_size=total_features,  # 130
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        seq_length=SEQ_LENGTH
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: {N_LAYERS} layers, {N_HEADS} heads, d_model={D_MODEL}")
    print(f"  Input features: {total_features}")

    criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\n[5/6] Training...")
    print("-" * 70)

    best_val_loss = float('inf')
    best_direction_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 15

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        direction_acc = calculate_direction_accuracy(model, val_loader, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save best model (by direction accuracy)
        if direction_acc > best_direction_acc:
            best_direction_acc = direction_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = " ⭐"
        elif val_loss < best_val_loss * 0.99:  # Also save if loss improves significantly
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
            f"Train: {train_loss:.6f}  "
            f"Val: {val_loss:.6f}  "
            f"Dir Acc: {direction_acc:.2%}  "
            f"LR: {current_lr:.2e}{marker}"
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # -------------------------------------------------------------------------
    # Final Evaluation
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("\n[6/6] Final Evaluation:")

    final_direction_acc = calculate_direction_accuracy(model, val_loader, device)
    final_val_loss = validate_epoch(model, val_loader, criterion, device)

    print(f"  Best Val Loss:      {final_val_loss:.6f}")
    print(f"  Direction Accuracy: {final_direction_acc:.2%}")

    # -------------------------------------------------------------------------
    # Save Model & Artifacts
    # -------------------------------------------------------------------------
    print("\nSaving model and artifacts...")

    model_dir = project_root / 'python_training' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save PyTorch model
    torch.save(model.state_dict(), model_dir / 'multi_tf_transformer_price.pth')
    print(f"  ✅ Saved: multi_tf_transformer_price.pth")

    # Save scaler
    with open(model_dir / 'multi_tf_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Saved: multi_tf_scaler.pkl")

    # Save config
    config = {
        'seq_length': SEQ_LENGTH,
        'pred_bars': PRED_BARS,
        'feature_size': total_features,
        'features_per_tf': n_features,
        'num_timeframes': len(TIMEFRAMES),
        'timeframes': TIMEFRAMES,
        'd_model': D_MODEL,
        'nhead': N_HEADS,
        'num_layers': N_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'best_val_loss': float(final_val_loss),
        'direction_accuracy': float(final_direction_acc),
        'feature_cols_per_tf': FEATURE_COLS_26
    }

    with open(model_dir / 'multi_tf_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✅ Saved: multi_tf_config.json")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🎯 Multi-TF Transformer READY!")
    print("=" * 70)
    print("\nFiles created:")
    print("  • models/multi_tf_transformer_price.pth")
    print("  • models/multi_tf_scaler.pkl")
    print("  • models/multi_tf_config.json")
    print(f"\nDirection Accuracy: {final_direction_acc:.1%}")
    print("\nReady for hybrid feature generation!")
    print("\nNext: python python_training/prepare_hybrid_features_multi_tf.py")

if __name__ == "__main__":
    main()
