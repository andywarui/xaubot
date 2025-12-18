"""
Transformer-based price change predictor for XAUUSD M1 data.
This model predicts future price movement (regression) which becomes
feature #27 for the hybrid LightGBM classifier.

Architecture:
- Input: 26 features over 30 timesteps (sequence)
- Output: Predicted % price change over next 5 bars (regression)

Usage:
    python train_transformer_price.py
"""
import sys
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================
SEQ_LENGTH = 30          # Number of past bars to look at
PRED_BARS = 5            # Predict price change over next N bars
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
D_MODEL = 64             # Transformer embedding dimension
N_HEADS = 8              # Attention heads
N_LAYERS = 2             # Encoder layers
DIM_FEEDFORWARD = 256    # FFN dimension
DROPOUT = 0.1


# =============================================================================
# Dataset
# =============================================================================
class ForexSequenceDataset(Dataset):
    """
    Creates sequences of features for Transformer input.
    Target: % price change over next PRED_BARS bars.
    """
    def __init__(self, data: np.ndarray, close_prices: np.ndarray, 
                 seq_length: int = SEQ_LENGTH, pred_bars: int = PRED_BARS):
        self.data = data  # Shape: [num_samples, num_features]
        self.close_prices = close_prices  # For calculating future returns
        self.seq_length = seq_length
        self.pred_bars = pred_bars
    
    def __len__(self):
        # Need seq_length for input + pred_bars for target
        return len(self.data) - self.seq_length - self.pred_bars + 1
    
    def __getitem__(self, idx):
        # Input: seq_length bars of features
        x = self.data[idx : idx + self.seq_length]
        
        # Target: % price change from last bar in sequence to pred_bars ahead
        current_close = self.close_prices[idx + self.seq_length - 1]
        future_close = self.close_prices[idx + self.seq_length + self.pred_bars - 1]
        
        # % change (scaled to make training easier)
        y = ((future_close - current_close) / current_close) * 100
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32)
        )


# =============================================================================
# Transformer Model
# =============================================================================
class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder for time series price prediction.
    
    Architecture:
    - Linear projection: feature_size -> d_model
    - Positional embedding (learnable)
    - Transformer encoder layers
    - Final linear: d_model -> 1 (price change prediction)
    """
    def __init__(
        self,
        feature_size: int = 26,
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
        
        # Project features to d_model dimension
        self.input_fc = nn.Linear(feature_size, d_model)
        
        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # We'll permute manually for clarity
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (regression: predict single value)
        self.fc_out = nn.Linear(d_model, 1)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
    
    def forward(self, src):
        """
        Args:
            src: [batch_size, seq_length, feature_size]
        Returns:
            prediction: [batch_size, 1]
        """
        batch_size, seq_len, _ = src.shape
        
        # Project features: [B, S, F] -> [B, S, D]
        src = self.input_fc(src)
        
        # Add positional embedding
        src = src + self.pos_embedding[:, :seq_len, :]
        
        # Transformer expects: [seq_length, batch_size, d_model]
        src = src.permute(1, 0, 2)
        
        # Encode
        encoded = self.transformer_encoder(src)  # [S, B, D]
        
        # Use last timestep for prediction
        last_step = encoded[-1, :, :]  # [B, D]
        
        # Output
        out = self.fc_out(last_step)  # [B, 1]
        return out


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
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
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(x_batch)
            
            # Direction accuracy: did we predict UP/DOWN correctly?
            pred_direction = (output > 0).float()
            actual_direction = (y_batch > 0).float()
            
            correct += (pred_direction == actual_direction).sum().item()
            total += y_batch.size(0)
    
    return correct / total if total > 0 else 0.0


# =============================================================================
# Main Training
# =============================================================================
def main():
    print("=" * 70)
    print("Transformer Price Predictor Training")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading M1 feature data...")
    
    train_df = pd.read_parquet(project_root / 'data' / 'processed' / 'features_m1_train.parquet')
    val_df = pd.read_parquet(project_root / 'data' / 'processed' / 'features_m1_val.parquet')
    
    print(f"  Train: {len(train_df):,} M1 bars")
    print(f"  Val:   {len(val_df):,} M1 bars")
    
    # Load feature order
    with open(project_root / 'config' / 'features_order.json', 'r') as f:
        feature_cols = json.load(f)
    
    print(f"  Features: {len(feature_cols)}")
    
    # -------------------------------------------------------------------------
    # Check for 'close' price in data
    # -------------------------------------------------------------------------
    # We need close prices for calculating target
    # The features_m1 files don't include close, so we need to load from raw
    print("\n[2/6] Loading close prices for target calculation...")
    
    m1_parquet = project_root / 'data' / 'processed' / 'xauusd_M1.parquet'
    if not m1_parquet.exists():
        print(f"ERROR: {m1_parquet} not found!")
        print("Please run build_features_m1.py first to create the parquet file.")
        sys.exit(1)
    
    m1_df = pd.read_parquet(m1_parquet)
    m1_df = m1_df.drop(columns=['tf'], errors='ignore')
    
    # Merge close prices with feature data based on time
    train_df = train_df.merge(m1_df[['time', 'close']], on='time', how='left')
    val_df = val_df.merge(m1_df[['time', 'close']], on='time', how='left')
    
    # Drop any NaN close prices
    train_df = train_df.dropna(subset=['close'])
    val_df = val_df.dropna(subset=['close'])
    
    print(f"  Train after merge: {len(train_df):,}")
    print(f"  Val after merge:   {len(val_df):,}")
    
    # -------------------------------------------------------------------------
    # Prepare Features
    # -------------------------------------------------------------------------
    print("\n[3/6] Preparing features and scaling...")
    
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    close_train = train_df['close'].values
    close_val = val_df['close'].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"  Feature shape: {X_train_scaled.shape}")
    
    # -------------------------------------------------------------------------
    # Create Datasets
    # -------------------------------------------------------------------------
    print("\n[4/6] Creating sequence datasets...")
    
    train_dataset = ForexSequenceDataset(X_train_scaled, close_train, SEQ_LENGTH, PRED_BARS)
    val_dataset = ForexSequenceDataset(X_val_scaled, close_val, SEQ_LENGTH, PRED_BARS)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences:   {len(val_dataset):,}")
    print(f"  Sequence length: {SEQ_LENGTH}")
    print(f"  Prediction horizon: {PRED_BARS} bars")
    
    # -------------------------------------------------------------------------
    # Initialize Model
    # -------------------------------------------------------------------------
    print("\n[5/6] Initializing Transformer model...")
    
    model = TimeSeriesTransformer(
        feature_size=len(feature_cols),
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
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\n[6/6] Training...")
    print("-" * 70)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 10
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        direction_acc = calculate_direction_accuracy(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""
        
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.6f}  "
              f"Val Loss: {val_loss:.6f}  "
              f"Direction Acc: {direction_acc:.2%}{marker}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # -------------------------------------------------------------------------
    # Final Evaluation
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("\nFinal Evaluation:")
    
    final_direction_acc = calculate_direction_accuracy(model, val_loader, device)
    print(f"  Best Val Loss:     {best_val_loss:.6f}")
    print(f"  Direction Accuracy: {final_direction_acc:.2%}")
    
    # -------------------------------------------------------------------------
    # Save Model & Artifacts
    # -------------------------------------------------------------------------
    print("\nSaving model and artifacts...")
    
    model_dir = project_root / 'python_training' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model
    torch.save(model.state_dict(), model_dir / 'transformer_price.pth')
    print(f"  Saved: {model_dir / 'transformer_price.pth'}")
    
    # Save scaler
    with open(model_dir / 'transformer_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved: {model_dir / 'transformer_scaler.pkl'}")
    
    # Save config
    config = {
        'seq_length': SEQ_LENGTH,
        'pred_bars': PRED_BARS,
        'feature_size': len(feature_cols),
        'd_model': D_MODEL,
        'nhead': N_HEADS,
        'num_layers': N_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'best_val_loss': float(best_val_loss),
        'direction_accuracy': float(final_direction_acc),
        'feature_cols': feature_cols
    }
    
    with open(model_dir / 'transformer_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {model_dir / 'transformer_config.json'}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run prepare_hybrid_features.py to add transformer predictions to features")
    print("  2. Retrain LightGBM with 27 features")
    print("  3. Run evaluate_hybrid.py to compare performance")


if __name__ == "__main__":
    main()
