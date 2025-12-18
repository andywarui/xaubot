import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import MinMaxScaler

# -----------------------
# Paths and basic config
# -----------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "python_training" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "multi_tf_transformer_price.pth"
SCALER_PATH = MODEL_DIR / "multi_tf_scaler.pkl"
CONFIG_PATH = MODEL_DIR / "multi_tf_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Dataset for 5-TF merged arrays
# X shape: (N, T, F)
# y shape: (N,)
# -----------------------

class MultiTFDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3, f"Expected X shape (N,T,F), got {X.shape}"
        assert y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)
        if y.ndim == 2:
            y = y[:, 0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),         # (T, F)
            torch.tensor(self.y[idx]).unsqueeze(0)  # (1,) -> (1)
        )

# -----------------------
# Transformer model
# -----------------------

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 20,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_fc = nn.Linear(feature_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_fc(x)  # (B, T, d_model)
        x = x + self.pos_embedding[:, : x.size(1), :]  # broadcast
        x = self.encoder(x)  # (B, T, d_model)
        last = x[:, -1, :]   # (B, d_model)
        out = self.fc_out(last)  # (B, 1)
        return out

# -----------------------
# Utility: load pre-merged 5-TF features
# Expecting files already created by your earlier pipeline
#   X_train_5tf.npy, y_train.npy, etc.
# -----------------------

def load_5tf_arrays(split: str):
    """
    Expects:
      DATA_DIR / f"X_5tf_{split}.npy"
      DATA_DIR / f"y_{split}.npy"
    """
    X = np.load(DATA_DIR / f"X_5tf_{split}.npy")
    # Handle 2D arrays (N, F) -> expand to 3D (N, 1, F)
    if X.ndim == 2:
        X = X[:, None, :]
    y = np.load(DATA_DIR / f"y_{split}.npy")
    return X, y

# -----------------------
# Training loop
# -----------------------

def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 20,
    lr: float = 1e-4,
    device: torch.device = DEVICE,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Mixed precision scaler (named amp_scaler to avoid confusion with MinMaxScaler)
    amp_scaler = GradScaler(enabled=(device.type == "cuda"))

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)  # (B, T, F)
            yb = yb.to(device)  # (B, 1)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda")):
                preds = model(xb)          # (B, 1)
                loss = criterion(preds, yb)

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            train_losses.append(loss.item())

        mean_train = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad(), autocast(enabled=(device.type == "cuda")):
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"Epoch {epoch:03d} | Train {mean_train:.6f} | Val {mean_val:.6f}")

    return model

# -----------------------
# Main
# -----------------------

def main():
    print(f"Device: {DEVICE}")

    # 1) Load raw 5-TF arrays
    print("[1/5] Loading 5-TF train/val/test arrays...")
    X_train_5tf, y_train = load_5tf_arrays("train")
    X_val_5tf, y_val = load_5tf_arrays("val")
    X_test_5tf, y_test = load_5tf_arrays("test")

    # X shape: (N, T, F)
    N_train, T, F = X_train_5tf.shape
    print(f"  Train shape: {X_train_5tf.shape}, Test shape: {X_test_5tf.shape}")

    # 2) Fit sklearn MinMaxScaler on *flattened* features, then reshape back
    print("[2/5] Fitting MinMaxScaler on train features...")
    # Flatten time dimension for scaling: (N*T, F)
    X_train_flat = X_train_5tf.reshape(-1, F)
    X_val_flat = X_val_5tf.reshape(-1, F)
    X_test_flat = X_test_5tf.reshape(-1, F)

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(X_train_flat)

    X_train_scaled = feature_scaler.transform(X_train_flat).reshape(N_train, T, F)
    X_val_scaled = feature_scaler.transform(X_val_flat).reshape(X_val_5tf.shape[0], T, F)
    X_test_scaled = feature_scaler.transform(X_test_flat).reshape(X_test_5tf.shape[0], T, F)

    # 3) Build loaders
    print("[3/5] Building Dataloaders...")
    train_ds = MultiTFDataset(X_train_scaled, y_train)
    val_ds = MultiTFDataset(X_val_scaled, y_val)
    test_ds = MultiTFDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

    # 4) Init model and train
    print("[4/5] Initializing Transformer model...")
    model = TimeSeriesTransformer(
        feature_size=F,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        seq_len=T,
    )

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=40,
        lr=1e-4,
        device=DEVICE,
    )

    # 5) Save model, sklearn scaler, and config
    print("[5/5] Saving model, scaler, and config...")
    torch.save(model.state_dict(), MODEL_PATH)

    # IMPORTANT: save only the sklearn MinMaxScaler here,
    # not the torch.cuda.amp.GradScaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(feature_scaler, f)

    config = {
        "feature_size": F,
        "seq_len": T,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "device": str(DEVICE),
        "timeframes": ["m1", "m5", "m15", "h1", "d1"],
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model to:   {MODEL_PATH}")
    print(f"Saved scaler to:  {SCALER_PATH}")
    print(f"Saved config to:  {CONFIG_PATH}")


if __name__ == "__main__":
    main()