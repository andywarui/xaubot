"""
Prepare Hybrid Features for LightGBM using trained Multi-TF Transformer.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "python_training" / "models"

MODEL_PATH = MODEL_DIR / "multi_tf_transformer_price.pth"
SCALER_PATH = MODEL_DIR / "multi_tf_scaler.pkl"
CONFIG_PATH = MODEL_DIR / "multi_tf_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS_26 = [
    "body", "body_abs", "candle_range", "close_position",
    "return_1", "return_5", "return_15", "return_60",
    "tr", "atr_14", "rsi_14",
    "ema_10", "ema_20", "ema_50",
    "hour_sin", "hour_cos",
    "M5_trend", "M5_position",
    "M15_trend", "M15_position",
    "H1_trend", "H1_position",
    "H4_trend", "H4_position",
    "D1_trend", "D1_position"
]


class MultiTFTransformer(nn.Module):
    def __init__(self, feature_size=130, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.15, seq_length=30):
        super().__init__()
        self.input_fc = nn.Linear(feature_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, src):
        src = self.input_fc(src)
        src = src + self.pos_embedding[:, :src.size(1), :]
        encoded = self.transformer_encoder(src)
        encoded = self.layer_norm(encoded)
        return self.fc_out(encoded[:, -1, :])


def load_and_merge_all_tfs(split="train"):
    m1_df = pd.read_parquet(DATA_DIR / f"features_m1_{split}.parquet")
    print(f"    M1 {split}: {len(m1_df):,} rows")
    
    available_cols = [c for c in FEATURE_COLS_26 if c in m1_df.columns]
    X_m1 = m1_df[available_cols].values
    close_prices = m1_df["close"].values if "close" in m1_df.columns else None
    labels = m1_df["label"].values if "label" in m1_df.columns else None
    times = pd.to_datetime(m1_df["time"].values)
    
    all_X = [X_m1]
    for tf in ["m5", "m15", "h1", "d1"]:
        tf_path = DATA_DIR / f"features_{tf}_{split}.parquet"
        if not tf_path.exists():
            X_tf = np.zeros((len(m1_df), len(available_cols)))
        else:
            tf_df = pd.read_parquet(tf_path)
            print(f"    {tf.upper()} {split}: {len(tf_df):,} rows")
            m1_times = pd.DataFrame({"time": times})
            tf_features = tf_df[["time"] + available_cols].copy()
            tf_features["time"] = pd.to_datetime(tf_features["time"])
            merged = pd.merge_asof(
                m1_times.sort_values("time"),
                tf_features.sort_values("time"),
                on="time", direction="backward"
            )
            X_tf = merged[available_cols].fillna(0).values
        all_X.append(X_tf)
    
    X_combined = np.concatenate(all_X, axis=1)
    print(f"    Combined shape: {X_combined.shape}")
    return X_combined, close_prices, labels, times


class SequenceDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length + 1)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.float32)


def generate_predictions(model, data_scaled, seq_length=30, batch_size=2048):
    model.eval()
    dataset = SequenceDataset(data_scaled, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                preds = model(batch)
            predictions.append(preds.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0).flatten()
    return np.concatenate([np.zeros(seq_length - 1), predictions])


def main():
    print("=" * 70)
    print("Preparing Hybrid Features using Trained Transformer")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load model
    print("\n[1/3] Loading trained model...")
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    model = MultiTFTransformer(
        feature_size=config["feature_size"], d_model=config["d_model"],
        nhead=config["nhead"], num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
        seq_length=config["seq_length"]
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print("    Model loaded!")

    # Load training data first to fit scaler
    print("\n[2/3] Loading data and fitting scaler...")
    X_train, close_train, labels_train, times_train = load_and_merge_all_tfs("train")
    
    # Fit MinMaxScaler on training data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    print("    Scaler fitted on training data!")

    # Process each split
    print("\n[3/3] Processing splits...")
    
    splits_data = {
        "train": (X_train, close_train, labels_train, times_train)
    }
    
    # Load val and test
    for split in ["val", "test"]:
        print(f"\n  Loading {split}...")
        X, close_prices, labels, times = load_and_merge_all_tfs(split)
        splits_data[split] = (X, close_prices, labels, times)
    
    # Process all splits
    for split in ["train", "val", "test"]:
        print(f"\n  Processing {split}...")
        X, close_prices, labels, times = splits_data[split]
        
        X_scaled = scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=0.0)
        
        print(f"    Generating predictions...")
        tf_predictions = generate_predictions(model, X_scaled, seq_length=config["seq_length"])
        
        hybrid_df = pd.DataFrame({
            "time": times, "close": close_prices,
            "label": labels, "multi_tf_signal": tf_predictions
        })
        
        m1_df = pd.read_parquet(DATA_DIR / f"features_m1_{split}.parquet")
        for col in [c for c in FEATURE_COLS_26 if c in m1_df.columns]:
            hybrid_df[col] = m1_df[col].values
        
        out_path = DATA_DIR / f"hybrid_features_{split}.parquet"
        hybrid_df.to_parquet(out_path, index=False)
        print(f"    Saved: {out_path}")
        print(f"    Shape: {hybrid_df.shape}")
        print(f"    Label dist: {dict(hybrid_df['label'].value_counts())}")

    # Save the correct scaler for future use
    import pickle
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n    Saved correct scaler to: {SCALER_PATH}")

    print("\n" + "=" * 70)
    print("Hybrid Features Ready!")
    print("Next: python python_training/train_lightgbm_hybrid.py")
    print("=" * 70)


if __name__ == "__main__":
    main()