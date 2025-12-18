import numpy as np
import pandas as pd
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# choose a single timeframe to use
TIMEFRAMES = ["m5"]
FEATURE_TEMPLATE = "features_{tf}_{split}.parquet"

# Target column name in your parquet files
TARGET_COL = "label"


def load_feature_array(tf: str, split: str):
    path = PROCESSED_DIR / FEATURE_TEMPLATE.format(tf=tf, split=split)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature parquet: {path}")

    df = pd.read_parquet(path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"{TARGET_COL} not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Target
    y = df[TARGET_COL].values.astype(np.float32)

    # Features: everything except time and target
    feature_cols = [
        c for c in df.columns
        if c not in [TARGET_COL, "time"]
    ]
    X = df[feature_cols].values.astype(np.float32)

    return X, y, feature_cols


def build_5tf_for_split(split: str):
    print(f"\n[{split}] loading per-TF feature arrays...")
    X_list = []
    y_base = None

    for tf in TIMEFRAMES:
        X_tf, y_tf, feat_cols = load_feature_array(tf, split)
        print(f"  tf={tf}: X shape {X_tf.shape}, y shape {y_tf.shape}")
        X_list.append(X_tf)

        if y_base is None:
            y_base = y_tf
        else:
            if y_tf.shape[0] != y_base.shape[0]:
                raise ValueError(
                    f"y length mismatch for split={split}: "
                    f"base={y_base.shape[0]} vs tf={tf} -> {y_tf.shape[0]}"
                )

    # Concatenate along feature dimension
    X_merged = np.concatenate(X_list, axis=-1)  # (N, sum F_tf)
    y = y_base

    np.save(PROCESSED_DIR / f"X_5tf_{split}.npy", X_merged)
    np.save(PROCESSED_DIR / f"y_{split}.npy", y)

    print(f"  => Saved X_5tf_{split}.npy {X_merged.shape}")
    print(f"  => Saved y_{split}.npy {y.shape}")


def main():
    for split in ["train", "val", "test"]:
        build_5tf_for_split(split)
    print("\nDone. 5-TF numpy files are in data/processed.")


if __name__ == "__main__":
    main()
