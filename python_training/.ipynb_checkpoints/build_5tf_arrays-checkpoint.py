# python_training/build_5tf_arrays.py

import os
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"

# ------------------------------------------------------------------
# 1) Inspect available .npy files (for your info)
# ------------------------------------------------------------------

def list_npy_files():
    print(f"Scanning {DATA_DIR} for .npy files...")
    if not DATA_DIR.exists():
        print(f"DATA_DIR does not exist: {DATA_DIR}")
        return
    files = sorted([p.name for p in DATA_DIR.glob("*.npy")])
    for f in files:
        print("  ", f)
    if not files:
        print("No .npy files found!")

# ------------------------------------------------------------------
# 2) EXPLICIT mapping: which arrays to merge per split
#    >>> YOU MUST EDIT THESE NAMES to match what list_npy_files() shows <<<
# ------------------------------------------------------------------

SPLIT_FILES = {
    "train": [
        # examples – replace these with your real feature files
        # e.g. "X_m1_train.npy", "X_m5_train.npy", ...
        # or simply ["X_train.npy"] if you only have one
        "X_m1_train.npy",
        "X_m5_train.npy",
        "X_m15_train.npy",
        "X_h1_train.npy",
        "X_d1_train.npy",
    ],
    "val": [
        "X_m1_val.npy",
        "X_m5_val.npy",
        "X_m15_val.npy",
        "X_h1_val.npy",
        "X_d1_val.npy",
    ],
    "test": [
        "X_m1_test.npy",
        "X_m5_test.npy",
        "X_m15_test.npy",
        "X_h1_test.npy",
        "X_d1_test.npy",
    ],
}

Y_FILES = {
    "train": "y_train.npy",
    "val": "y_val.npy",
    "test": "y_test.npy",
}

# ------------------------------------------------------------------
# 3) Merge logic
# ------------------------------------------------------------------

def build_5tf_for_split(split: str):
    feature_files = SPLIT_FILES[split]
    y_file = Y_FILES[split]

    print(f"\n[{split}] Merging:")
    X_list = []
    shapes = []

    for fname in feature_files:
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file for split '{split}': {path}")
        arr = np.load(path)
        X_list.append(arr)
        shapes.append(arr.shape)
        print(f"  {fname} -> shape {arr.shape}")

    # Basic shape check
    N, T = X_list[0].shape[0], X_list[0].shape[1]
    for s in shapes[1:]:
        if s[0] != N or s[1] != T:
            raise ValueError(
                f"Shape mismatch in {split}: {shapes}. "
                "All arrays must share same (N, T)."
            )

    # Concatenate along feature axis
    X_merged = np.concatenate(X_list, axis=-1)  # (N, T, sum F_tf)
    print(f"  => merged shape: {X_merged.shape}")

    # Load y
    y_path = DATA_DIR / y_file
    if not y_path.exists():
        raise FileNotFoundError(f"Missing label file for split '{split}': {y_path}")
    y = np.load(y_path)
    print(f"  y file {y_file} -> shape {y.shape}")

    # Save merged X
    out_X_path = DATA_DIR / f"X_5tf_{split}.npy"
    np.save(out_X_path, X_merged)
    print(f"  Saved {out_X_path.name}")


def main():
    list_npy_files()

    # IMPORTANT: edit SPLIT_FILES above before running
    answer = input("\nDid you edit SPLIT_FILES to match your filenames? [y/N] ")
    if answer.lower() != "y":
        print("Aborting. Edit SPLIT_FILES in this script and rerun.")
        return

    for split in ["train", "val", "test"]:
        build_5tf_for_split(split)

    print("\nDone. You should now have X_5tf_train/val/test.npy in data/processed.")

if __name__ == "__main__":
    main()
