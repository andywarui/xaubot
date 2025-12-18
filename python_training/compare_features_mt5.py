"""
Compare MT5-logged feature vectors against Python-engineered features for parity.
Usage:
  python python_training/compare_features_mt5.py --log mt5_expert_advisor/feature_log.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def load_feature_names(project_root: Path) -> list[str]:
    with open(project_root / "config" / "features_order.json", "r") as f:
        return json.load(f)


def load_mt5_log(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path, sep=";")
    if "time" not in df.columns:
        raise ValueError("MT5 log missing 'time' column")
    df["time"] = pd.to_datetime(df["time"])
    return df


def load_python_features(project_root: Path, feature_names: list[str]) -> pd.DataFrame:
    parts = []
    for name in ["features_m1_train.parquet", "features_m1_val.parquet", "features_m1_test.parquet"]:
        path = project_root / "data" / "processed" / name
        if path.exists():
            cols = ["time"] + feature_names
            df = pd.read_parquet(path, columns=cols)
            parts.append(df)
    if not parts:
        raise FileNotFoundError("No features_m1_*.parquet files found under data/processed")
    return pd.concat(parts, ignore_index=True)


def summarize_diffs(mt5_df: pd.DataFrame, py_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    merged = mt5_df.merge(py_df, on="time", suffixes=("_mt5", ""), how="inner")
    if merged.empty:
        raise ValueError("No overlapping timestamps between MT5 log and Python features")

    rows = []
    for feat in feature_names:
        mt5_col = f"{feat}_mt5"
        if mt5_col not in merged.columns:
            raise ValueError(f"MT5 log missing column {mt5_col}")
        diff = (merged[mt5_col] - merged[feat]).abs()
        rows.append({
            "feature": feat,
            "count": int(diff.count()),
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
        })
    return pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MT5 feature log vs Python features")
    parser.add_argument("--log", type=Path, default=Path("mt5_expert_advisor/feature_log.csv"), help="Path to MT5 feature log CSV")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    feature_names = load_feature_names(project_root)
    mt5_df = load_mt5_log(args.log)

    expected_cols = ["time"] + [f"{n}" for n in feature_names]
    missing = [c for c in expected_cols if c not in mt5_df.columns]
    if missing:
        raise ValueError(f"MT5 log missing columns: {missing}")

    py_df = load_python_features(project_root, feature_names)
    summary = summarize_diffs(mt5_df, py_df, feature_names)

    print(f"MT5 rows: {len(mt5_df):,}")
    print(f"Python feature rows: {len(py_df):,}")
    overlap = len(mt5_df.merge(py_df[['time']], on='time', how='inner'))
    print(f"Overlapping timestamps: {overlap:,}")
    print("\nTop deviations (sorted by max_abs_diff):")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
