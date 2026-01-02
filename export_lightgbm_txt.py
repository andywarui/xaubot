"""
Export LightGBM model to text format for DLL loading.
"""
import lightgbm as lgb
from pathlib import Path

def export_model_txt():
    print("=" * 70)
    print("EXPORT LIGHTGBM MODEL TO TEXT FORMAT")
    print("=" * 70)

    # Load model
    model_path = Path("python_training/models/lightgbm_xauusd.pkl")
    print(f"\nLoading: {model_path}")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return False

    model = lgb.Booster(model_file=str(model_path))
    print(f"  Trees: {model.num_trees()}")
    print(f"  Features: {model.num_feature()}")

    # Save as text
    txt_path = Path("python_training/models/lightgbm_xauusd.txt")
    model.save_model(str(txt_path), num_iteration=-1)

    print(f"\nSaved: {txt_path}")
    print(f"Size: {txt_path.stat().st_size // 1024} KB")

    # Copy to MT5 folders
    import shutil
    destinations = [
        "mt5_expert_advisor/Files/lightgbm_xauusd.txt",
        "MT5_XAUBOT/Files/lightgbm_xauusd.txt"
    ]

    for dest in destinations:
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(txt_path), str(dest_path))
        print(f"  Copied to: {dest}")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)

    return True

if __name__ == "__main__":
    success = export_model_txt()
    exit(0 if success else 1)
