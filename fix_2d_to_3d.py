from pathlib import Path

path = Path("python_training/prepare_hybrid_features_multi_tf.py")
text = path.read_text()

# Find and replace the load lines to add dimension expansion
old = '''    X = np.load(DATA_DIR / f"X_5tf_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy")'''

new = '''    X = np.load(DATA_DIR / f"X_5tf_{split}.npy")
    # Handle 2D arrays (N, F) -> expand to 3D (N, 1, F)
    if X.ndim == 2:
        X = X[:, None, :]
    y = np.load(DATA_DIR / f"y_{split}.npy")'''

if old not in text:
    print("Pattern not found. Showing lines with np.load:")
    for i, line in enumerate(text.split('\n'), 1):
        if 'np.load' in line:
            print(f"  {i}: {line}")
    raise SystemExit(1)

text = text.replace(old, new)
path.write_text(text)
print("✓ Patched: 2D arrays now auto-expand to 3D")
