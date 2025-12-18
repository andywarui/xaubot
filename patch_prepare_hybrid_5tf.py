from pathlib import Path
import re

path = Path("python_training/prepare_hybrid_features_multi_tf.py")
text = path.read_text()

# 1) Ensure loads exist
if "X_train_5tf = np.load" not in text:
    raise SystemExit("Could not find X_train_5tf load; inspect file manually.")

# 2) Insert [:, None, :] after the three loads (if not already present)
pattern_loads = r"""X_train_5tf\s*=\s*np\.load\([^)]*\)\s*
X_val_5tf\s*=\s*np\.load\([^)]*\)\s*
X_test_5tf\s*=\s*np\.load\([^)]*\)"""

def add_expand_dims(m):
    block = m.group(0)
    if "[:, None, :]" in block:
        return block  # already patched
    return block + """
# Treat each row as a single-step sequence: (N, 1, F)
X_train_5tf = X_train_5tf[:, None, :]
X_val_5tf   = X_val_5tf[:, None, :]
X_test_5tf  = X_test_5tf[:, None, :]
"""

new_text, n1 = re.subn(pattern_loads, add_expand_dims, text, flags=re.MULTILINE)

if n1 == 0:
    raise SystemExit("Failed to patch loads; pattern not found.")

# 3) Replace any 2D shape unpack with 3D
new_text, n2 = re.subn(
    r"N_train\s*,\s*T\s*,\s*F\s*=\s*X_train_5tf\.shape",
    "N_train, T, F = X_train_5tf.shape",
    new_text,
)

if "N_train, T, F = X_train_5tf.shape" not in new_text:
    # Ensure at least one occurrence exists (some scripts may not have it)
    pass

path.write_text(new_text)
print("Patched prepare_hybrid_features_multi_tf.py (added [:, None, :] after loads).")
