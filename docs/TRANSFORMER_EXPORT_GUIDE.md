# Transformer Model Export Guide

Since PyTorch isn't installed in this environment, here's how to export the Transformer model:

## Option 1: Export on Your Training Machine

If you have a machine with PyTorch installed (where you trained the model):

```bash
# Navigate to project
cd /path/to/xaubot

# Install required packages (if not already installed)
pip install torch onnx onnxruntime scikit-learn

# Run export script
python python_training/export_transformer_onnx.py
```

This will create:
- `python_training/models/transformer.onnx`
- `python_training/models/transformer_scaler_params.json`
- `python_training/models/transformer_config.json`

## Option 2: Install PyTorch Here

```bash
cd /home/user/xaubot

# Install PyTorch CPU version (smaller, faster)
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install ONNX
pip3 install onnx onnxruntime

# Run export
python3 python_training/export_transformer_onnx.py
```

## Option 3: Use Pre-Exported Model

If you mentioned "Phase 3 complete" and already have the Transformer ONNX, just copy it to:
```
python_training/models/transformer.onnx
python_training/models/transformer_scaler_params.json
```

## Verification

After export, verify the files exist:
```bash
ls -lh python_training/models/transformer*
```

You should see:
```
transformer.onnx                    (~500KB - 2MB)
transformer_scaler_params.json      (~5-10KB)
transformer_config.json             (~2KB)
```

---

**Next:** Once you have these files, I'll integrate them into the ensemble EA.

**Current Status:** I've created the export script. You just need to run it in an environment with PyTorch.
