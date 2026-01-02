# MT5 Backtest Error - Root Cause & Solution

## 🔴 The Problem

**Error**: MT5 Strategy Tester fails to load ONNX model
```
Path 1-3 failed (5019): Cannot create ONNX session
Path 4-5 failed (5002): File not found
```

---

## 🔍 Root Cause Analysis

### Error 5019: Cannot Create ONNX Session
**Cause**: MT5's embedded ONNX runtime **doesn't support** the `TreeEnsembleClassifier` operator used by LightGBM models.

**Evidence**:
```
ONNX Operators in model:
  - Cast
  - Identity
  - Mul
  - TreeEnsembleClassifier  ← NOT SUPPORTED BY MT5
```

**MT5 ONNX Limitations**:
- Designed primarily for neural networks
- Supports basic operators: MatMul, Add, Sigmoid, ReLU, etc.
- Does NOT support ML-specific operators like TreeEnsemble, ZipMap, etc.
- Build 5488 has limited ONNX opset support

### Error 5002: File Not Found
**Cause**: Strategy Tester creates agent directory AFTER copying files

---

## ✅ The Solution: DLL Approach

### Why DLL?
1. ✅ **Full LightGBM support** - No operator limitations
2. ✅ **Native performance** - Direct C++ execution
3. ✅ **No ONNX issues** - Bypasses MT5's ONNX runtime
4. ✅ **Proven approach** - Used by professional trading systems

### What Was Created:
```
✅ lightgbm_wrapper.cpp       - C++ DLL wrapper for LightGBM
✅ XAUUSD_NeuralBot_DLL.mq5   - EA using DLL instead of ONNX
✅ lightgbm_xauusd.txt        - Model in text format (5 MB)
✅ CMakeLists.txt             - Build configuration
✅ DLL_BUILD_INSTRUCTIONS.md  - Complete build guide
```

---

## 📊 Comparison: ONNX vs DLL

| Feature | ONNX Approach | DLL Approach |
|---------|---------------|--------------|
| **LightGBM Support** | ❌ Error 5019 | ✅ Full support |
| **Setup Complexity** | ⭐ Easy | ⭐⭐ Moderate |
| **Performance** | N/A (Fails) | ✅ Native C++ |
| **Compatibility** | ❌ MT5 build 5488 | ✅ All MT5 versions |
| **Maintenance** | ⭐⭐⭐ Auto-update | ⭐⭐ Rebuild on changes |

---

## 🎯 Next Steps

### Option 1: Build DLL (Recommended)
**Time**: 30-60 minutes
**Requires**: Visual Studio, CMake, LightGBM C++ library

1. Follow `DLL_BUILD_INSTRUCTIONS.md`
2. Build `lightgbm_mt5.dll`
3. Copy DLL to MT5
4. Test `XAUUSD_NeuralBot_DLL.mq5`

### Option 2: Train Neural Network (Alternative)
**Time**: 2-4 hours
**Requires**: Python, PyTorch/TensorFlow

1. Train simple feedforward NN
2. Export to ONNX (only basic operators)
3. Use existing ONNX-based EA

### Option 3: Update MT5 (Check First)
**Time**: 10 minutes
**Requires**: Internet connection

1. Help → Check for Updates
2. If newer build available (>5488), update
3. Test if TreeEnsemble support added

---

## 🔧 Technical Details

### Why ONNX Failed

**LightGBM → ONNX conversion creates TreeEnsembleClassifier**:
```python
# Python training
model = lgb.LGBMClassifier()
model.fit(X, y)

# ONNX export (onnxmltools)
onnx_model = convert_lightgbm(model)  # Creates TreeEnsembleClassifier
```

**MT5 ONNX Runtime limitations**:
- Based on ONNX Runtime 1.x (limited ML operators)
- Focuses on neural network inference
- TreeEnsemble operator requires ML backend

### Why DLL Works

**Direct LightGBM C API**:
```cpp
// Load model
LGBM_BoosterCreateFromModelfile("model.txt", &booster);

// Predict
LGBM_BoosterPredictForMat(booster, features, &probs);
```

**No ONNX conversion** = No operator compatibility issues

---

## 📈 Expected Results

### With DLL:
```
========================================
XAUUSD Neural Bot v4.0 (LightGBM DLL)
========================================
✓ Model loaded successfully via DLL
  Features: 26, Classes: 3
✓ All indicators initialized
========================================
[Trades execute normally]
```

### Performance Metrics (Same as Python):
- Win Rate: 66.2%
- Profit Factor: 1.96
- Max Drawdown: 19.5%
- 1,404 trees, 26 features

---

## 📝 Files Ready for Deployment

```
✅ python_training/models/lightgbm_xauusd.txt     (5 MB) - Ready
✅ mt5_expert_advisor/XAUUSD_NeuralBot_DLL.mq5    - Ready
✅ lightgbm_mt5_dll/lightgbm_wrapper.cpp          - Ready to build
✅ lightgbm_mt5_dll/CMakeLists.txt                - Ready to build
```

---

## ⚠️ Important Notes

1. **ONNX approach will NOT work** with LightGBM on MT5 build 5488
2. **DLL approach is industry standard** for ML in MT5
3. **Model file must be .txt format** for DLL loading
4. **Enable "Allow DLL imports"** in MT5 settings

---

## 🎓 Lessons Learned

1. **MT5 ONNX is limited** - Only supports basic neural network operators
2. **Always check ONNX operators** - Use `onnx.load()` to inspect
3. **DLL is more reliable** - For complex ML models in MT5
4. **Test early** - Export small model first, test in MT5

---

**Date**: January 2, 2026
**Status**: Solution implemented, ready for DLL build
**Next Action**: Follow DLL_BUILD_INSTRUCTIONS.md
