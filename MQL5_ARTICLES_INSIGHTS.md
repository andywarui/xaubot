# Critical Insights from MQL5 ONNX Articles

**Date**: January 1, 2026
**Sources**: 4 MQL5 articles on ONNX integration
**Impact**: GAME-CHANGING - Found what we were missing!

---

## 🎯 THE BREAKTHROUGH DISCOVERIES

### 1. ⚠️ **MUST Set Explicit Input/Output Shapes**

**From Article**: "Dynamic dimensions (-1 in ONNX) require explicit shape definition before inference."

**What We Were Missing**:
```mql5
// After loading model, we MUST do this:
OnnxSetInputShape(handle, 0, BATCH_SIZE, INPUT_SIZE);   // [1, 26]
OnnxSetOutputShape(handle, 0, BATCH_SIZE, OUTPUT_SIZE); // [1, 3]
```

**Why It Matters**:
- ONNX models with dynamic shapes (-1) won't work without this
- Causes `ERR_ONNX_INVALID_PARAMETER` (5805) if missing
- **We never did this in ANY of our EAs!**

---

### 2. ⚠️ **MUST Resize Output Array Before Inference**

**From Article**: "Failing to resize output vectors before inference causes ERR_ONNX_INVALID_PARAMETER"

**What We Were Missing**:
```mql5
float output_probs[OUTPUT_SIZE];
ArrayResize(output_probs, OUTPUT_SIZE);      // ← REQUIRED!
ArrayInitialize(output_probs, 0.0);

OnnxRun(handle, ONNX_NO_CONVERSION, input, output_probs);
```

**Why It Matters**:
- MT5 doesn't auto-resize arrays from ONNX output
- Uninitialized arrays cause runtime errors
- **We had arrays but didn't explicitly resize!**

---

### 3. ⚠️ **Data Type Consistency is Critical**

**From Article**: "Data type mismatches trigger errors like 'invalid parameter size'. Use float32 consistently."

**What We Did Right**:
- ✅ We convert `double` → `float` for ONNX input
- ✅ We use `float[]` arrays

**What We Can Improve**:
- Verify all feature calculations use consistent types
- Add validation that input dimensions match exactly

---

### 4. ⚠️ **MUST Validate Predictions Match Python**

**From Article**: "Test accuracy parity - Validate that MT5 predictions match Python results using identical test data"

**What We Never Did**:
We exported ONNX model but NEVER verified:
- Does it make the same predictions as Python?
- Are the probabilities similar?
- Does output sum to 1.0?

**The Fix**:
```mql5
// In OnInit, test with dummy data:
float test_input[26];
float test_output[3];
// ... fill test data ...
OnnxRun(handle, ONNX_NO_CONVERSION, test_input, test_output);

// Verify output
float sum = 0.0;
for(int i = 0; i < 3; i++) sum += test_output[i];
if(sum < 0.95 || sum > 1.05)
    Print("WARNING: Output doesn't look like probabilities!");
```

---

### 5. 🔧 **Multiple Path Fallback Pattern**

**From Article**: Best practice is to try multiple model locations

**What We Should Do**:
```mql5
string paths[] = {
    InpModelPath,
    "Files\\" + InpModelPath,
    "MQL5\\Files\\" + InpModelPath,
    TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + InpModelPath,
    TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + InpModelPath
};

for(int i = 0; i < ArraySize(paths); i++)
{
    g_onnx_handle = OnnxCreate(paths[i], ONNX_DEFAULT);
    if(g_onnx_handle != INVALID_HANDLE)
    {
        Print("✓ Loaded from: ", paths[i]);
        break;
    }
}
```

**Why It Matters**:
- Different MT5 configurations store files in different places
- Strategy Tester uses Common\Files
- Live trading might use Terminal-specific folder
- **We only tried one path!**

---

### 6. ⚠️ **Normalization/Scaling Must Match Training**

**From Article**: "The same scaling technique and its parameters used for training data must be applied in MT5"

**What We're Missing**:
- We calculate raw features but NO normalization
- Training data was likely normalized (StandardScaler or MinMaxScaler)
- MT5 features are on different scales than Python

**The Problem**:
```python
# Python training:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Mean=0, Std=1
model.fit(X_scaled, y)

# MT5 (what we do):
features = calculate_raw_features()  # NO scaling!
prediction = OnnxRun(features)       # Wrong scale!
```

**The Solution**:
1. Save scaler parameters during training
2. Load in MT5
3. Apply same normalization to features

---

### 7. 🔧 **Common Error Codes**

**From Articles**:
- `ERR_ONNX_INVALID_PARAMETER (5805)` - Dimension mismatch or unresized array
- `Invalid parameter size` - Data type mismatch (float vs double)
- Model load fails - File not found or unsupported operators

---

## 🎯 WHAT WE'VE BEEN DOING WRONG

### Our Previous Approach:
```mql5
// ❌ INCOMPLETE IMPLEMENTATION
g_onnx_handle = OnnxCreate(InpModelPath, ONNX_DEFAULT);
// Missing: Shape configuration
// Missing: Test inference
// Missing: Path fallback

float input[26];
float output[];  // ❌ Not resized!
// Missing: ArrayResize(output, 3);

if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input, output))
    return false;  // Just fail silently

// Missing: Output validation
// Missing: Normalization
```

### Proper Approach (From Articles):
```mql5
// ✅ COMPLETE IMPLEMENTATION

// 1. Try multiple paths
for(int i = 0; i < ArraySize(paths); i++)
{
    g_onnx_handle = OnnxCreate(paths[i], ONNX_DEFAULT);
    if(g_onnx_handle != INVALID_HANDLE) break;
}

// 2. Set explicit shapes
OnnxSetInputShape(g_onnx_handle, 0, 1, 26);
OnnxSetOutputShape(g_onnx_handle, 0, 1, 3);

// 3. Test with dummy data
float test_input[26];
float test_output[3];
ArrayResize(test_output, 3);  // ✅ Resize!
ArrayInitialize(test_output, 0.0);

if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, test_input, test_output))
{
    Print("ONNX test failed! Error: ", GetLastError());
    return INIT_FAILED;
}

// 4. Validate output
float sum = 0.0;
for(int i = 0; i < 3; i++) sum += test_output[i];
if(sum < 0.95 || sum > 1.05)
    Print("WARNING: Output validation failed!");

Print("✓ ONNX test passed!");
```

---

## 🚀 IMMEDIATE ACTION PLAN

### Step 1: Use the New FIXED EA ✅
**File**: `MT5_XAUBOT/Experts/XAUUSD_Neural_Bot_FIXED.mq5`

**What It Does**:
- ✅ Sets input/output shapes explicitly
- ✅ Tests model in OnInit with dummy data
- ✅ Validates output format
- ✅ Tries 5 different file paths
- ✅ Properly resizes arrays
- ✅ Detailed error logging

### Step 2: Deploy and Test
```powershell
cd C:\Users\KRAFTLAB\Documents\xaubot
git pull origin claude/mt5-model-research-yfrWh

# Copy FIXED EA
Copy-Item -Path "MT5_XAUBOT\Experts\XAUUSD_Neural_Bot_FIXED.mq5" `
          -Destination "C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\" `
          -Force

# Copy model to BOTH locations
Copy-Item -Path "MT5_XAUBOT\Files\lightgbm_xauusd.onnx" `
          -Destination "C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\" `
          -Force

Copy-Item -Path "MT5_XAUBOT\Files\lightgbm_xauusd.onnx" `
          -Destination "C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Terminal\Common\Files\" `
          -Force

# Compile and test
# MetaEditor → F7
# Strategy Tester → Run
```

### Step 3: Check Initialization Log
Look for these messages:
```
✓ Model loaded from: [path]
✓ ONNX shapes configured: Input[1,26] → Output[1,3]
Testing ONNX inference with dummy data...
Test prediction output:
  Class 0: [value]
  Class 1: [value]
  Class 2: [value]
  Sum: [~1.0]
✓ ONNX model test passed!
✓ All indicators initialized
Bot initialized successfully!
```

**If you see this** → Model works! Continue to backtest
**If you see errors** → Report exact error code and message

---

## 🎓 KEY LEARNINGS SUMMARY

### What Broke Our Implementation:
1. ❌ Never set OnnxSetInputShape/OnnxSetOutputShape
2. ❌ Didn't resize output arrays before inference
3. ❌ Only tried one file path
4. ❌ No validation that model actually works
5. ❌ No normalization of features
6. ❌ Never verified predictions match Python

### What the Fixed EA Does:
1. ✅ Sets explicit shapes (CRITICAL!)
2. ✅ Resizes arrays properly
3. ✅ Tries 5 different paths
4. ✅ Tests model at initialization
5. ✅ Validates output format
6. ✅ Detailed error logging

### Still Missing (Future Work):
- Feature normalization/scaling
- Python-MT5 prediction parity testing
- Model performance validation

---

## 📊 EXPECTED RESULTS

### If Model Loads Successfully:
```
2026.01.01 00:00:00   ========================================
2026.01.01 00:00:00   XAUUSD Neural Bot v3.0 (FIXED ONNX)
2026.01.01 00:00:00   ========================================
2026.01.01 00:00:00   ✓ Model loaded from: Common\Files\lightgbm_xauusd.onnx
2026.01.01 00:00:00   Setting ONNX input/output shapes...
2026.01.01 00:00:00   ✓ ONNX shapes configured: Input[1,26] → Output[1,3]
2026.01.01 00:00:00   Testing ONNX inference with dummy data...
2026.01.01 00:00:00   Test prediction output:
2026.01.01 00:00:00     Class 0: 0.334
2026.01.01 00:00:00     Class 1: 0.333
2026.01.01 00:00:00     Class 2: 0.333
2026.01.01 00:00:00     Sum: 1.000 (should be ~1.0 for probabilities)
2026.01.01 00:00:00   ✓ ONNX model test passed!
2026.01.01 00:00:00   ✓ All indicators initialized
2026.01.01 00:00:00   ========================================
2026.01.01 00:00:00   Bot initialized successfully!
2026.01.01 00:00:00   ========================================
```

### If Model Still Fails:
Look for specific error codes:
- `5805` → Dimension mismatch (check shapes)
- `5024` → File not found (check paths)
- `5003` → Invalid handle (model didn't load)

---

## 🎯 CONCLUSION

**The articles revealed that we were missing CRITICAL initialization steps!**

**Before**: We just called `OnnxCreate()` and hoped it would work
**Now**: We properly configure shapes, test the model, and validate output

**Next Step**: Deploy the FIXED EA and report results!

---

**Created**: January 1, 2026
**Status**: READY FOR TESTING
**Branch**: `claude/mt5-model-research-yfrWh`
**Files Added**: `XAUUSD_Neural_Bot_FIXED.mq5`
