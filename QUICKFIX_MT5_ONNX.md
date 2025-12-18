# Quick Fix: MT5 ONNX Integration

## Problem

The ONNX model has **2 outputs** (label + probabilities), but the MT5 EA expects **1 output** (just probabilities).

Current error: `OnnxRun()` fails because it can't handle the 2-output format.

---

## Solution: Modify MT5 EA

Update `XAUUSD_NeuralBot_M1.mq5` around line 366:

### BEFORE (broken):
```cpp
// Run ONNX inference
float input_data[];
ArrayResize(input_data, 26);
ArrayCopy(input_data, features);

float output_data[];
ArrayResize(output_data, 3);  // ❌ Expects single output with 3 values

if(!OnnxRun(g_onnxHandle, ONNX_NO_CONVERSION, input_data, output_data))
{
   Print("ERROR: ONNX inference failed");
   return 1;
}

// Get probabilities
double p_short = output_data[0];
double p_hold = output_data[1];
double p_long = output_data[2];
```

### AFTER (fixed):
```cpp
// Run ONNX inference
float input_data[];
ArrayResize(input_data, 26);
ArrayCopy(input_data, features);

// ONNX model has 2 outputs:
// Output 0: label (1 value)
// Output 1: probabilities (dictionary, converts to 3 values)
float output_data[];
ArrayResize(output_data, 4);  // ✅ 1 for label + 3 for probabilities

// Note: May need to use OnnxRunBatch instead of OnnxRun
// First try: Keep OnnxRun and see if it auto-handles 2 outputs
if(!OnnxRun(g_onnxHandle, ONNX_NO_CONVERSION, input_data, output_data))
{
   Print("ERROR: ONNX inference failed");
   return 1;
}

// Skip first output (label), use probabilities
// MT5 flattens the probability dictionary {0: p0, 1: p1, 2: p2} to [p0, p1, p2]
double p_short = output_data[1];  // Was output_data[0]
double p_hold = output_data[2];   // Was output_data[1]
double p_long = output_data[3];   // Was output_data[2]

// Debug: Print to verify values
if(EnablePredictionLog)
{
   Print(StringFormat("ONNX Raw Output: [%.6f, %.6f, %.6f, %.6f]", 
         output_data[0], output_data[1], output_data[2], output_data[3]));
   Print(StringFormat("Probabilities: SHORT=%.6f, HOLD=%.6f, LONG=%.6f", 
         p_short, p_hold, p_long));
}
```

---

## Alternative: Use OnnxRunBatch

If the above doesn't work (probabilities are wrong), use `OnnxRunBatch()`:

```cpp
// Define output shapes for 2 outputs
// Output 0: label [1]
// Output 1: probabilities - may need [1,3] or just [3]
// Try different configurations if first doesn't work

// Configuration A: Try [1] + [3]
long output_shapes[] = {1, 3};  // label=1, probs=3

float output_data[];
ArrayResize(output_data, 4);  // 1 + 3 = 4

if(!OnnxRunBatch(g_onnxHandle, input_data, output_data, output_shapes))
{
   // Configuration B: Try [1,1] + [1,3]
   long output_shapes2[] = {1, 1, 1, 3};
   ArrayResize(output_data, 4);
   
   if(!OnnxRunBatch(g_onnxHandle, input_data, output_data, output_shapes2))
   {
      Print("ERROR: ONNX inference failed with both configs");
      return 1;
   }
}

// Extract probabilities (skip first value which is the label)
double p_short = output_data[1];
double p_hold = output_data[2];
double p_long = output_data[3];
```

---

## Testing Steps

1. **Make the change** in MetaEditor
2. **Compile** - should have 0 errors
3. **Open Strategy Tester**:
   - Symbol: XAUUSD
   - Period: M5 (EA runs on M5 but trades M1)
   - Dates: 2024-10-01 to 2024-10-07 (1 week test)
   - Model: Every tick based on real ticks
   - Inputs: `EnablePredictionLog=true`

4. **Run test** and check Experts tab for:
   ```
   ONNX Raw Output: [1.000000, 0.267766, 0.628488, 0.103747]
   Probabilities: SHORT=0.267766, HOLD=0.628488, LONG=0.103747
   ```
   
   **Validation**:
   - First value should be 0, 1, or 2 (predicted class)
   - Next 3 values should sum to ~1.0 (probabilities)
   - All probabilities should be in range [0, 1]

5. **If probabilities look wrong**:
   - Try OnnxRunBatch approach
   - Check MQL5 documentation for ONNX output format
   - Add more debug prints to see raw output

---

## Common Issues

### Issue 1: All probabilities are 0 or very large numbers
**Cause**: Wrong array indexing  
**Fix**: Check if probabilities are at indices [0,1,2] or [1,2,3]

### Issue 2: Sum of probabilities ≠ 1.0
**Cause**: Dictionary format not flattened correctly  
**Fix**: Use OnnxRunBatch with explicit output shapes

### Issue 3: Compilation error "OnnxRunBatch not found"
**Cause**: Old MT5 build  
**Fix**: Update MT5 to build 4650+ (released 2024)

### Issue 4: ONNX file not found
**Cause**: File not in MQL5/Files/ directory  
**Fix**: Copy from `mt5_expert_advisor/Files/lightgbm_xauusd.onnx` to `C:\Users\<USER>\AppData\Roaming\MetaQuotes\Terminal\<BROKER_ID>\MQL5\Files\`

---

## Validation Checklist

After the fix works:

- [ ] EA compiles with 0 errors
- [ ] ONNX model loads in OnInit (no error message)
- [ ] Predictions print to log with 3 probabilities
- [ ] Probabilities sum to ~1.0 (within 0.001)
- [ ] Each probability is between 0.0 and 1.0
- [ ] EA opens trades (signal ≠ HOLD)

Once validated:

- [ ] Run feature parity test: `python python_training/compare_features_mt5.py`
- [ ] Run ONNX parity test: `python python_training/onnx_parity_test.py`

---

## Contact

If stuck:
1. Check `MT5_ONNX_INTEGRATION.md` for detailed explanation
2. Review MQL5 ONNX documentation: https://www.mql5.com/en/docs/python_metatrader5/onnxhandle
3. Test with simple ONNX model to isolate issue

---

**File to Edit**: `mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5`  
**Function**: `PredictSignal()` around line 366  
**Change**: Update output array size and indexing for 2-output ONNX model
