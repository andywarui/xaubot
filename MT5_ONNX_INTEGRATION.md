# MT5 ONNX Integration Guide

## ONNX Model Outputs

The exported ONNX model (`lightgbm_xauusd.onnx`) has **2 outputs**:

1. **Output 0: label** (tensor(int64), shape [1])
   - Predicted class: 0=SHORT, 1=HOLD, 2=LONG
   - This is the argmax of probabilities
   
2. **Output 1: probabilities** (seq(map(int64,tensor(float))))
   - Dictionary mapping class index → probability
   - Format: `{0: p_short, 1: p_hold, 2: p_long}`
   - Sum equals 1.0

## Current MT5 EA Implementation

The EA in `XAUUSD_NeuralBot_M1.mq5` currently calls:

```cpp
float output_data[];
ArrayResize(output_data, 3);

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

## Problem

`OnnxRun()` with `ONNX_NO_CONVERSION` expects a single output tensor, but the model has 2 outputs.

## Solutions

### Option A: Use OnnxRunBatch (Recommended)

```cpp
// Define output shapes (2 outputs)
long output_shapes[] = {
   1,          // Output 0: label [1]
   1, 3        // Output 1: probabilities [1, 3] (dict is converted to array)
};

float output_data[];
ArrayResize(output_data, 4);  // 1 for label + 3 for probabilities

if(!OnnxRunBatch(g_onnxHandle, input_data, output_data, output_shapes))
{
   Print("ERROR: ONNX inference failed");
   return 1;
}

// Skip first output (label), use probabilities
double p_short = output_data[1];
double p_hold = output_data[2];
double p_long = output_data[3];
```

### Option B: Simplify ONNX to Single Output

Modify the ONNX graph to remove the label output and keep only probabilities. This requires:

1. Using `onnx.helper` to manually edit the graph
2. Removing the "label" output node
3. Setting "probabilities" as the only output

**This is more complex** and may not be necessary if Option A works.

### Option C: Train with sklearn-onnx

Instead of `onnxmltools`, use `skl2onnx` with explicit probability output:

```python
from skl2onnx import to_onnx

onnx_model = to_onnx(
    model,
    X[:1].astype(np.float32),
    target_opset=12,
    options={'zipmap': False}
)
```

This should output a single tensor [1, 3] with probabilities.

## Recommended Next Steps

1. **Test Option A** first - update EA to use `OnnxRunBatch()`
2. If that doesn't work, **implement Option C** (retrain with skl2onnx)
3. As a last resort, **manually edit ONNX graph** (Option B)

## Feature Order Contract

**CRITICAL**: The ONNX model expects features in this exact order (26 features):

```json
[
  "body", "body_abs", "candle_range", "close_position", 
  "return_1", "return_5", "return_15", "return_60",
  "tr", "atr_14", "rsi_14", "ema_10", "ema_20", "ema_50",
  "hour_sin", "hour_cos",
  "M5_trend", "M5_position",
  "M15_trend", "M15_position",
  "H1_trend", "H1_position",
  "H4_trend", "H4_position",
  "D1_trend", "D1_position"
]
```

This is stored in `config/features_order.json` and must match:
- Python training data column order
- MT5 `CalculateM1Features()` function output order
- ONNX model input tensor order

**Any mismatch will cause incorrect predictions!**

## Validation Steps

After fixing the ONNX integration:

1. **Compile EA** with no errors
2. **Enable logging**: `EnableFeatureLog=true`, `EnablePredictionLog=true`
3. **Run EA** for 100 bars on Strategy Tester
4. **Run parity test**: `python python_training/onnx_parity_test.py`
5. **Check results**:
   - All 26 features within ±2% of Python values
   - Probabilities match within 1e-5
   - Class predictions match 99.9%+

## Current Status

- ✅ ONNX model exported with 2 outputs (label + probabilities)
- ✅ Python validation: LightGBM vs ONNX match within 1.8e-7
- ✅ Feature order documented in `config/features_order.json`
- ⏳ MT5 EA needs update to handle 2-output ONNX model
- ⏳ Parity test pending EA fix

## Files

- ONNX model: `mt5_expert_advisor/Files/lightgbm_xauusd.onnx`
- Metadata: `mt5_expert_advisor/Files/config/model_config.json`
- Feature contract: `config/features_order.json`
- EA source: `mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5`
- Parity test: `python_training/onnx_parity_test.py`
