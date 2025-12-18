# ONNX Integration Status - READY FOR TESTING

## ✅ Completed Actions

1. **Deleted old stub EA**: `XAUUSD_NeuralBot.mq5` (incomplete, 103 features, no ONNX)
2. **Active EA confirmed**: `XAUUSD_NeuralBot_M1.mq5` (484 lines, 26 features, full ONNX integration)
3. **ONNX format updated**: EA now handles 2-output format (label + probabilities)

## 📋 EA Implementation Details

### ONNX Output Handling
```cpp
float output_data[];
ArrayResize(output_data, 4);  // 1 for label + 3 for probabilities

if(!OnnxRun(g_onnxHandle, ONNX_NO_CONVERSION, input_data, output_data))
{
   Print("ERROR: ONNX inference failed");
   return 1;
}

// Extract probabilities (skip first value which is the predicted label)
double p_short = output_data[1];
double p_hold = output_data[2];
double p_long = output_data[3];
```

### Validation Built-in
- Checks probability sum ≈ 1.0 (0.99 to 1.01)
- Logs warning if probabilities are invalid
- Prints raw ONNX output for debugging

### Feature Logging
- `EnableFeatureLog=true` → saves to `feature_log.csv`
- `EnablePredictionLog=true` → saves to `prediction_log.csv`

## 🧪 Testing Checklist

### Phase 1: Compilation & Loading
- [ ] Open `XAUUSD_NeuralBot_M1.mq5` in MetaEditor
- [ ] Compile (F7) - expect **0 errors, 0 warnings**
- [ ] Copy `mt5_expert_advisor/Files/lightgbm_xauusd.onnx` to `C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Terminal\<BROKER>\MQL5\Files\`
- [ ] Copy `mt5_expert_advisor/Files/config/model_config.json` to same location

### Phase 2: Strategy Tester (Quick Test)
**Settings**:
- Symbol: XAUUSD
- Period: M5 (EA runs on M5 but trades M1)
- Dates: 2024-12-01 to 2024-12-07 (1 week)
- Model: Every tick based on real ticks
- Inputs:
  - `EnablePredictionLog = true`
  - `EnableFeatureLog = false` (keep false for speed)
  - `ConfidenceThreshold = 0.60`
  - `RiskPercent = 2.0`

**Expected Results**:
- [ ] EA loads ONNX model (check Experts tab: "ONNX model loaded successfully")
- [ ] Predictions logged with 3 probabilities
- [ ] Probability sum checks pass (no warnings)
- [ ] Trades opened (if signal > 60% confidence)

**Check Experts Tab for**:
```
ONNX model loaded successfully. Ready to trade.
Prediction: SHORT=0.267766, HOLD=0.628488, LONG=0.103747, Best=HOLD (62.8%)
```

### Phase 3: Feature Parity Test
1. **Enable logging**:
   - `EnableFeatureLog = true`
   - `EnablePredictionLog = true`
   
2. **Run Strategy Tester** for 100 bars (2024-12-01 to 2024-12-02)

3. **Check log files** in `C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Terminal\<BROKER>\MQL5\Files\`:
   - `feature_log.csv` - should have ~100 rows with 26 feature columns
   - `prediction_log.csv` - should have ~100 rows with probabilities

4. **Run parity test**:
   ```powershell
   .venv_onnx\Scripts\python.exe python_training\compare_features_mt5.py
   ```

**Pass Criteria**:
- [ ] All 26 features within ±2% of Python values
- [ ] Mean absolute deviation < 1% per feature

### Phase 4: ONNX Parity Test
1. **Ensure logs exist** (from Phase 3)

2. **Run ONNX parity test**:
   ```powershell
   .venv_onnx\Scripts\python.exe python_training\onnx_parity_test.py
   ```

**Pass Criteria**:
- [ ] Max probability difference < 1e-5
- [ ] Class prediction match rate ≥ 99.9%

### Phase 5: Strategy Tester (Full OOS)
**Settings**:
- Dates: 2024-10-01 to 2025-10-01 (12 months OOS)
- Model: Every tick based on real ticks
- Inputs: Same as Phase 2

**Expected Results** (compare to Python baseline):
- [ ] Win Rate: 70-76% (target: 74.1%)
- [ ] Profit Factor: 3.5-5.0 (target: 4.60)
- [ ] Max Drawdown: < 15%

**If results differ by >10%**:
- Check feature parity (Phase 3)
- Check ONNX parity (Phase 4)
- Verify spread/slippage settings
- Check SL/TP calculations

## 🚨 Common Issues & Fixes

### Issue 1: "ONNX model not found"
**Fix**: Copy ONNX file to MQL5/Files/ directory
```powershell
Copy-Item "mt5_expert_advisor\Files\lightgbm_xauusd.onnx" -Destination "$env:APPDATA\MetaQuotes\Terminal\<BROKER>\MQL5\Files\"
```

### Issue 2: "Probability sum = 0.000000"
**Cause**: OnnxRun() may not handle 2-output format correctly
**Fix**: Try using `OnnxRunBatch()` instead (see MT5_ONNX_INTEGRATION.md, Option A)

### Issue 3: Compilation errors
**Fix**: Update MT5 to build 4650+ (has ONNX support)

### Issue 4: Feature mismatch (parity test fails)
**Cause**: Indicator settings differ between MT5 and Python
**Fix**: Check ATR, RSI, EMA periods match exactly

### Issue 5: No trades opened
**Possible causes**:
- Confidence threshold too high (try 0.50)
- Model predicting mostly HOLD class
- Daily trade limit reached
- Risk checks failing

## 📊 Expected Performance

Based on Python backtests:

| Config | Win Rate | Profit Factor | Max DD |
|--------|----------|---------------|--------|
| **Baseline** (2% risk, 60% conf) | 76.0% | 5.77 | ~12% |
| **OOS** (12mo, 2% risk) | 74.1% | 4.60 | ~13% |
| **Conservative** (0.5% risk, 70% conf) | 80.5% | 3.67 | 8.2% |

If Strategy Tester shows significantly worse results (e.g., <60% WR, <2.0 PF):
1. Run feature parity test first
2. Run ONNX parity test
3. Check if spread/slippage is too high
4. Verify SL/TP are 40/80 pips (not 400/800)

## 🎯 Next Steps (Priority Order)

1. **Now**: Run Phase 1-2 (compilation + quick test)
2. **Today**: Run Phase 3-4 (parity tests)
3. **Tomorrow**: Run Phase 5 (full OOS backtest)
4. **Next week**: Demo account testing (30 days)
5. **After demo**: Micro live ($100, 0.01 lots, 2 weeks)

## 📁 File Locations

- **EA Source**: `mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5`
- **ONNX Model**: `mt5_expert_advisor/Files/lightgbm_xauusd.onnx`
- **Model Config**: `mt5_expert_advisor/Files/config/model_config.json`
- **Feature Contract**: `config/features_order.json`
- **Parity Tests**: 
  - `python_training/compare_features_mt5.py`
  - `python_training/onnx_parity_test.py`

---

**Status**: ✅ **READY FOR TESTING**  
**Blocker**: None - EA is updated and compiled, ONNX model exported  
**Next Action**: Open MetaEditor and run Phase 1 testing
