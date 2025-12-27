# Final Analysis: Why Backtest is Trading at a Loss / Zero Trades

**Date**: 2025-12-27
**Question**: "why is the backtest trading at a loss?"

---

## Executive Summary

After comprehensive investigation including downloading real XAUUSD data from Kaggle and testing multiple model configurations, we discovered:

**ROOT CAUSE**: The pre-trained models require a **Transformer component** that generates `multi_tf_signal` - a critical feature we cannot calculate without the Transformer model.

---

## Complete Testing History

| Test | Model | Data Source | Features | multi_tf_signal | Result |
|------|-------|-------------|----------|-----------------|---------|
| 1 | lightgbm_synthetic.onnx | Synthetic GBM | 26 | N/A | **149 trades, -76.41% loss** |
| 2 | lightgbm_balanced.onnx | Synthetic GBM | 27 | 0.0 (placeholder) | **0 trades** |
| 3 | lightgbm_balanced.onnx | **REAL Kaggle data** | 27 | 0.0 (placeholder) | **0 trades** |

---

## Why Test #1 Lost Money (-76% Loss)

**Model**: `lightgbm_synthetic.onnx` (trained on synthetic data)

### Problems:
1. **Trained on random noise**: Used Geometric Brownian Motion (GBM) which has no real market patterns
2. **Low win rate**: 28.86% (below the 33.3% break-even for 2:1 risk/reward)
3. **Overfitting**: 81% training accuracy but only 29% live win rate (52% gap!)

### Why it Failed:
```
What the model "learned":
- Random patterns in synthetic data
- Noise, not signal

What happened in backtest:
- Different random noise
- Patterns don't match
- Most trades lose
```

**Result**: 106 losing trades, 43 winning trades = -$7,641 loss

---

## Why Tests #2 & #3 Got Zero Trades

**Model**: `lightgbm_balanced.onnx` (trained on 4.6M real XAUUSD samples)

### The Discovery

Found in training code (`prepare_hybrid_features_multi_tf-checkpoint.py:186`):

```python
hybrid_df = pd.DataFrame({
    "time": times,
    "close": close_prices,
    "label": labels,
    "multi_tf_signal": tf_predictions  # ← TRANSFORMER OUTPUT!
})
```

**`multi_tf_signal` is NOT a calculated feature - it's the prediction from a separate Transformer model!**

### The Architecture

The `lightgbm_balanced` model is part of a **two-stage ensemble**:

```
Stage 1: Transformer Model
├─ Input: 5 timeframes (M1, M5, M15, H1, H4) × 26 features = 130 features
├─ Architecture: LSTM/Transformer sequence model
└─ Output: Signal score (-1 to +1) → This becomes multi_tf_signal

Stage 2: LightGBM Model
├─ Input: multi_tf_signal + 26 M1 features = 27 total features
└─ Output: Final trading decision (SHORT/HOLD/LONG)
```

### What We're Missing

1. Transformer model files are in Git LFS (inaccessible - HTTP 502)
   - `multi_tf_transformer_price.pth` (132 bytes = LFS pointer)
   - `multi_tf_scaler.pkl` (129 bytes = LFS pointer)

2. Without Transformer:
   - We set `multi_tf_signal = 0.0` (neutral/no signal)
   - LightGBM interprets this as "Transformer says don't trade"
   - Predicts HOLD for every single bar
   - **Result: 0 trades**

---

## What We Accomplished

✅ **Downloaded Real XAUUSD Data**:
- Source: Kaggle (6.7M rows, 2004-2024)
- Filtered: 1.06M rows (2022-2024)
- File: `data/raw/XAU_1m_data.csv` (328MB)
- Processed: `python_backtesting/xauusd_m1_real_backtest.parquet` (91MB)

✅ **Identified Root Cause**:
- Understood the ensemble architecture
- Found why 0 trades occur
- Located missing Transformer component

✅ **Created Complete Infrastructure**:
- Real data processing pipeline
- ONNX conversion scripts
- 27-feature calculation methods
- Comprehensive analysis documents

---

## Solutions (In Order of Preference)

### Option 1: Train New LightGBM on 26 Features (RECOMMENDED)

**Pros**:
- ✅ Single model (no Transformer dependency)
- ✅ Can train on real Kaggle data we already have
- ✅ Direct, straightforward approach

**Cons**:
- ⚠️ Lower accuracy than ensemble (but will actually trade!)

**Steps**:
1. Use real data: `data/raw/XAU_1m_data.csv`
2. Calculate 26 features (no multi_tf_signal)
3. Train LightGBM
4. Export to ONNX
5. Backtest and tune

**Estimated Time**: 1-2 hours

---

### Option 2: Get Transformer Model Working

**Pros**:
- ✅ Use pre-trained ensemble (potentially best performance)
- ✅ Full two-stage architecture

**Cons**:
- ❌ Transformer files in Git LFS (inaccessible)
- ❌ Need to export from PyTorch to ONNX
- ❌ Complex multi-timeframe feature calculation
- ❌ Higher risk of errors

**Steps**:
1. Get Transformer `.pth` file from LFS or retrain
2. Calculate 130 features (5 TFs × 26)
3. Export Transformer to ONNX
4. Implement multi-TF feature pipeline
5. Run ensemble backtest

**Estimated Time**: 4-8 hours (if Transformer accessible)

---

### Option 3: Use Simpler Validation (Quick Test)

**Pros**:
- ✅ Can test immediately with synthetic model
- ✅ See if it's just validation being too strict

**Cons**:
- ❌ Still using synthetic-trained model
- ❌ Won't solve fundamental data quality issue

**Steps**:
1. Lower confidence threshold: 60% → 40%
2. Disable some hybrid validation layers
3. Rerun backtest

**Estimated Time**: 15 minutes

---

## Recommended Next Steps

### Immediate (TODAY):

**Train a new 26-feature LightGBM model on real data**

```bash
# 1. Create training script
python_training/train_lightgbm_real_26features.py

# 2. Use the Kaggle data we already have
Input: data/raw/XAU_1m_data.csv (328MB, 6.7M rows)

# 3. Train on 2022-2023, test on 2024
Split: 80% train, 20% test

# 4. Export to ONNX
Output: python_training/models/lightgbm_real_26features.onnx

# 5. Update backtest to use new model
python_backtesting/run_backtest.py

# 6. Run backtest
Expected: 50-200 trades over 3 years
```

### Short-term (THIS WEEK):

1. **If new model shows promise** (>45% win rate, profit factor >1.2):
   - Implement walk-forward validation
   - Test on out-of-sample 2024 data
   - Tune hyperparameters
   - Deploy to MT5 demo account for paper trading

2. **If results are poor**:
   - Revisit target definition (predict profitability, not direction)
   - Implement dynamic SL/TP based on ATR
   - Add regime detection
   - Consider Option 2 (Transformer ensemble)

### Long-term (NEXT MONTH):

1. Train Transformer model on real multi-timeframe data
2. Create full ensemble system
3. A/B test: Single LightGBM vs Ensemble
4. Deploy best performer to live trading with micro-lots

---

## Files Created/Modified

### Data Files:
- `data/raw/XAU_1m_data.csv` (328MB) - Real XAUUSD from Kaggle
- `python_backtesting/xauusd_m1_real_backtest.parquet` (91MB) - Processed for backtesting

### Scripts:
- `python_backtesting/prepare_real_data.py` - Process Kaggle data
- `python_training/convert_balanced_to_onnx.py` - Convert .txt models to ONNX
- `python_backtesting/diagnose_signals.py` - Signal diagnostic tool

### Models:
- `python_training/models/lightgbm_synthetic.onnx` (781KB) - Synthetic model
- `python_training/models/lightgbm_balanced.onnx` (729KB) - Real ensemble model (needs Transformer)

### Documentation:
- `MODEL_PERFORMANCE_ANALYSIS.md` - Comprehensive research findings
- `FINAL_ANALYSIS_AND_SOLUTION.md` - This document

---

## Key Learnings

1. **Ensemble models require ALL components**: Can't use LightGBM part without Transformer
2. **Synthetic data != Real data**: 81% synthetic accuracy ≠ real performance
3. **Feature importance matters**: `multi_tf_signal` is critical - setting to 0 blocks all trades
4. **Data access is crucial**: Git LFS issues blocked real data for weeks
5. **Simple can be better**: Pure 26-feature model > incomplete 27-feature ensemble

---

## Answer to Original Question

> **"why is the backtest trading at a loss?"**

**Short Answer**:
- Synthetic model (-76% loss): Trained on random data, not real markets
- Real model (0 trades): Missing Transformer component, can't generate signals

**Solution**:
Train a new 26-feature LightGBM model on the real Kaggle data we now have.

---

## Conclusion

The investigation revealed a sophisticated ensemble architecture that requires components we don't have access to. The path forward is to train a simpler, standalone model on real data that can actually execute trades.

**We have everything we need**:
- ✅ 328MB real XAUUSD data
- ✅ Processing pipeline
- ✅ Backtesting infrastructure
- ✅ Understanding of what went wrong

**Next**: Train the 26-feature model and finally get profitable backtesting results.
