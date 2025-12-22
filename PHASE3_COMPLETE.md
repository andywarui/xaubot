# Phase 3 MT5 Integration - Complete Implementation Summary

## Overview
Phase 3 implements the complete 2-model hybrid architecture for MT5 integration:
- **Transformer**: Multi-timeframe signal generation [1, 30, 130] → [1, 1]
- **LightGBM**: Final classification with 27 features → [HOLD, BUY, SELL]

## Files Created/Modified

### Python Export Scripts
| File | Purpose |
|------|---------|
| `export_transformer_onnx.py` | Exports Transformer to ONNX with PyTorch dynamo |
| `fix_lightgbm_onnx.py` | Fixes LightGBM ONNX for MT5 (removes ZipMap, fixes batch) |
| `export_scaler_json.py` | Exports MinMaxScaler parameters to JSON |
| `validate_mt5_pipeline.py` | Comprehensive 2-model pipeline validation |
| `parity_test_generator.py` | Generates test cases for Python-MQL5 parity |

### MT5 Model Files (MQL5/Files/NeuralBot/)
| File | Shape | Description |
|------|-------|-------------|
| `transformer.onnx` | [1, 30, 130] → [1, 1] | Transformer model for multi_tf_signal |
| `hybrid_lightgbm.onnx` | [1, 27] → [1] + [1, 3] | LightGBM classifier (label + probs) |
| `scaler_params.json` | 130 features | MinMaxScaler parameters |
| `validation_results.json` | - | Python validation test results |

### MQL5 Include Files (mt5_expert_advisor/Include/)
| File | Classes | Purpose |
|------|---------|---------|
| `FeatureCalculator.mqh` | CMinMaxScaler, CFeatureCalculator | Feature calculation matching Python |
| `SequenceBuffer.mqh` | CSequenceBuffer, CTransformerInference, CLightGBMInference | Sequence buffering and ONNX wrappers |
| `SafetyGuards.mqh` | CSafetyGuard, CFallbackHandler, CPerformanceMonitor | Error handling and safety |

### MQL5 Expert Advisors
| File | Description |
|------|-------------|
| `XAUUSD_NeuralBot_Hybrid.mq5` | New hybrid EA with 2-model pipeline |
| `XAUUSD_NeuralBot_M1.mq5` | Original single-model EA (unchanged) |

### MQL5 Scripts
| File | Purpose |
|------|---------|
| `Scripts/TestFeatureCalculator.mq5` | Feature and ONNX validation tests |
| `Files/NeuralBot/parity_tests/ParityValidator.mq5` | Parity validation helper |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MT5 Hybrid Neural Bot                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │ CFeatureCalculator│    │        CSequenceBuffer           │   │
│  │ - 130 TF features │───▶│  Rolling buffer [30, 130]        │   │
│  │ - 26 LGB features │    │  - Scaled with MinMaxScaler      │   │
│  └──────────────────┘    └───────────────┬──────────────────┘   │
│                                          │                       │
│                                          ▼                       │
│                          ┌──────────────────────────────────┐   │
│                          │     CTransformerInference        │   │
│                          │  Input: [1, 30, 130] float32     │   │
│                          │  Output: multi_tf_signal [1, 1]  │   │
│                          └───────────────┬──────────────────┘   │
│                                          │                       │
│                                          ▼                       │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │ 26 LGB Features  │───▶│    Combine 27 Features           │   │
│  │ (raw, unscaled)  │    │  [multi_tf_signal] + [26 LGB]    │   │
│  └──────────────────┘    └───────────────┬──────────────────┘   │
│                                          │                       │
│                                          ▼                       │
│                          ┌──────────────────────────────────┐   │
│                          │     CLightGBMInference           │   │
│                          │  Input: [1, 27] float32          │   │
│                          │  Output: label [1] + probs [1,3] │   │
│                          └───────────────┬──────────────────┘   │
│                                          │                       │
│                                          ▼                       │
│                          ┌──────────────────────────────────┐   │
│                          │        Trade Decision            │   │
│                          │  HOLD (0) / BUY (1) / SELL (2)   │   │
│                          │  + Confidence >= Threshold       │   │
│                          └──────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Safety Layer: CSafetyGuard, CFallbackHandler            │   │
│  │  - Feature validation, probability checks                 │   │
│  │  - Circuit breaker, fallback signals                      │   │
│  │  - Spread/volume guards                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Feature Order

### 27 LightGBM Features (hybrid_lightgbm.onnx)
```
Index 0:  multi_tf_signal (from Transformer)
Index 1:  body
Index 2:  body_abs
Index 3:  candle_range
Index 4:  close_position
Index 5:  return_1
Index 6:  return_5
Index 7:  return_15
Index 8:  return_60
Index 9:  tr
Index 10: atr_14
Index 11: rsi_14
Index 12: ema_10
Index 13: ema_20
Index 14: ema_50
Index 15: hour_sin
Index 16: hour_cos
Index 17: M5_trend
Index 18: M5_position
Index 19: M15_trend
Index 20: M15_position
Index 21: H1_trend
Index 22: H1_position
Index 23: H4_trend
Index 24: H4_position
Index 25: D1_trend
Index 26: D1_position
```

## Validation Results

### Python Validation (validate_mt5_pipeline.py)
- ✅ 100/100 random inference tests passed
- ✅ 5/5 edge cases passed (zeros, ones, large, negative, NaN)
- ✅ Transformer output range: [-0.90, 0.71]
- ✅ Average confidence: 0.74

### Model Specifications
| Model | Input | Output | Format |
|-------|-------|--------|--------|
| Transformer | [1, 30, 130] float32 | [1, 1] float32 | ONNX opset 18 |
| LightGBM | [1, 27] float32 | label [1] int64, probs [1, 3] float32 | ONNX opset 15 |
| Scaler | 130 features | scale[], min[] | JSON |

## Git Commits
1. `Phase 3 Task 1: Export Transformer to ONNX`
2. `Phase 3 Task 2: Export LightGBM ONNX for MT5`
3. `Phase 3 Task 3: Export scaler to JSON for MT5`
4. `Phase 3 Task 4: Python validation script for 2-model pipeline`
5. `Phase 3 Tasks 5-8: Complete MT5 MQL5 implementation`
6. `Phase 3 Tasks 9-10: Parity tests and safety guards`

## Next Steps (Phase 4+)

1. **MT5 Testing**
   - Compile MQL5 files in MetaEditor
   - Run TestFeatureCalculator.mq5 script
   - Validate feature parity with Python

2. **Strategy Tester**
   - Backtest XAUUSD_NeuralBot_Hybrid.mq5
   - Compare performance vs original EA
   - Tune confidence threshold

3. **Live Testing**
   - Demo account testing
   - Monitor inference times
   - Verify risk management

4. **Optimization**
   - Feature importance analysis
   - Threshold optimization
   - Ensemble refinements

## Usage

### To compile and test in MT5:
1. Copy `mt5_expert_advisor/` contents to `[MT5 Data Folder]/MQL5/`
2. Copy model files to `MQL5/Files/NeuralBot/`
3. Compile `XAUUSD_NeuralBot_Hybrid.mq5` in MetaEditor
4. Run `Scripts/TestFeatureCalculator.mq5` to validate
5. Attach EA to XAUUSD M1 chart

### Python validation:
```bash
python validate_mt5_pipeline.py
```

### Parity testing:
```bash
python parity_test_generator.py
```

---
*Generated: Phase 3 Complete*
*Status: All 10 tasks implemented and committed*
