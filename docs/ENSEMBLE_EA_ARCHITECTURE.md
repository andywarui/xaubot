# Ensemble EA Architecture

## Overview

The `XAUUSD_NeuralBot_Ensemble.mq5` Expert Advisor implements a sophisticated ensemble learning system that combines **LightGBM** and **Transformer** models with **Hybrid Validation** filters.

## Model Architecture

### 1. LightGBM Model (Primary Classifier)

**Input**: 26 features (single M1 bar)
**Output**: 3-class probabilities [SHORT, HOLD, LONG]

**Feature Set**:
- Price features (4): body, body_abs, candle_range, close_position
- Returns (4): 1, 5, 15, 60-minute returns
- Technical indicators (6): TR, ATR, RSI, EMA10, EMA20, EMA50
- Time features (2): hour_sin, hour_cos
- Multi-timeframe context (10): M5, M15, H1, H4, D1 trend and position

**Model File**: `lightgbm_xauusd.onnx`
**Shape**: [1, 26] → [1] label + [1, 3] probabilities

### 2. Transformer Model (Temporal Predictor)

**Input**: 30-bar sequence × 130 features
**Output**: Single price change prediction (regression)

**Feature Set**: 26 features (same as LightGBM) × 5 timeframes (M1, M5, M15, H1, H4) = 130 features

**Architecture**:
- Multi-headed self-attention (4 heads)
- Positional encoding for temporal awareness
- Sequence length: 30 bars (warmup required)

**Model File**: `transformer.onnx`
**Scaler File**: `transformer_scaler_params.json` (MinMaxScaler parameters)
**Shape**: [1, 30, 130] → [1, 1] price change

### 3. Ensemble Voting Logic

```mql5
// Pseudocode:
lgb_signal = PredictLightGBM(confidence);
trans_pred = PredictTransformer();  // Returns price change

// Convert Transformer regression to classification
if (trans_pred > 0.1%) → trans_signal = LONG
if (trans_pred < -0.1%) → trans_signal = SHORT
else → trans_signal = HOLD

// Agreement requirement
if (lgb_signal == trans_signal AND lgb_confidence >= 60%)
   → Execute signal with ensemble confidence
else
   → HOLD (no agreement)
```

**Key Parameters**:
- `UseEnsemble`: Enable/disable Transformer (fallback to LightGBM-only)
- `EnsembleAgreementThreshold`: 0.60 (min confidence when both agree)
- `TransformerSignalThreshold`: 0.1 (% change threshold for Transformer)
- `AllowLightGBMFallback`: true (use LightGBM if Transformer fails)

## Hybrid Validation (6-Layer Filtering)

Both single and ensemble models use the same validation system:

1. **Spread Filter**: Max $2.00 spread
2. **RSI Filter**: Avoid overbought (>70) / oversold (<30)
3. **MACD Alignment**: Require bullish MACD for LONG, bearish for SHORT
4. **ADX Trend Strength**: Minimum ADX 20 (trending market)
5. **ATR Volatility Filter**: Range $1.50-$8.00 (sweet spot)
6. **MTF EMA Alignment**: Price must align with M15 and H1 EMA20

**Research Basis**: Hybrid validation improved accuracy from 58% → 85% (MQL5 Blog #765167)

## Sequence Buffer Management

The Transformer requires a rolling 30-bar window:

```mql5
// On each new M1 bar:
Calculate130Features(features, shift=1);  // Last closed bar
UpdateSequenceBuffer(features);  // Shift buffer, insert new

// Status tracking:
g_sequence_count++;  // Counts bars collected
if (g_sequence_count >= 30) → g_sequence_ready = true
```

**Warmup**: EA needs 30 M1 bars (30 minutes) before Transformer predictions start

## File Structure

```
mt5_expert_advisor/
├── XAUUSD_NeuralBot_Single.mq5      # Single model (LightGBM + Hybrid)
├── XAUUSD_NeuralBot_Ensemble.mq5    # Ensemble (LightGBM + Transformer + Hybrid)
└── [ONNX models to be placed in MQL5/Files/]
    ├── lightgbm_xauusd.onnx
    ├── transformer.onnx
    └── transformer_scaler_params.json
```

## A/B Testing Strategy

### Phase 1: Export Transformer Model

```bash
cd python_training
python export_transformer_onnx.py
# Produces:
#   - transformer.onnx
#   - transformer_scaler_params.json
#   - transformer_config.json
```

Copy files to MT5 terminal:
- `%APPDATA%\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Files\`

### Phase 2: Backtest (3 Years Historical Data)

**Single Model Test**:
- EA: `XAUUSD_NeuralBot_Single.mq5`
- Period: 2022-01-01 to 2025-01-01
- Symbol: XAUUSD
- Timeframe: M1 (trades on M1 bars)

**Ensemble Model Test**:
- EA: `XAUUSD_NeuralBot_Ensemble.mq5`
- Same period/symbol/timeframe
- Wait for 30-bar warmup before first trade

**Metrics to Compare**:
1. **Win Rate** (% profitable trades)
2. **Profit Factor** (gross profit / gross loss)
3. **Max Drawdown** (%)
4. **Sharpe Ratio** (risk-adjusted returns)
5. **Average Trade Duration**
6. **Trades Per Month** (signal frequency)
7. **Rejection Rate** (ML signals filtered by hybrid validation)

### Phase 3: Paper Trading

Deploy best-performing version to demo account:
- Monitor for 2+ weeks
- Compare live performance vs backtest
- Track model drift using Python monitoring system
- Validate hybrid filter effectiveness

## Configuration Presets

### Conservative (Single Model)

```mql5
// Best for volatile markets, higher selectivity
RiskPercent = 0.3;
ConfidenceThreshold = 0.70;
EnableHybridValidation = true;
RequireMTFAlignment = true;
ADX_MinStrength = 25.0;
```

### Balanced (Ensemble)

```mql5
// Default ensemble configuration
RiskPercent = 0.5;
ConfidenceThreshold = 0.65;
UseEnsemble = true;
EnsembleAgreementThreshold = 0.60;
EnableHybridValidation = true;
ADX_MinStrength = 20.0;
```

### Aggressive (Single Model, Reduced Filters)

```mql5
// Higher frequency, lower selectivity
RiskPercent = 0.8;
ConfidenceThreshold = 0.60;
EnableHybridValidation = true;
RequireMTFAlignment = false;  // Disable MTF filter
ADX_MinStrength = 15.0;
```

## Performance Expectations

Based on research and model characteristics:

| Metric | Single Model | Ensemble Model |
|--------|-------------|----------------|
| Win Rate | 63-68% | 70-75% |
| Profit Factor | 1.4-1.6 | 1.6-1.9 |
| Trades/Month | 40-60 | 25-40 |
| Max Drawdown | 8-12% | 6-10% |
| Sharpe Ratio | 1.2-1.5 | 1.5-1.9 |

**Hypothesis**: Ensemble model should achieve:
- Higher win rate (Transformer adds temporal context)
- Lower trade frequency (stricter agreement requirement)
- Better risk-adjusted returns (reduced false signals)

## Monitoring and Maintenance

Use Python monitoring system (`python_monitoring/`):

```bash
# Real-time performance tracking
python monitor_live_performance.py

# Dashboard
streamlit run dashboard.py
```

**Key Alerts**:
- Model accuracy < 55% (drift detection)
- High rejection rate (filters too strict)
- Unusual signal distribution (e.g., 90%+ HOLD)

**Maintenance Schedule**:
1. **Weekly**: Review dashboard, check accuracy trends
2. **Monthly**: Analyze rejected signals, tune filter parameters
3. **Quarterly**: Retrain models with latest data if accuracy degraded

## Technical Details

### Memory Requirements

- **Single Model**: ~1 MB (LightGBM ONNX)
- **Ensemble Model**: ~25 MB (LightGBM + Transformer ONNX)
- **Sequence Buffer**: 30 bars × 130 features × 4 bytes = ~15.6 KB

### Computational Load

- **Single Model**: ~1-2ms per prediction (LightGBM inference)
- **Ensemble Model**: ~10-15ms per prediction (both models + voting)
- **Feature Calculation**: ~5ms for 26 features, ~20ms for 130 features

### Indicator Handles

Both EAs create persistent indicator handles:
- M1: ATR, RSI, EMA10, EMA20, EMA50
- Multi-TF: EMA20 for M5, M15, H1, H4, D1
- Validation: MACD, ADX (M15 timeframe)

**Note**: Calculate130Features() creates temporary handles (released after use)

## Troubleshooting

### Transformer Model Not Loading

1. Check file exists: `Terminal Data Folder/MQL5/Files/transformer.onnx`
2. Verify ONNX shape: [1, 30, 130] → [1, 1]
3. Enable debug logs: `EnableOnnxDebugLogs = true`
4. Check Experts tab for error messages

### Scaler Not Loading

If `transformer_scaler_params.json` not found:
- EA will use default [0,1] range (degraded performance)
- Warning message in logs
- Consider implementing proper JSON parser for production

### Sequence Buffer Not Ready

- EA needs 30 M1 bars before Transformer activates
- Check logs: "Sequence buffer: Warming up (need 30 bars)"
- LightGBM will be used during warmup period

### Ensemble Always Returns HOLD

Possible causes:
1. Models disagree on signal direction
2. LightGBM confidence < 60%
3. Transformer not ready (< 30 bars collected)
4. Hybrid validation rejecting signals

Check logs for "ENSEMBLE DISAGREE" or "FILTER REJECT" messages

## Code Locations

Key functions in `XAUUSD_NeuralBot_Ensemble.mq5`:

| Function | Lines | Purpose |
|----------|-------|---------|
| Calculate26Features() | 425-553 | LightGBM feature engineering |
| Calculate130Features() | 558-689 | Transformer feature engineering (5 TFs) |
| UpdateSequenceBuffer() | 694-717 | Rolling 30-bar window management |
| PredictLightGBM() | 722-769 | LightGBM ONNX inference |
| PredictTransformer() | 774-824 | Transformer ONNX inference + normalization |
| PredictEnsemble() | 829-873 | Voting logic (agreement requirement) |
| ValidateLongSignal() | 928-1015 | 6-layer hybrid validation (LONG) |
| ValidateShortSignal() | 1020-1107 | 6-layer hybrid validation (SHORT) |
| OnTick() | 1112-1201 | Main trading loop |

## References

- Research Report: `docs/MT5_IMPLEMENTATION_RESEARCH_REPORT.md`
- Deployment Guide: `docs/DEPLOYMENT_GUIDE.md`
- Optimization Guide: `docs/OPTIMIZATION_PARAMETERS.md`
- Transformer Export: `docs/TRANSFORMER_EXPORT_GUIDE.md`
- Python Monitoring: `python_monitoring/README.md`

---

**Status**: ✅ Ensemble EA complete (1,202 lines)
**Next Step**: Export Transformer ONNX model and run 3-year backtest comparison
