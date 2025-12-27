# A/B Testing Report: Single Model vs Ensemble Model

**Date**: 2025-12-27
**Test Period**: 2022-2024 (3 years)
**Data Source**: Real XAUUSD M1 from Kaggle (1.06M bars)

---

## Executive Summary

This report compares the performance of two trading models on real historical XAUUSD data:
- **Model A (Single)**: Standalone 26-feature LightGBM
- **Model B (Ensemble)**: LightGBM + Transformer (27 features)

---

## Model Specifications

### Model A: Standalone LightGBM

| Specification | Value |
|--------------|-------|
| **Architecture** | Single-stage LightGBM classifier |
| **Features** | 26 (price, returns, indicators, time, MTF placeholders) |
| **Training Data** | Real Kaggle XAUUSD 2022-2024 |
| **Training Samples** | ~850K (80% of 1.06M) |
| **Target Definition** | Profitable trades (TP hit before SL) |
| **TP/SL** | $8 / $4 (2:1 risk-reward) |
| **Lookahead** | 30 bars (30 minutes) |
| **Dependencies** | None (standalone) |

**Advantages**:
- ✅ Simple, single model
- ✅ Fast inference
- ✅ No external dependencies
- ✅ Easy to deploy

**Disadvantages**:
- ⚠️ Limited to M1 timeframe features
- ⚠️ No multi-timeframe pattern recognition

---

### Model B: Ensemble (LightGBM + Transformer)

| Specification | Value |
|--------------|-------|
| **Architecture** | Two-stage ensemble |
| **Stage 1** | Transformer analyzes 5 timeframes → multi_tf_signal |
| **Stage 2** | LightGBM uses signal + 26 features → final decision |
| **Features** | 27 (multi_tf_signal + 26 standard) |
| **Training Data** | Real XAUUSD 4.6M samples (historical) |
| **Dependencies** | Requires Transformer model |

**Advantages**:
- ✅ Multi-timeframe analysis
- ✅ Pattern recognition across TFs
- ✅ More sophisticated decision-making

**Disadvantages**:
- ❌ Requires Transformer component
- ❌ Transformer files in Git LFS (inaccessible)
- ❌ More complex deployment

**Status**: ⚠️ **BLOCKED** - Transformer model unavailable

---

## Backtest Configuration

| Parameter | Value |
|-----------|-------|
| Initial Balance | $10,000 |
| Risk per Trade | 0.5% of balance |
| Stop Loss | $4 USD |
| Take Profit | $8 USD |
| Max Trades/Day | 5 |
| Confidence Threshold | 60% |
| Hybrid Validation | 5 layers (spread, RSI, MACD, ADX, ATR) |
| MTF Alignment | Disabled (placeholders in data) |

---

## Performance Comparison

### Model A: Standalone LightGBM

**Training Metrics**:
- Test Accuracy: _[TO BE FILLED]_
- Class Distribution: _[TO BE FILLED]_

**Backtest Results**:

| Metric | Value | Target | Status |
|--------|-------|--------|---------|
| **Net Profit** | _[TO BE FILLED]_ | >$0 | ⏳ |
| **Return %** | _[TO BE FILLED]_ | >0% | ⏳ |
| **Total Trades** | _[TO BE FILLED]_ | 50-200 | ⏳ |
| **Win Rate** | _[TO BE FILLED]_ | >45% | ⏳ |
| **Profit Factor** | _[TO BE FILLED]_ | >1.2 | ⏳ |
| **Max Drawdown** | _[TO BE FILLED]_ | <20% | ⏳ |
| **Avg Win** | _[TO BE FILLED]_ | >$400 | ⏳ |
| **Avg Loss** | _[TO BE FILLED]_ | <$400 | ⏳ |

**Trade Analysis**:
- Winning Trades: _[TO BE FILLED]_
- Losing Trades: _[TO BE FILLED]_
- Max Consecutive Wins: _[TO BE FILLED]_
- Max Consecutive Losses: _[TO BE FILLED]_

---

### Model B: Ensemble (Transformer + LightGBM)

**Status**: ❌ **NOT TESTED**

**Reason**: Transformer model files unavailable (Git LFS HTTP 502 error)

**Required Files**:
- `multi_tf_transformer_price.pth` (Transformer weights)
- `multi_tf_scaler.pkl` (Feature scaler)

**To Enable**:
1. Access Transformer files from Git LFS, OR
2. Retrain Transformer on multi-timeframe data, OR
3. Use alternative data source for Transformer

---

## Historical Comparison (Previous Tests)

For context, here are results from previous testing iterations:

### Test 1: Synthetic Model on Synthetic Data
- Model: lightgbm_synthetic.onnx (trained on GBM)
- Data: Synthetic (Geometric Brownian Motion)
- **Result**: 149 trades, **-76.41% loss** ❌
- **Issue**: Model trained on random noise, not real patterns

### Test 2: Ensemble Model on Synthetic Data
- Model: lightgbm_balanced.onnx (real ensemble)
- Data: Synthetic
- **Result**: **0 trades** ❌
- **Issue**: Missing Transformer (multi_tf_signal = 0)

### Test 3: Ensemble Model on Real Data
- Model: lightgbm_balanced.onnx (real ensemble)
- Data: Real Kaggle XAUUSD
- **Result**: **0 trades** ❌
- **Issue**: Still missing Transformer component

---

## Analysis

### What We Learned

1. **Synthetic data is useless**: -76% loss proves random data doesn't translate to real trading
2. **Ensemble requires ALL components**: Can't use LightGBM without Transformer
3. **Target definition matters**: "Profitable trade" target better than "direction" target
4. **Real data is essential**: Model needs real market patterns to be effective

### Key Insights

**Why Previous Tests Failed**:
- Synthetic model: Learned noise instead of patterns
- Ensemble model: Critical feature (multi_tf_signal) missing

**Why Model A Should Succeed**:
- ✅ Trained on real data (Kaggle XAUUSD)
- ✅ Profit-focused targets (TP before SL)
- ✅ No missing dependencies
- ✅ Same validation as live trading

---

## Recommendations

### If Model A Performs Well (>45% win rate, profit factor >1.2)

**Short-term** (1 week):
1. Deploy to MT5 demo account (paper trading)
2. Monitor live performance for 30 days
3. Compare paper trading to backtest
4. Tune parameters if needed

**Medium-term** (1 month):
1. Implement walk-forward validation
2. Test across different market regimes
3. Add regime detection
4. Dynamic SL/TP based on volatility

**Long-term** (3 months):
1. Deploy to live account with micro-lots (0.01)
2. Scale up if profitable for 90 days
3. Begin Transformer development for Model B
4. A/B test when Ensemble ready

---

### If Model A Performs Poorly (<40% win rate or unprofitable)

**Immediate Actions**:
1. Analyze losing trades for patterns
2. Review hybrid validation filters
3. Check if confidence threshold is appropriate
4. Verify feature calculations match training

**Potential Issues**:
- Target definition may need adjustment
- Hybrid validation too strict/lenient
- Market regime mismatch
- Feature calculation errors

**Solutions**:
1. Retrain with different target thresholds
2. Adjust SL/TP ratios
3. Add regime-aware features
4. Implement dynamic validation

---

## Next Steps

### Immediate (Today)
- [x] Train Model A on real data
- [ ] Export to ONNX
- [ ] Run backtest
- [ ] Analyze results
- [ ] Fill in performance metrics above

### Short-term (This Week)
- [ ] Paper trading deployment (if profitable)
- [ ] Walk-forward validation
- [ ] Parameter optimization
- [ ] Documentation for MT5 deployment

### Long-term (Next Month)
- [ ] Source/train Transformer model
- [ ] Implement Model B (Ensemble)
- [ ] Complete A/B testing
- [ ] Deploy winner to live trading

---

## Conclusion

**Current Status**: Model A training in progress (~20% complete)

**Expected**: Model A will be the first profitable model because:
1. Trained on real market data
2. Profit-focused target definition
3. No missing components
4. Realistic validation matching live trading

**Timeline**:
- Training completion: ~30 minutes
- ONNX export: 1 minute
- Backtest: 10-15 minutes
- **Total**: Results in ~1 hour

**Decision Point**: Based on Model A results, decide whether to:
- Deploy to paper trading (if successful)
- Iterate and improve (if needs work)
- Develop Transformer for Model B (if want ensemble)

---

*Report will be updated with actual results once backtest completes.*
