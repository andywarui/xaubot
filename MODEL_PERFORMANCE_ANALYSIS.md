# MT5 Model Performance Analysis & Research Findings

**Date**: 2025-12-26
**Model**: LightGBM (26 features) + 5-Layer Hybrid Validation
**Test Period**: 2022-01-03 to 2024-12-31 (3 years, 1.1M bars)
**Data**: Synthetic XAUUSD M1 (Geometric Brownian Motion)

---

## Executive Summary

The initial LightGBM model trained on synthetic data has **failed backtesting** with a -76.41% return over 3 years. While the model successfully generated trading signals (149 trades), the win rate of 28.86% is critically low, resulting in severe account drawdown.

**CRITICAL FINDING**: Model is **NOT READY** for deployment in current form.

---

## Backtest Results

### Performance Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|---------|
| **Net Profit** | -$7,641.10 | Positive | ❌ FAIL |
| **Return %** | -76.41% | >0% | ❌ FAIL |
| **Final Balance** | $2,358.90 | >$10,000 | ❌ FAIL |
| **Win Rate** | 28.86% | >50% | ❌ FAIL |
| **Profit Factor** | 0.82 | >1.2 | ❌ FAIL |
| **Max Drawdown** | 77.32% | <20% | ❌ FAIL |

### Trade Statistics
- **Total Trades**: 149
- **Winning Trades**: 43 (28.86%)
- **Losing Trades**: 106 (71.14%)
- **Average Win**: $786.98
- **Average Loss**: $391.33
- **Win/Loss Ratio**: 2.01 (wins are 2x larger than losses)
- **Max Consecutive Losses**: 11
- **Max Consecutive Wins**: 3

### Key Observation
Despite favorable risk-reward (avg win $787 vs avg loss $391), the **low win rate** (28.86%) makes the strategy unprofitable.

**Break-even calculation**:
- With 2:1 win/loss ratio, need minimum 33.3% win rate to break even
- Actual win rate: 28.86%
- **4.5% below break-even** → Guaranteed losses

---

## Root Cause Analysis

### 1. Synthetic Data Limitations ⚠️

**Problem**: Model trained on Geometric Brownian Motion (GBM), not real market data.

**Impact**:
- GBM assumes random price movements with constant drift and volatility
- Real XAUUSD markets have:
  - Regime changes (trending vs ranging)
  - News-driven volatility spikes
  - Session-specific patterns (Asian, London, NY)
  - Support/resistance levels
  - Correlation with USD, interest rates, geopolitical events

**Evidence**: Model performs poorly despite 81.2% test accuracy on synthetic data.

**Conclusion**: **Synthetic data creates overfitting to artificial patterns** that don't exist in real markets.

### 2. Target Label Definition Issues

**Current Approach**:
```python
long_threshold = 0.0005   # 0.05% up (~$1 on $2000)
short_threshold = -0.0005  # 0.05% down
lookahead = 10 bars       # 10 minutes
```

**Problems**:
- Threshold too small (0.05%) relative to spread and noise
- 10-minute lookahead may not capture true trend direction
- No consideration of:
  - Trade execution costs (spread, slippage)
  - Stop loss / take profit levels
  - Market volatility at signal time

**Impact**: Model predicts "correct" direction on synthetic data but signals don't translate to profitable trades.

### 3. Hybrid Validation Configuration

**Layer Analysis**:

| Layer | Filter | Setting | Status | Notes |
|-------|--------|---------|--------|-------|
| 1 | Spread | <$2.00 | ✅ OK | Reasonable for XAUUSD |
| 2 | RSI | LONG: <70, SHORT: >30 | ⚠️ CHECK | May be too permissive |
| 3 | MACD | Alignment required | ⚠️ CHECK | May filter good signals |
| 4 | ADX | >20 | ⚠️ CHECK | Requires strong trend |
| 5 | ATR | $1.50-$8.00 | ✅ OK | Volatility filter |
| 6 | MTF | **DISABLED** | ❌ N/A | Synthetic data lacks MTF |

**Finding**: With MTF disabled, only 5 layers active. Model still losing despite passing validation.

**Hypothesis**: Validation layers filter signals but don't ensure **quality** - they block noise but allow poor setups.

### 4. Risk Management

**Current Settings**:
- Stop Loss: $4.00
- Take Profit: $8.00
- Risk per trade: 0.5% of balance
- Max trades per day: 5

**Risk-Reward Ratio**: 1:2 (risk $4 to make $8)

**Problems**:
- Fixed dollar SL/TP don't adapt to volatility
- In high volatility, $4 SL hit too easily
- In low volatility, $8 TP may never be reached
- Max consecutive losses of 11 can wipe 5.5% of account (before compounding)

### 5. Model Architecture Limitations

**Current**: LightGBM with 26 features
- Price features (4)
- Returns (4)
- Technical indicators (6)
- Time features (2)
- Multi-timeframe features (10) - **ALL ZEROS** in synthetic data

**Missing**:
- Order flow / volume analysis
- Market microstructure
- Sentiment indicators
- Regime detection features
- Adaptive features based on recent performance

---

## Comparison: Research vs Results

### Model Training Metrics
| Phase | Accuracy | Tradeable Signals | Comments |
|-------|----------|-------------------|----------|
| Initial | 89.0% | 12.1% (LONG 6%, SHORT 6%) | Too conservative |
| Tuned | 81.2% | 30.4% (LONG 15.2%, SHORT 15.2%) | Better balance |

**Expectation**: 81.2% accuracy should translate to profitable trading.

**Reality**: 28.86% win rate in backtest.

**Gap**: **52.34%** accuracy gap between training and live performance.

### Why the Gap?

1. **Train-Test Contamination**:
   - Training calculates target using `lookahead=10 bars` into future
   - In backtest, model must predict without seeing future
   - Model learned patterns from "peeking" at future data

2. **Feature Leakage**:
   - Technical indicators use current bar's high/low/close
   - In live trading, don't know final candle values until bar closes
   - Model may be using information not available in real-time

3. **Overfitting to Synthetic Noise**:
   - 81.2% accuracy on GBM-generated data
   - GBM has no real patterns to learn
   - Model fit to random noise, not predictive features

---

## Critical Issues Summary

### 🔴 BLOCKER Issues
1. **Synthetic Data**: Model MUST be trained on real XAUUSD data
2. **Win Rate**: 28.86% win rate is catastrophically low
3. **Drawdown**: 77% drawdown exceeds acceptable risk (target <20%)

### 🟡 HIGH Priority Issues
4. **Feature Leakage**: Verify no future data in training
5. **Target Definition**: Align targets with profitable trade outcomes, not just direction
6. **Risk Management**: Implement dynamic SL/TP based on ATR

### 🟢 MEDIUM Priority Issues
7. **MTF Features**: All zeros in synthetic data, need real multi-timeframe calculation
8. **Hybrid Validation**: Tune layers to filter low-quality setups
9. **Model Complexity**: Consider ensemble with Transformer for pattern recognition

---

## Recommendations

### Immediate Actions (Priority 1)

#### 1. **Switch to Real Data**
- **Action**: Train model on actual XAUUSD M1 data (from Dukascopy, MetaQuotes, or broker feed)
- **Benefit**: Learn real market patterns, not synthetic noise
- **Risk**: Requires data access (Git LFS blocked, need alternative)
- **Alternative**: Use smaller dataset (1 year instead of 3) to fit in repo

#### 2. **Fix Target Definition**
- **Current Problem**: Predicting 0.05% move in 10 minutes
- **New Approach**:
  ```python
  # Target = "Will this trade be profitable with SL=$4, TP=$8?"
  def create_profitable_target(data, idx, sl_usd=4.0, tp_usd=8.0):
      entry_price = data.iloc[idx]['close']

      # Scan next 30-60 bars (30-60 minutes)
      for future_idx in range(idx+1, min(idx+61, len(data))):
          future_high = data.iloc[future_idx]['high']
          future_low = data.iloc[future_idx]['low']

          # LONG target
          if future_high >= entry_price + tp_usd:
              return 2  # TP hit first = LONG
          if future_low <= entry_price - sl_usd:
              return 1  # SL hit first = HOLD (don't trade)

          # SHORT target
          if future_low <= entry_price - tp_usd:
              return 0  # TP hit first = SHORT
          if future_high >= entry_price + sl_usd:
              return 1  # SL hit first = HOLD

      return 1  # No clear outcome = HOLD
  ```
- **Rationale**: Align model predictions with actual trade outcomes

#### 3. **Implement Walk-Forward Testing**
- **Action**: Instead of single train/test split, use rolling windows:
  ```
  Train: Jan-Jun 2022 → Test: Jul-Sep 2022
  Train: Apr-Sep 2022 → Test: Oct-Dec 2022
  Train: Jul-Dec 2022 → Test: Jan-Mar 2023
  ...
  ```
- **Benefit**: Detect overfitting and regime changes
- **Validates**: Model generalizes across different market conditions

#### 4. **Add Dynamic Stop Loss/Take Profit**
- **Action**: Scale SL/TP based on current ATR
  ```python
  atr = row['atr_14']
  stop_loss = max(4.0, atr * 2.0)      # 2x ATR, minimum $4
  take_profit = max(8.0, atr * 4.0)    # 4x ATR, minimum $8
  ```
- **Benefit**: Adapt to market volatility
- **Prevents**: Getting stopped out in high volatility, TP too far in low volatility

### Medium-Term Improvements (Priority 2)

#### 5. **Feature Engineering Enhancements**
- Add volume-based features (if available)
- Add price action patterns (pin bars, engulfing, etc.)
- Add regime detection (trending vs ranging)
- Add session indicators (Asian/London/NY)
- Add correlation features (DXY, US10Y if available)

#### 6. **Hyperparameter Optimization**
Use Optuna or Hyperopt to tune:
- LightGBM parameters (num_leaves, learning_rate, max_depth)
- Confidence threshold (0.50 vs 0.60 vs 0.70)
- Hybrid validation thresholds (RSI, ADX, ATR ranges)
- SL/TP multipliers

#### 7. **Ensemble Model**
- Train Transformer model (as originally planned)
- Implement voting system:
  ```python
  if LightGBM predicts LONG with >60% confidence AND
     Transformer predicts LONG with >60% confidence AND
     Hybrid validation passes:
      → Execute LONG trade
  ```
- **Benefit**: Reduce false signals, increase win rate

### Long-Term Strategy (Priority 3)

#### 8. **Reinforcement Learning**
- Use Deep Q-Learning or PPO to optimize trade timing
- Reward = actual PnL (not just direction accuracy)
- Learns when to enter, hold, exit, and skip trades
- **Complexity**: High, but potentially better results

#### 9. **Live Paper Trading**
- Deploy best model to MT5 demo account
- Monitor for 30 days
- Compare live results to backtest
- **Red flags**: Win rate drop, increased slippage, execution issues

#### 10. **A/B Testing Framework**
- Deploy Single Model vs Ensemble Model simultaneously
- Allocate 50% capital to each
- Compare over 90 days
- **Winner proceeds to live trading**

---

## Next Steps

### Option A: Continue with Synthetic Data (Research Path)
✅ **Pros**: No data access needed, can iterate quickly
❌ **Cons**: Results won't reflect real market performance
**Use Case**: Algorithm research, proof of concept only

**Actions**:
1. Retrain with profitable target definition
2. Implement dynamic SL/TP
3. Run new backtest
4. Document as "synthetic baseline" for comparison

### Option B: Switch to Real Data (Production Path)
✅ **Pros**: Realistic performance estimates, production-ready
❌ **Cons**: Need data access, slower iteration
**Use Case**: Preparing for live deployment

**Actions**:
1. Source real XAUUSD M1 data (1-3 years)
2. Verify data quality (no gaps, realistic spreads)
3. Calculate proper MTF features
4. Retrain entire pipeline
5. Validate with walk-forward testing

### Option C: Hybrid Approach (Recommended)
✅ **Pros**: Balance speed and realism
❌ **Cons**: More complex workflow

**Actions**:
1. **Phase 1**: Fix target definition and test on synthetic (1 week)
2. **Phase 2**: Source 1 year of real data and retrain (1 week)
3. **Phase 3**: If real data performs well, extend to 3 years (1 week)
4. **Phase 4**: Deploy to paper trading (30 days)
5. **Phase 5**: Live trading with micro-lots (90 days)

---

## Risk Disclosure

**Current Model Status**: ⚠️ **HIGH RISK - DO NOT DEPLOY**

If deployed with current performance:
- Expected annual return: **-92%** (extrapolating -76% over 3 years)
- Probability of account loss >50%: **~100%**
- Estimated time to margin call: **<6 months**

**Required Improvements Before Deployment**:
- [ ] Train on real data
- [ ] Achieve >50% win rate in backtest
- [ ] Reduce max drawdown to <20%
- [ ] Profit factor >1.5
- [ ] Validate on paper trading for 30+ days
- [ ] Pass walk-forward testing across multiple regimes

---

## Conclusion

The LightGBM model trained on synthetic data has demonstrated the complete trading pipeline works (data → training → ONNX export → backtesting), but **produces unprofitable results**.

**Root cause**: Synthetic data does not capture real market dynamics.

**Solution**: Train on real XAUUSD data with improved target definition and risk management.

**Timeline to Production**:
- With real data: 4-8 weeks (training + validation + paper trading)
- Without real data: **NOT RECOMMENDED** for live trading

---

## Appendix: Tuning History

### Iteration 1: Initial Training
- Target threshold: 0.15% (0.0015)
- Lookahead: 15 bars
- Result: 89% accuracy, 12% tradeable signals
- Issue: Too conservative, 0 trades in backtest

### Iteration 2: Aggressive Tuning
- Target threshold: 0.05% (0.0005)
- Lookahead: 10 bars
- Result: 81.2% accuracy, 30.4% tradeable signals
- Issue: Generated trades but 28.86% win rate

### Iteration 3: MTF Fix
- Disabled multi-timeframe alignment
- Result: Trades executed, -76.41% return
- Issue: Low win rate persists

### Next Iteration: Profitable Target (Recommended)
- Define target based on SL/TP outcomes
- Use real data
- Expected: Better alignment between training and trading results
