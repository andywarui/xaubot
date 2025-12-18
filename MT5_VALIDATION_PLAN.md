# MT5 Strategy Tester Validation Plan

## Pre-Test Checklist

### 1. Files Ready
- [x] ONNX model: `mt5_expert_advisor/Files/lightgbm_xauusd.onnx` (222 KB)
- [x] EA code: `mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5`
- [x] Feature order: `config/features_order.json` (26 features)
- [x] Model metadata: `config/model_meta.json`

### 2. Copy Files to MT5
```
Source: mt5_expert_advisor/Files/lightgbm_xauusd.onnx
Target: C:\Users\[USER]\AppData\Roaming\MetaQuotes\Terminal\[BROKER]\MQL5\Files\lightgbm_xauusd.onnx

Source: mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5
Target: C:\Users\[USER]\AppData\Roaming\MetaQuotes\Terminal\[BROKER]\MQL5\Experts\XAUUSD_NeuralBot_M1.mq5
```

### 3. Compile EA
1. Open MetaEditor (F4 in MT5)
2. Open `XAUUSD_NeuralBot_M1.mq5`
3. Press F7 to compile
4. Check for **0 errors**

---

## Strategy Tester Configuration

### Test 1: Match Python "REALISTIC" Scenario

**EA Parameters:**
- `RiskPercent`: 2.0 (2%)
- `ConfidenceThreshold`: 0.60
- `MaxTradesPerDay`: 100 (no limit for test)
- `MaxDailyLoss`: 100.0 (no limit for test)
- `EnableFeatureLog`: false
- `EnablePredictionLog`: false

**MT5 Tester Settings:**
- Symbol: XAUUSD
- Period: M5 (chart timeframe)
- Date Range: Last 12 months (match OOS period)
  - From: 2024-10-01
  - To: 2025-10-01
- Model: **Every tick based on real ticks** (most accurate)
- Initial Deposit: $50.00
- Leverage: 1:100 (or broker default)
- Optimization: Disabled

**Expected Results (from Python OOS):**
- Trades: ~15,000
- Win Rate: ~74%
- Profit Factor: ~4.6
- Net Profit: ~$16,400
- Max Drawdown: <10%

---

### Test 2: Feature Parity Check

**EA Parameters:**
- `EnableFeatureLog`: **true**
- `FeatureLogFile`: "feature_parity_check.csv"
- `EnablePredictionLog`: **true**
- `PredictionLogFile`: "prediction_parity_check.csv"
- Date Range: Small window (1 week)

**After Test:**
1. Copy CSVs from `MQL5/Files/` to repo
2. Run parity check:
   ```bash
   python python_training/compare_features_mt5.py --log mt5_expert_advisor/feature_parity_check.csv
   ```
3. **Expected**: All features within ±2% deviation
4. **If >2% deviation**: Investigate indicator calculations

---

### Test 3: Live-Ready Configuration

**EA Parameters:**
- `RiskPercent`: 0.5 (0.5% - live setting)
- `ConfidenceThreshold`: 0.70 (conservative)
- `MaxTradesPerDay`: 5
- `MaxDailyLoss`: 3.0 (3% daily limit)
- Date Range: Last 12 months

**Expected Results (from Python OOS):**
- Trades: ~8,300
- Win Rate: ~80%
- Profit Factor: ~3.7
- Net Profit: ~$1,700

---

## Validation Criteria

### ✅ Pass Criteria
1. **Win Rate**: Within ±5% of Python backtest
2. **Profit Factor**: Within ±20% of Python backtest
3. **Trade Count**: Within ±10% of Python backtest
4. **Max Drawdown**: <15%
5. **Feature Parity**: All features within ±2%
6. **ONNX Parity**: Predictions match Python ONNX inference

### ⚠️ Warning Signs
- Win rate <65%
- Profit factor <2.0
- Max drawdown >20%
- Trade count vastly different (>20% off)
- Frequent ONNX errors in logs

### ❌ Fail Criteria
- Win rate <60%
- Profit factor <1.5
- Max drawdown >30%
- ONNX model fails to load
- Feature deviation >5%

---

## Post-Test Analysis

### 1. Compare MT5 vs Python
Create comparison table:

| Metric | Python OOS | MT5 Tester | Difference |
|--------|-----------|-----------|------------|
| Trades | 15,324 | ? | ? |
| Win Rate | 74.1% | ? | ? |
| Profit Factor | 4.60 | ? | ? |
| Net Profit | $16,436 | ? | ? |
| Max DD | ? | ? | ? |

### 2. Review MT5 Logs
Check for:
- [ ] No ONNX load errors
- [ ] All trades within session hours (12:00-17:00 UTC)
- [ ] No invalid feature calculations
- [ ] Proper stop loss / take profit placement

### 3. Equity Curve Analysis
- [ ] Compare MT5 equity curve to Python
- [ ] Check for unusual drawdown spikes
- [ ] Verify smooth growth (not erratic)

---

## Next Steps After Validation

### If Tests Pass (✅):
1. **Demo Account** (30 days minimum)
   - Use Live-Ready config (0.5% risk)
   - Monitor daily for 1 month
   - Compare real execution to Strategy Tester
   
2. **Micro Live** (if demo successful)
   - $500-1000 capital
   - 0.5% risk
   - Track for 3 months
   
3. **Scale Up** (if consistently profitable)
   - Increase capital gradually
   - Max 1% risk even at scale

### If Tests Fail (❌):
1. **Feature Parity Failed**:
   - Debug indicator calculations
   - Check session filter logic
   - Verify higher timeframe context

2. **Performance Degraded**:
   - Review MT5 spread settings
   - Check for slippage modeling
   - Verify execution logic

3. **ONNX Issues**:
   - Re-export ONNX model
   - Verify input/output shapes
   - Test ONNX in Python first

---

## Monitoring Setup (Live Trading)

### Real-Time Monitoring
- [ ] EA status indicator
- [ ] Current drawdown %
- [ ] Trades today count
- [ ] Daily P&L vs limit

### Alert Conditions
- Daily loss >2%
- Max drawdown >10%
- 3 consecutive losses
- Spread >4 pips at trade time

### Weekly Report Template
```
Week: [Date Range]
Trades: X
Win Rate: X%
Net P&L: $X
Max DD: X%
Status: ✅ / ⚠️ / ❌
```

---

**Created**: December 12, 2025
**Last Updated**: December 12, 2025
**Next Review**: After MT5 Strategy Tester validation
