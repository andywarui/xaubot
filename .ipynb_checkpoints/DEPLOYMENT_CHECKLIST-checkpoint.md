# MT5 Deployment Checklist (Go/No-Go)

**Last Updated:** December 12, 2025  
**Status:** Pre-Deployment

---

## 🔴 PHASE 0: Pre-Deployment Validation

### ONNX & Feature Parity
- [ ] **ONNX Export Verification**
  - [ ] Export ONNX from final LightGBM model
  - [ ] Verify input shape: [1, 26]
  - [ ] Verify output shape: [1, 3]
  - [ ] Record model hash/checksum
  - [ ] Test Python ONNX inference matches LightGBM predictions

- [ ] **Feature Parity Check**
  - [ ] Copy ONNX model to `MQL5/Files/lightgbm_xauusd.onnx`
  - [ ] Compile EA in MetaEditor (0 errors)
  - [ ] Run EA with `EnableFeatureLog=true` for 1 week
  - [ ] Copy `feature_log.csv` from `MQL5/Files/` to repo
  - [ ] Run: `python python_training/compare_features_mt5.py --log mt5_expert_advisor/feature_log.csv`
  - [ ] **PASS CRITERIA:** All 26 features within ±2% deviation
  - [ ] **IF FAIL:** Debug indicator calculations, verify session filter, check higher TF context

- [ ] **ONNX Inference Parity**
  - [ ] Run EA with `EnablePredictionLog=true`
  - [ ] Compare MT5 probabilities vs Python ONNX on same feature rows
  - [ ] **PASS CRITERIA:** Predictions match within ±0.01 probability
  - [ ] **IF FAIL:** Re-export ONNX, verify feature order, check input normalization

**GO/NO-GO:** ⬜ All parity checks PASS → Proceed to Strategy Tester

---

## 🟡 PHASE 1: MT5 Strategy Tester

### Test 1: Full Period (Realistic Config)
- [ ] **Configuration**
  - Symbol: XAUUSD | Period: M5 | Model: Every tick (real ticks)
  - Date: 2024-10-01 to 2025-10-01 (12 months OOS)
  - Initial Deposit: $50 | Risk: 2% | Confidence: 0.60
  - `MaxTradesPerDay=100` | `MaxDailyLoss=100` (no limits)

- [ ] **Expected Results (Python OOS)**
  - Trades: ~15,000
  - Win Rate: 74.1% ±5%
  - Profit Factor: 4.60 ±20%
  - Net Profit: ~$16,400
  - Max DD: <10%

- [ ] **Actual Results**
  - Trades: _______
  - Win Rate: _______%
  - Profit Factor: _______
  - Net Profit: $_______
  - Max DD: _______%

- [ ] **Validation**
  - [ ] Win Rate within ±5% → ✅ / ❌
  - [ ] Profit Factor within ±20% → ✅ / ❌
  - [ ] Trade Count within ±10% → ✅ / ❌
  - [ ] Max DD <15% → ✅ / ❌
  - [ ] No ONNX errors in logs → ✅ / ❌

### Test 2: Live-Ready Config
- [ ] **Configuration**
  - Same period | Risk: 0.5% | Confidence: 0.70
  - `MaxTradesPerDay=5` | `MaxDailyLoss=3.0`

- [ ] **Expected Results**
  - Trades: ~8,300 | WR: 80.5% | PF: 3.67 | Profit: ~$1,700

- [ ] **Actual Results**
  - Trades: _______ | WR: _______% | PF: _______ | Profit: $_______

- [ ] **Validation**
  - [ ] Results align with Python OOS → ✅ / ❌
  - [ ] Daily loss limits enforced correctly → ✅ / ❌
  - [ ] Max trades per day enforced → ✅ / ❌

**GO/NO-GO:** ⬜ Both tests PASS → Proceed to Demo

---

## 🟢 PHASE 2: Demo Account (30 Days Minimum)

### Setup
- [ ] Broker demo account created
- [ ] EA attached to XAUUSD M5 chart
- [ ] **Live-Ready Configuration:**
  ```
  RiskPercent = 0.5
  ConfidenceThreshold = 0.70
  MaxTradesPerDay = 5
  MaxDailyLoss = 3.0
  EnableFeatureLog = false
  EnablePredictionLog = false
  ```

### Daily Monitoring (First Week)
- [ ] **Day 1:** EA running | Trades: ___ | P&L: $___
- [ ] **Day 2:** EA running | Trades: ___ | P&L: $___
- [ ] **Day 3:** EA running | Trades: ___ | P&L: $___
- [ ] **Day 4:** EA running | Trades: ___ | P&L: $___
- [ ] **Day 5:** EA running | Trades: ___ | P&L: $___
- [ ] **Day 6:** EA running | Trades: ___ | P&L: $___
- [ ] **Day 7:** EA running | Trades: ___ | P&L: $___

### Weekly Checkpoints (Weeks 2-4)
- [ ] **Week 2:** Trades: ___ | WR: ___% | Net: $___ | DD: ___%
- [ ] **Week 3:** Trades: ___ | WR: ___% | Net: $___ | DD: ___%
- [ ] **Week 4:** Trades: ___ | WR: ___% | Net: $___ | DD: ___%

### 30-Day Validation
- [ ] **Total Trades:** _______ (Expected: ~150-200 at 5/day)
- [ ] **Win Rate:** _______% (Target: ≥70%)
- [ ] **Profit Factor:** _______ (Target: ≥3.0)
- [ ] **Max Drawdown:** _______% (Limit: <15%)
- [ ] **Circuit Breakers Triggered:** _____ times
- [ ] **Slippage vs Expected:** Acceptable ✅ / High ❌
- [ ] **Spread vs Expected:** Acceptable ✅ / High ❌
- [ ] **Execution Speed:** Good ✅ / Poor ❌

**GO/NO-GO:** ⬜ Demo results match OOS expectations → Proceed to Micro Live

---

## 🔵 PHASE 3: Micro Live ($500-1000, ≤0.5% Risk)

### Pre-Live Checklist
- [ ] Demo phase completed successfully (30+ days)
- [ ] Broker live account funded ($500-1000)
- [ ] VPS setup (optional but recommended)
- [ ] Emergency contact/kill switch ready

### Configuration
- [ ] **Initial Capital:** $_______
- [ ] **Risk per Trade:** 0.5% (MAX 1.0%)
- [ ] **Confidence:** 0.70
- [ ] **Daily Loss Limit:** 3%
- [ ] **Max Trades/Day:** 3

### Month 1 Monitoring
- [ ] **Week 1:** Trades: ___ | WR: ___% | Net: $___ | DD: ___%
  - [ ] Compare to demo: Similar ✅ / Worse ❌
  - [ ] Slippage acceptable: ✅ / ❌
  - [ ] No requotes/rejections: ✅ / ❌

- [ ] **Week 2:** Trades: ___ | WR: ___% | Net: $___ | DD: ___%
- [ ] **Week 3:** Trades: ___ | WR: ___% | Net: $___ | DD: ___%
- [ ] **Week 4:** Trades: ___ | WR: ___% | Net: $___ | DD: ___%

### Red Flag Checks
- [ ] Win rate >65% → ✅ / ❌
- [ ] Profit factor >2.0 → ✅ / ❌
- [ ] Max DD <15% → ✅ / ❌
- [ ] Execution quality acceptable → ✅ / ❌
- [ ] No unexpected broker issues → ✅ / ❌

**GO/NO-GO:** ⬜ 3+ months profitable → Consider scaling

---

## 🔄 PHASE 4: Ongoing Monitoring & Maintenance

### Monthly Reviews
- [ ] **Month ___:** WR: ___% | PF: ___ | DD: ___% | Status: ✅ ⚠️ ❌
- [ ] Compare to OOS baseline (74% WR, 4.6 PF)
- [ ] Review LONG:SHORT ratio (should be ~1:5-6)
- [ ] Check for regime changes (WR drops, PF degrades)

### Quarterly Stress Tests
- [ ] **Q__ 20__:** Re-run OOS test with latest 12 months
  - [ ] WR: ___% (vs backtest: ___%)
  - [ ] PF: ___ (vs backtest: ___)
  - [ ] Model holding up: ✅ / ❌

- [ ] Run degraded WR simulation (-10% WR)
  - [ ] Still profitable: ✅ / ❌

- [ ] Run regime shift test (remove best 20% trades)
  - [ ] Still profitable: ✅ / ❌

### Retraining Triggers
Retrain model if ANY of:
- [ ] Win rate drops below 65% for 4+ weeks
- [ ] Profit factor <2.0 for 2+ months
- [ ] Max DD exceeds 20%
- [ ] Trade frequency changes dramatically (>50%)
- [ ] Market structure shift detected

### Emergency Actions
If red flags appear:
1. **Immediate:** Reduce risk to 0.25%
2. **Immediate:** Increase confidence to 0.75
3. **Day 1:** Collect 2 weeks of feature + prediction logs
4. **Week 1:** Run parity checks
5. **Week 2:** Analyze trades for patterns
6. **Week 3:** Decide: Parameter adjust / Retrain / Pause

---

## 📊 Success Metrics Dashboard

| Phase | Status | WR | PF | DD | Notes |
|-------|--------|----|----|----|----|
| Parity Check | ⬜ | - | - | - | |
| Strategy Tester | ⬜ | - | - | - | |
| Demo (30d) | ⬜ | - | - | - | |
| Micro Live (3m) | ⬜ | - | - | - | |
| Full Live | ⬜ | - | - | - | |

---

## 🚨 ABORT CONDITIONS (Stop Trading Immediately)

- ❌ Feature parity fails (>5% deviation)
- ❌ Strategy Tester WR <60%
- ❌ Demo account drops >20%
- ❌ Live WR <60% for 2+ weeks
- ❌ Drawdown exceeds 25% at any time
- ❌ Profit factor <1.5 for 1+ month
- ❌ Broker execution issues (high slippage, frequent requotes)
- ❌ ONNX inference errors

---

**Emergency Contact:** [Your contact]  
**Kill Switch:** [Stop EA, close positions, withdraw to safety]

**Backtest Baseline (OOS):**  
WR: 74.1% | PF: 4.60 | Trades: 15,324 | Period: Oct 2024 - Oct 2025

**Conservative Target (Live-Ready):**  
WR: 80.5% | PF: 3.67 | Risk: 0.5% | Confidence: 0.70
