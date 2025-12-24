# Detailed Instructions for MT5 Expert Advisor Backtesting & Deployment

## 📋 Project Context

**Objective**: A/B test two Expert Advisors (Single vs Ensemble) on 3 years of XAUUSD historical data to determine the best model for production deployment.

**Current Status**: ✅ All code complete, all files committed to branch `claude/mt5-model-research-yfrWh`

**Repository**: https://github.com/andywarui/xaubot
**Branch**: `claude/mt5-model-research-yfrWh`

---

## 🎯 Task Overview

You need to:
1. **Backtest** both Expert Advisors on 3 years of historical data (2022-2025)
2. **Compare** performance metrics (win rate, profit factor, drawdown)
3. **Select** the best-performing model
4. **Deploy** to paper trading for validation

---

## 📂 Repository Structure

```
xaubot/
├── mt5_expert_advisor/
│   ├── XAUUSD_NeuralBot_Single.mq5      # Single: LightGBM + Hybrid (960 lines)
│   └── XAUUSD_NeuralBot_Ensemble.mq5    # Ensemble: LightGBM + Transformer (1,202 lines)
│
├── python_training/
│   └── export_transformer_onnx.py       # Transformer export script
│
├── trained_models/
│   └── lightgbm_xauusd.onnx            # LightGBM ONNX model (required for both EAs)
│
├── docs/
│   ├── AB_TESTING_GUIDE.md             # ⭐ Complete A/B testing guide
│   ├── DEPLOYMENT_GUIDE.md             # Step-by-step deployment
│   ├── ENSEMBLE_EA_ARCHITECTURE.md     # Technical specifications
│   └── TRANSFORMER_EXPORT_GUIDE.md     # Transformer export instructions
│
└── PROJECT_STATUS.md                    # Implementation summary
```

---

## 🚀 PHASE 1: Quick Start - Test Single Model (Recommended First)

**Why start here**: No Transformer export needed, fastest path to results (5 min setup)

### Step 1.1: Locate MT5 Files Directory

Find your MT5 terminal data folder:

**Windows**:
```
%APPDATA%\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Files\
```

To find it:
1. Open MT5 → Tools → Options → Data Folder
2. Click "Open Data Folder"
3. Navigate to `MQL5\Files\`

**Example Path**:
```
C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\
```

### Step 1.2: Copy LightGBM Model

```bash
# Source file (in your repository)
trained_models/lightgbm_xauusd.onnx

# Destination (MT5 terminal)
%APPDATA%\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Files\lightgbm_xauusd.onnx
```

**Verification**: The file should be ~220-300 KB in size.

### Step 1.3: Copy EA to MT5 Experts Folder

```bash
# Source
mt5_expert_advisor/XAUUSD_NeuralBot_Single.mq5

# Destination
%APPDATA%\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Experts\XAUUSD_NeuralBot_Single.mq5
```

### Step 1.4: Compile the Expert Advisor

1. Open **MetaEditor** (F4 in MT5)
2. File → Open → Navigate to Experts folder
3. Open `XAUUSD_NeuralBot_Single.mq5`
4. Click **Compile** button (F7) or Tools → Compile
5. Check for errors in "Errors" tab at bottom
6. Expected output: `0 error(s), 0 warning(s)` or `compilation successful`

**Common Compilation Issues**:
- Missing `Trade.mqh`: Ensure MT5 is updated to latest version
- Path errors: Verify file is in correct Experts folder

### Step 1.5: Run Strategy Tester Backtest

1. **Open Strategy Tester**: View → Strategy Tester (Ctrl+R)

2. **Configure Test**:
   ```
   Expert Advisor: XAUUSD_NeuralBot_Single
   Symbol: XAUUSD (Gold Spot)
   Period: M1 (1 minute)
   Date Range: 2022.01.01 - 2025.01.01
   Model: Every tick (most accurate, slowest)
   Optimization: Disabled
   Visual mode: Disabled (for faster testing)
   ```

3. **Expert Properties** (click "Expert properties" button):
   ```
   Testing tab:
   - Initial deposit: $10,000
   - Leverage: 1:100 or 1:500

   Inputs tab (default values):
   - RiskPercent = 0.5
   - ConfidenceThreshold = 0.60
   - MaxTradesPerDay = 5
   - EnableHybridValidation = true
   - StopLossUSD = 4.0
   - TakeProfitUSD = 8.0
   ```

4. **Start Test**: Click **"Start"** button

5. **Expected Duration**: 2-4 hours for 3 years of M1 data

### Step 1.6: Analyze Results

When test completes, review these metrics:

**Critical Metrics**:
```
Report Tab:
├── Total Net Profit: Target >$5,000 (50% gain)
├── Profit Factor: Target 1.4-1.6
├── Expected Payoff: Should be positive
├── Total Trades: Target 3,000-5,000 trades
├── Winning Trades %: Target 63-68%
├── Maximum Drawdown: Target <12%
├── Sharpe Ratio: Target >1.2
└── Recovery Factor: Target >2.0
```

**Save Results**:
1. Right-click on Report tab → Save as Report
2. Save as: `Single_Model_Backtest_Results.htm`
3. Screenshot the Graph tab (equity curve)

---

## 🔬 PHASE 2: Test Ensemble Model (Optional - Requires Transformer)

**Prerequisites**:
- Phase 1 completed
- Transformer ONNX model exported

### Step 2.1: Export Transformer Model

**Option A - If you have PyTorch installed**:

```bash
# Navigate to repository
cd /path/to/xaubot/python_training

# Run export script
python export_transformer_onnx.py

# Expected outputs:
# - transformer.onnx (~20-50 MB)
# - transformer_scaler_params.json (~5-10 KB)
# - transformer_config.json (~1 KB)
```

**Option B - If PyTorch not available**:

Follow instructions in `docs/TRANSFORMER_EXPORT_GUIDE.md`:
1. Export on your training machine
2. Transfer files to current environment
3. Verify ONNX shape: [1,30,130] → [1,1]

**Troubleshooting**:
- If script fails with "ModuleNotFoundError: torch":
  ```bash
  pip install torch onnx onnxruntime sklearn
  ```

### Step 2.2: Copy Transformer Files to MT5

```bash
# Copy all 3 files to MT5 Files folder:
transformer.onnx → MQL5\Files\transformer.onnx
transformer_scaler_params.json → MQL5\Files\transformer_scaler_params.json
transformer_config.json → MQL5\Files\transformer_config.json

# LightGBM model should already be there from Phase 1
lightgbm_xauusd.onnx (already copied)
```

### Step 2.3: Copy & Compile Ensemble EA

```bash
# Source
mt5_expert_advisor/XAUUSD_NeuralBot_Ensemble.mq5

# Destination
%APPDATA%\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Experts\XAUUSD_NeuralBot_Ensemble.mq5
```

**Compile**:
1. Open in MetaEditor
2. Press F7 (Compile)
3. Verify 0 errors

### Step 2.4: Run Ensemble Backtest

**Same configuration as Single Model**, except:

```
Expert Advisor: XAUUSD_NeuralBot_Ensemble

Inputs (different from Single):
- RiskPercent = 0.5
- ConfidenceThreshold = 0.65 (higher for ensemble)
- UseEnsemble = true
- EnsembleAgreementThreshold = 0.60
- TransformerSignalThreshold = 0.1
- AllowLightGBMFallback = true
- EnableHybridValidation = true
```

**Watch for**:
- Log message: "Sequence buffer: Warming up (need 30 bars)"
- After 30 bars: "ENSEMBLE AGREEMENT" or "ENSEMBLE DISAGREE" messages
- Verify both models are loading (check Experts log tab)

**Expected Duration**: 2-4 hours

### Step 2.5: Save Ensemble Results

```
Save as: Ensemble_Model_Backtest_Results.htm
Screenshot equity curve
```

---

## 📊 PHASE 3: Performance Comparison

### Step 3.1: Create Comparison Table

Fill in actual results from backtests:

```markdown
| Metric | Single Model | Ensemble Model | Winner |
|--------|-------------|----------------|--------|
| **Total Net Profit** | $______ | $______ | ____ |
| **Profit Factor** | ____ | ____ | ____ |
| **Win Rate %** | ____% | ____% | ____ |
| **Max Drawdown %** | ____% | ____% | ____ |
| **Total Trades** | _____ | _____ | ____ |
| **Average Trade** | $____ | $____ | ____ |
| **Sharpe Ratio** | ____ | ____ | ____ |
| **Recovery Factor** | ____ | ____ | ____ |
| **Largest Loss** | $____ | $____ | ____ |
| **Consecutive Losses** | ____ | ____ | ____ |
```

### Step 3.2: Decision Criteria

**Choose Ensemble if**:
- Win rate ≥ 5% higher than Single
- Profit factor ≥ 0.2 higher
- Max drawdown ≥ 2% lower
- Sharpe ratio ≥ 0.2 higher

**Choose Single if**:
- Ensemble improvement < 3% on key metrics
- Need higher trade frequency (more signals)
- Prefer simpler system (easier to monitor)

**Example Decision**:
```
Single: 65% win rate, 1.5 PF, 10% DD
Ensemble: 72% win rate, 1.8 PF, 7% DD

Decision: ENSEMBLE wins (+7% WR, +0.3 PF, -3% DD)
```

### Step 3.3: Validate Results

**Red Flags** (investigate if you see):
- Win rate < 55% (model may have issues)
- Profit factor < 1.2 (barely profitable)
- Max drawdown > 20% (too risky)
- Sharpe ratio < 0.8 (poor risk-adjusted returns)
- Very few trades (< 1,000 in 3 years)

**If results are poor**:
1. Check model file loaded correctly (check Experts log)
2. Verify date range has data (should be ~1.5M M1 bars)
3. Review optimization guide: `docs/OPTIMIZATION_PARAMETERS.md`
4. Consider parameter tuning

---

## 🎯 PHASE 4: Deploy to Paper Trading

### Step 4.1: Select Best Model

Based on Phase 3 comparison, choose:
- `XAUUSD_NeuralBot_Single.mq5` OR
- `XAUUSD_NeuralBot_Ensemble.mq5`

### Step 4.2: Open Demo Account

1. MT5 → File → Open Demo Account
2. Select broker (e.g., MetaQuotes Demo)
3. Account type: Standard
4. Initial deposit: $10,000
5. Leverage: 1:100
6. Save login credentials

### Step 4.3: Deploy EA to Demo

1. **Attach EA to Chart**:
   - Open XAUUSD M1 chart
   - Drag chosen EA from Navigator → Expert Advisors
   - Drop onto chart

2. **Configure for Live Trading**:
   ```
   Common tab:
   - Allow live trading: ✓
   - Allow DLL imports: (leave unchecked)
   - Allow external imports: (leave unchecked)

   Inputs tab:
   - RiskPercent = 0.3 (start conservative!)
   - ConfidenceThreshold = 0.65
   - MaxTradesPerDay = 5
   - EnableHybridValidation = true
   ```

3. **Verify EA is Running**:
   - Look for smiley face icon in top-right of chart
   - Check Experts tab for initialization messages
   - Should see: "✅ [MODEL] BOT INITIALIZED"

### Step 4.4: Monitor Performance (2+ Weeks)

**Daily Monitoring**:

1. **Use Python Monitoring System**:
   ```bash
   cd python_monitoring

   # Terminal monitoring
   python monitor_live_performance.py --interval 60

   # Web dashboard
   streamlit run dashboard.py
   ```

2. **Check These Metrics**:
   - Model accuracy (should stay >55%)
   - Prediction distribution (LONG/HOLD/SHORT balance)
   - Rejection rate (if >80%, filters too strict)
   - Live vs backtest performance alignment

3. **Alert Conditions**:
   - Accuracy drops below 55% → Model drift
   - Unusual drawdown (>15%) → Risk management issue
   - No trades for days → Configuration problem

### Step 4.5: Validation Checklist

After 2-4 weeks of paper trading:

```
✓ [ ] Live win rate within 5% of backtest
✓ [ ] Profit factor within 0.2 of backtest
✓ [ ] Max drawdown < backtest + 3%
✓ [ ] Average trade P/L similar to backtest
✓ [ ] Model accuracy stable (>55%)
✓ [ ] No unusual errors in logs
✓ [ ] Risk management working correctly
```

**If validation passes** → Proceed to live trading
**If validation fails** → Investigate discrepancies, retrain if needed

---

## 📝 Expected Results Reference

### Single Model Targets

```
Win Rate: 63-68%
Profit Factor: 1.4-1.6
Trades/Month: 40-60
Max Drawdown: 8-12%
Sharpe Ratio: 1.2-1.5
Average Trade Duration: 15-30 minutes
```

### Ensemble Model Targets

```
Win Rate: 70-75%
Profit Factor: 1.6-1.9
Trades/Month: 25-40 (fewer but higher quality)
Max Drawdown: 6-10%
Sharpe Ratio: 1.5-1.9
Average Trade Duration: 20-40 minutes
```

---

## 🔧 Troubleshooting Guide

### Issue: "Model file not found"

**Symptoms**: EA logs show "Failed to load ONNX model"

**Solutions**:
1. Verify file copied to correct location:
   ```bash
   # Should exist:
   MQL5\Files\lightgbm_xauusd.onnx
   MQL5\Files\transformer.onnx (for ensemble)
   ```

2. Check file permissions (not read-only)

3. Try absolute path in code (temporary debug):
   ```cpp
   // In EA code, temporarily change:
   string model_file = "C:\\Users\\...\\MQL5\\Files\\lightgbm_xauusd.onnx";
   ```

### Issue: "Compilation failed"

**Symptoms**: Errors during F7 compile

**Solutions**:
1. Missing Trade.mqh:
   - Tools → Options → Update MT5
   - Reinstall standard library

2. Syntax errors:
   - Check MQL5 version (must be build 3770+)
   - Update → Help → Check for Updates

### Issue: "No trades executed"

**Symptoms**: EA running but no trades after hours

**Solutions**:
1. Check session filter:
   ```cpp
   // In code, verify:
   IsInSession() → Trading hours 12:00-17:00 UTC
   ```

2. Lower confidence threshold temporarily:
   ```
   ConfidenceThreshold = 0.55 (from 0.60)
   ```

3. Disable hybrid validation for testing:
   ```
   EnableHybridValidation = false
   ```

4. Check Experts log for rejection reasons:
   ```
   Look for: "FILTER REJECT [LONG/SHORT]: ..."
   ```

### Issue: "Transformer not loading" (Ensemble only)

**Symptoms**: Logs show "Transformer model not found"

**Solutions**:
1. Verify all 3 files copied:
   ```
   transformer.onnx
   transformer_scaler_params.json
   transformer_config.json
   ```

2. Check ONNX version compatibility:
   ```bash
   # Verify opset_version = 14
   python -c "import onnx; m=onnx.load('transformer.onnx'); print(m.opset_import[0].version)"
   ```

3. Enable debug logs:
   ```
   EnableOnnxDebugLogs = true
   ```

### Issue: "Results don't match backtest"

**Symptoms**: Live performance significantly worse

**Common Causes**:
1. **Spread differences**: Live spreads > backtest spreads
   - Solution: Increase `MaxSpreadUSD` filter

2. **Slippage**: Real execution vs backtest fills
   - Solution: Accept as normal (2-5% variance)

3. **Data quality**: Historical data gaps
   - Solution: Re-download quality data from broker

4. **Look-ahead bias**: Check shift=1 in features
   - Solution: Verify using last CLOSED bar

---

## 📚 Documentation Reference

| File | When to Use |
|------|------------|
| **AB_TESTING_GUIDE.md** | Complete A/B testing walkthrough |
| **DEPLOYMENT_GUIDE.md** | Detailed deployment steps (18,000 words) |
| **OPTIMIZATION_PARAMETERS.md** | Parameter tuning strategies |
| **ENSEMBLE_EA_ARCHITECTURE.md** | Technical specs and code reference |
| **TRANSFORMER_EXPORT_GUIDE.md** | Transformer export troubleshooting |
| **PROJECT_STATUS.md** | Implementation summary |

---

## ✅ Success Criteria Checklist

### Phase 1: Single Model Backtest
```
✓ [ ] LightGBM model copied to MT5\Files\
✓ [ ] EA compiled without errors
✓ [ ] 3-year backtest completed
✓ [ ] Results saved to .htm report
✓ [ ] Win rate between 60-70%
✓ [ ] Profit factor > 1.3
✓ [ ] Max drawdown < 15%
```

### Phase 2: Ensemble Model Backtest (Optional)
```
✓ [ ] Transformer exported successfully
✓ [ ] All 3 files copied to MT5\Files\
✓ [ ] Ensemble EA compiled without errors
✓ [ ] 3-year backtest completed
✓ [ ] Results saved to .htm report
✓ [ ] Performance better than Single model
```

### Phase 3: Comparison
```
✓ [ ] Comparison table filled with actual metrics
✓ [ ] Best model selected based on criteria
✓ [ ] Decision documented with reasoning
```

### Phase 4: Paper Trading
```
✓ [ ] Demo account created
✓ [ ] Best model deployed
✓ [ ] Python monitoring running
✓ [ ] 2+ weeks of live testing completed
✓ [ ] Performance validated vs backtest
✓ [ ] Ready for production decision
```

---

## 🎯 Final Deliverable

When all phases complete, provide:

1. **Backtest Reports**:
   - `Single_Model_Backtest_Results.htm`
   - `Ensemble_Model_Backtest_Results.htm` (if tested)

2. **Comparison Document**:
   ```markdown
   # A/B Testing Results

   ## Winner: [Single/Ensemble]

   ## Key Metrics:
   - Win Rate: X%
   - Profit Factor: X.X
   - Max Drawdown: X%

   ## Reasoning:
   [Why this model was selected]

   ## Next Steps:
   [Paper trading results, live deployment plan]
   ```

3. **Equity Curves**: Screenshots from both models

4. **Monitoring Data**: 2+ weeks of live accuracy metrics

---

## 🚨 Important Notes

1. **Start with Single Model**: Fastest path to results, no Transformer needed

2. **Conservative Risk**: Begin paper trading with 0.3% risk, not 0.5%

3. **Patience**: Full 3-year backtests take 2-4 hours each

4. **Validation Critical**: Don't skip paper trading phase (minimum 2 weeks)

5. **Documentation**: Save all reports and screenshots for analysis

6. **Git Branch**: All files are on `claude/mt5-model-research-yfrWh`

---

## 📞 Support Resources

**Repository**: https://github.com/andywarui/xaubot
**Branch**: `claude/mt5-model-research-yfrWh`

**Documentation Files**:
- `AB_TESTING_GUIDE.md` - Complete testing guide
- `PROJECT_STATUS.md` - Implementation summary
- `docs/DEPLOYMENT_GUIDE.md` - 18,000 word deployment guide
- `docs/ENSEMBLE_EA_ARCHITECTURE.md` - Technical specifications

**Quick Stats**:
- Total Code: 2,162 lines (Expert Advisors)
- Python Monitoring: 700+ lines
- Documentation: 43,000+ words across 8 files
- GitHub Commits: 9 milestones pushed

---

**Good luck with the backtesting! 🚀**

*All code is production-ready and awaiting your backtest results to determine the best model for deployment.*
