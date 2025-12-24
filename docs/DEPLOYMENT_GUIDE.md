# XAUUSD Neural Bot - Deployment Guide

Complete guide for deploying the hybrid ML+Technical trading system on MT5.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [Configuration](#configuration)
5. [Testing & Validation](#testing--validation)
6. [Live Deployment](#live-deployment)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Optimization Guide](#optimization-guide)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

**For experienced users:**

```bash
# 1. Copy EA to MT5
cp mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5 "<MT5_PATH>/MQL5/Experts/"

# 2. Copy model files
cp python_training/models/lightgbm_xauusd.onnx "<MT5_PATH>/MQL5/Files/"

# 3. Compile EA in MetaEditor

# 4. Attach to XAUUSD M1 chart with recommended settings (see Configuration section)

# 5. Start monitoring
cd python_monitoring
pip install -r requirements.txt
python monitor_live_performance.py &
streamlit run dashboard.py
```

---

## System Requirements

### Hardware
- **CPU:** Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM:** Minimum 8GB, recommended 16GB
- **Storage:** 5GB free space (for MT5 + historical data)
- **Internet:** Stable connection (< 50ms latency to broker)

### Software
- **MetaTrader 5:** Build 3770 or later
- **Python:** 3.8-3.11 (for monitoring scripts)
- **OS:** Windows 10/11, Linux (via Wine), macOS (via Wine/Crossover)

### Broker Requirements
- **Symbol:** XAUUSD (Gold vs USD)
- **Minimum Spread:** < $2 USD typical
- **Leverage:** Minimum 1:100
- **Minimum Deposit:** $500 recommended for 0.5% risk
- **Order Execution:** < 50ms average
- **Historical Data:** At least 1 year of M1 data

---

## Installation Steps

### Step 1: Setup MetaTrader 5

1. **Download and Install MT5**
   - Get from your broker or [metaquotes.net](https://www.metaquotes.net/en/metatrader5)
   - Complete installation wizard
   - Login to demo or live account

2. **Verify Data Folder**
   - In MT5: `File → Open Data Folder`
   - This opens your MT5 data directory (e.g., `C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXX\`)
   - Note this path for later

3. **Enable Algo Trading**
   - Tools → Options → Expert Advisors
   - ✅ Enable "Allow automated trading"
   - ✅ Enable "Allow DLL imports" (if needed)
   - ✅ Enable "Allow WebRequest for listed URL"

4. **Add XAUUSD to Market Watch**
   - Right-click in Market Watch
   - "Show All" or search for "XAUUSD"
   - Ensure quotes are streaming

### Step 2: Install Expert Advisor

1. **Copy EA File**
   ```bash
   # From project root
   cp mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5 "<MT5_DATA_FOLDER>/MQL5/Experts/"
   ```

   **Windows example:**
   ```cmd
   copy mt5_expert_advisor\XAUUSD_NeuralBot_M1.mq5 "C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\XXXXX\MQL5\Experts\"
   ```

2. **Copy ONNX Model**
   ```bash
   cp python_training/models/lightgbm_xauusd.onnx "<MT5_DATA_FOLDER>/MQL5/Files/"
   ```

   **Verify files:**
   - EA at: `MQL5/Experts/XAUUSD_NeuralBot_M1.mq5`
   - Model at: `MQL5/Files/lightgbm_xauusd.onnx`

3. **Compile EA**
   - Open MetaEditor (Tools → MetaQuotes Language Editor)
   - File → Open → Navigate to `MQL5/Experts/XAUUSD_NeuralBot_M1.mq5`
   - Click "Compile" button (F7) or Tools → Compile
   - Check "Errors" tab - should show "0 error(s), 0 warning(s)"
   - If successful: `XAUUSD_NeuralBot_M1.ex5` created

4. **Refresh Navigator**
   - In MT5, right-click Navigator panel
   - "Refresh" to see new EA
   - Should appear under "Expert Advisors"

### Step 3: Setup Python Monitoring (Optional but Recommended)

1. **Install Python Dependencies**
   ```bash
   cd python_monitoring
   pip install -r requirements.txt
   ```

2. **Verify MT5 Python Connection**
   ```bash
   python -c "import MetaTrader5 as mt5; print('OK' if mt5.initialize() else 'FAIL')"
   ```
   Should print "OK"

3. **Test Monitor Script**
   ```bash
   python monitor_live_performance.py --duration 0.1  # 6 minute test
   ```

---

## Configuration

### Recommended Settings (Conservative)

#### Basic Settings
```
RiskPercent = 0.5              # Risk 0.5% per trade
ConfidenceThreshold = 0.65     # Higher threshold = fewer, better trades
MaxTradesPerDay = 3            # Limit daily exposure
MaxDailyLoss = 4.0             # Stop trading after 4% daily loss
StopLossUSD = 4.0              # $4 stop loss (reasonable for XAUUSD)
TakeProfitUSD = 8.0            # $8 take profit (2:1 R:R)
MaxMarginPercent = 50.0        # Use max 50% of free margin per trade
```

#### Hybrid Validation (ENABLED - Research-backed)
```
EnableHybridValidation = true  # Enable ML + Technical filters
MaxSpreadUSD = 2.0             # Reject if spread > $2
RSI_OverboughtLevel = 70.0     # Reject LONG if RSI > 70
RSI_OversoldLevel = 30.0       # Reject SHORT if RSI < 30
ATR_MinLevel = 1.5             # Min volatility (avoid choppy markets)
ATR_MaxLevel = 8.0             # Max volatility (avoid extreme moves)
ADX_MinStrength = 20.0         # Min trend strength (avoid ranging)
RequireMTFAlignment = true     # Require M15 & H1 EMA alignment
```

#### Logging (Enable for monitoring)
```
EnableFeatureLog = false       # Only for debugging (generates large files)
EnablePredictionLog = true     # REQUIRED for monitoring dashboard
PredictionLogFile = "prediction_log.csv"
EnableOnnxDebugLogs = false    # Only for troubleshooting model issues
```

### Aggressive Settings (Higher Risk)

```
RiskPercent = 1.0              # 1% risk per trade
ConfidenceThreshold = 0.55     # Lower threshold = more trades
MaxTradesPerDay = 5            # More trading opportunities
StopLossUSD = 3.0              # Tighter stop
TakeProfitUSD = 6.0            # Tighter target
EnableHybridValidation = false # Pure ML (not recommended)
```

### Setting Profiles by Session

**London Session (08:00-17:00 UTC) - High liquidity:**
```
MaxSpreadUSD = 1.5             # Tighter spread
ATR_MinLevel = 2.0             # Higher volatility expected
MaxTradesPerDay = 5            # More opportunities
```

**NY Session (13:00-22:00 UTC) - Highest volatility:**
```
ATR_MinLevel = 2.5             # Even higher volatility
ATR_MaxLevel = 10.0            # Allow more extreme moves
StopLossUSD = 5.0              # Wider stops for volatility
TakeProfitUSD = 10.0           # Wider targets
```

**Asian Session (00:00-09:00 UTC) - Lower liquidity:**
```
MaxSpreadUSD = 2.5             # Wider spreads expected
ATR_MinLevel = 1.0             # Lower volatility
MaxTradesPerDay = 2            # Fewer opportunities
ConfidenceThreshold = 0.70     # Higher confidence needed
```

---

## Testing & Validation

### Phase 1: Strategy Tester (Backtest)

1. **Open Strategy Tester**
   - View → Strategy Tester (Ctrl+R)

2. **Configure Test**
   ```
   Expert Advisor: XAUUSD_NeuralBot_M1
   Symbol: XAUUSD
   Period: M1
   Dates: Last 6 months
   Mode: Every tick (most accurate)
   Optimization: Disabled (for now)
   ```

3. **Start Test**
   - Click "Start"
   - Monitor "Journal" and "Experts" tabs for errors
   - Wait for completion (may take 30+ minutes)

4. **Analyze Results**
   - **Minimum acceptable metrics:**
     - Total trades: > 50
     - Win rate: > 60%
     - Profit factor: > 1.3
     - Max drawdown: < 15%
     - Sharpe ratio: > 0.5

   - **Good metrics:**
     - Win rate: 70-80%
     - Profit factor: 1.8-2.5
     - Max drawdown: 8-12%
     - Sharpe ratio: 1.0-1.5

5. **Review Trades**
   - Check "Graph" tab for equity curve
   - Ensure smooth growth (not spiky)
   - Review individual trades in "Results" tab

### Phase 2: Demo Account (Forward Test)

**CRITICAL: Never skip this step before live trading!**

1. **Create Demo Account**
   - File → Open an Account → Demo Account
   - Choose reputable broker
   - Fund with realistic amount (e.g., $10,000)

2. **Attach EA to Chart**
   - Open XAUUSD M1 chart
   - Drag EA from Navigator to chart
   - Configure with conservative settings
   - ✅ Enable "Allow live trading"
   - Click "OK"

3. **Run for Minimum 2 Weeks**
   - **Week 1:** Monitor intensively
     - Check trades 2-3x daily
     - Verify hybrid filters working
     - Watch for errors in "Experts" tab

   - **Week 2:** Normal monitoring
     - Daily dashboard checks
     - Weekly performance review

4. **Validation Checklist**
   ```
   ✅ No runtime errors in logs
   ✅ Predictions being logged correctly
   ✅ Hybrid filters rejecting signals appropriately
   ✅ Actual spread < MaxSpreadUSD
   ✅ Trades executing within 1 second
   ✅ SL/TP placed correctly
   ✅ No unexpected position sizing
   ✅ Dashboard showing reasonable accuracy (> 60%)
   ✅ Profit factor > 1.3
   ✅ Win rate aligns with backtest (±10%)
   ```

5. **Compare Demo vs Backtest**
   - Similar win rate (±10% acceptable)
   - Similar trade frequency (±20% acceptable)
   - Similar profit factor (±0.3 acceptable)

   **Red flags:**
   - Win rate drops > 15%
   - Profit factor < 1.0
   - Significantly more rejected signals than expected
   - Execution delays > 2 seconds

---

## Live Deployment

### Pre-Live Checklist

```
Account Setup:
✅ Funded with risk capital only (money you can afford to lose)
✅ Appropriate leverage (1:100 minimum, 1:200-1:500 recommended)
✅ VPS or stable internet (< 50ms latency to broker)
✅ Email/phone alerts configured in MT5

EA Configuration:
✅ Conservative settings loaded
✅ EnablePredictionLog = true
✅ EnableHybridValidation = true
✅ Tested on demo for 2+ weeks
✅ Backtest results acceptable

Monitoring Setup:
✅ Python monitoring scripts running
✅ Dashboard accessible (streamlit run dashboard.py)
✅ Alert thresholds configured
✅ Daily review schedule established

Risk Management:
✅ Maximum 0.5% risk per trade
✅ Maximum 5% total portfolio risk
✅ Daily loss limit set (4% recommended)
✅ Emergency stop-loss plan documented
```

### Going Live

1. **Start Small**
   ```
   Week 1: RiskPercent = 0.25%  (half normal)
   Week 2: RiskPercent = 0.35%
   Week 3: RiskPercent = 0.5%   (normal)
   ```

2. **Attach EA**
   - Use same process as demo
   - ✅ Triple-check settings before clicking "OK"
   - ✅ Verify "Allow live trading" is enabled
   - ✅ Confirm correct ONNX model loaded

3. **Monitor First Trade**
   - Stay at computer for first 1-2 hours
   - Watch for first signal
   - Verify trade execution
   - Check SL/TP placement
   - Confirm prediction logged

4. **Daily Routine**
   - **Morning (before market open):**
     - Check dashboard for overnight performance
     - Review any trades taken
     - Verify EA still running (check for errors)

   - **During trading hours:**
     - Monitor dashboard for model accuracy
     - Watch for filter rejection patterns
     - Check spread conditions

   - **Evening (after market close):**
     - Review full day performance
     - Check monitoring_results.json
     - Note any unusual patterns

### VPS Deployment (Recommended)

**Why VPS?**
- 24/7 uptime (no PC shutdowns)
- Low latency to broker servers
- No local internet/power issues

**Setup:**

1. **Choose Provider**
   - Recommended: ForexVPS, BeeksFX, Vultr, DigitalOcean
   - Specs: 2 vCPU, 4GB RAM, Windows Server

2. **Install MT5 on VPS**
   - Same process as local installation
   - Login to your broker account

3. **Copy Files to VPS**
   ```bash
   # Use Remote Desktop or SCP
   scp mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5 user@vps:/path/to/mt5/MQL5/Experts/
   scp python_training/models/lightgbm_xauusd.onnx user@vps:/path/to/mt5/MQL5/Files/
   ```

4. **Setup Auto-Start**
   - Configure MT5 to start with Windows
   - Auto-login enabled
   - EA auto-attached to chart (save workspace)

5. **Remote Monitoring**
   - Access VPS via Remote Desktop
   - Or: Setup Python monitoring with webhook alerts

---

## Monitoring & Maintenance

### Daily Monitoring (5 minutes)

1. **Check Dashboard**
   ```bash
   streamlit run dashboard.py
   ```
   - Review last 24h accuracy
   - Check active positions
   - Verify prediction count is normal

2. **Review Logs**
   - MT5 Experts tab: No errors
   - Filter rejection reasons (if any)
   - Execution quality

3. **Account Health**
   - Daily P/L
   - Drawdown level
   - Margin usage

### Weekly Review (30 minutes)

1. **Performance Metrics**
   - Weekly win rate vs target (70-80%)
   - Profit factor > 1.3
   - Average trade duration
   - Filter rejection rate

2. **Model Accuracy Analysis**
   ```bash
   python monitor_live_performance.py --duration 0.1
   ```
   - Overall accuracy > 60%
   - Per-class accuracy check
   - High-confidence accuracy > 75%

3. **Parameter Adjustment (if needed)**
   - If accuracy drops < 55%: Consider retraining
   - If too many rejections: Relax filters slightly
   - If too few trades: Lower confidence threshold

### Monthly Maintenance

1. **Model Retraining**
   ```bash
   # Fetch latest data
   python src/fetch_mt5_data.py --days 365

   # Retrain model
   python src/train_lightgbm.py

   # Export to ONNX
   python src/export_to_onnx.py

   # Validate
   python python_training/validate_mt5_pipeline.py

   # Deploy new model (restart EA after copying)
   cp python_training/models/lightgbm_xauusd.onnx <MT5_PATH>/MQL5/Files/
   ```

2. **Backtest New Model**
   - Run Strategy Tester on last 3 months
   - Compare metrics with previous model
   - Only deploy if improvement confirmed

3. **Review Overall Strategy**
   - Is hybrid validation helping? (Compare EnableHybridValidation true vs false)
   - Are filter parameters optimal? (Run optimization in Strategy Tester)
   - Should confidence threshold change?

---

## Optimization Guide

### Strategy Tester Optimization

1. **Open Strategy Tester → Settings → Optimization**

2. **Parameters to Optimize** (start with these):
   ```
   ConfidenceThreshold: 0.55 to 0.75, step 0.05
   RSI_OverboughtLevel: 65 to 75, step 5
   RSI_OversoldLevel: 25 to 35, step 5
   ATR_MinLevel: 1.0 to 2.5, step 0.5
   ADX_MinStrength: 15 to 25, step 5
   ```

3. **Optimization Settings**
   ```
   Mode: Genetic Algorithm (faster) or Complete (thorough)
   Optimization Criterion: Profit Factor or Sharpe Ratio
   Forward Period: 25% (for walk-forward validation)
   ```

4. **Run Optimization**
   - Takes 2-12 hours depending on parameters
   - Review "Optimization Results" tab
   - Sort by Profit Factor or Sharpe Ratio

5. **Validate Results**
   - Best parameters should work across forward period
   - Compare multiple top results (not just #1)
   - Test on different time periods
   - Avoid over-optimization (unrealistic results)

### A/B Testing Framework

Test one change at a time:

**Test 1: Hybrid Validation Impact**
```
Run A: EnableHybridValidation = false (2 weeks)
Run B: EnableHybridValidation = true  (2 weeks)
Compare: Win rate, profit factor, trade count
```

**Test 2: Confidence Threshold**
```
Run A: ConfidenceThreshold = 0.60 (2 weeks)
Run B: ConfidenceThreshold = 0.65 (2 weeks)
Run C: ConfidenceThreshold = 0.70 (2 weeks)
Compare: Win rate vs trade frequency trade-off
```

**Test 3: Risk Per Trade**
```
Run A: RiskPercent = 0.5% (2 weeks)
Run B: RiskPercent = 0.75% (2 weeks)
Compare: Returns vs drawdown
```

---

## Troubleshooting

### EA Won't Load

**Error: "Cannot load ONNX model"**
```
Solutions:
1. Verify file exists:
   - Check MQL5/Files/lightgbm_xauusd.onnx exists
   - File size should be ~131 bytes (LightGBM) or larger

2. Check EA logs (Experts tab):
   - Look for "ONNX load" messages
   - Note which path it tried

3. Try absolute path:
   - Copy full path from Windows Explorer
   - Update EA to use absolute path temporarily

4. Recompile EA:
   - Open in MetaEditor
   - Compile again (F7)
   - Reload EA on chart
```

**Error: "Indicator initialization failed"**
```
Solutions:
1. Wait for historical data to load
   - Close EA
   - Let chart load 1000+ bars
   - Reattach EA

2. Check indicator periods:
   - ATR(14), RSI(14), EMA(10,20,50) need enough bars
   - Ensure chart has 100+ bars minimum

3. Verify symbol:
   - Must be XAUUSD exactly
   - Check symbol name in Market Watch
```

### No Trades Being Taken

**Check 1: Session Hours**
```
Current: IsInSession() returns (dt.hour >= 12 && dt.hour < 17)

If outside this window, EA won't trade.
Solution: Modify session hours or disable session check temporarily
```

**Check 2: Hybrid Filters**
```
Enable EnablePredictionLog = true
Check logs for "FILTER REJECT" messages:
   - Spread too high → Wait for better spread or increase MaxSpreadUSD
   - RSI overbought/oversold → Normal, wait for better conditions
   - MACD bearish/bullish → Trend not aligned, normal filtering
   - ADX too weak → Market ranging, normal filtering
   - ATR outside range → Volatility unsuitable, normal filtering
   - Price below/above MTF EMAs → Trend not confirmed, normal filtering

If too many rejections:
   - Temporarily set EnableHybridValidation = false
   - Compare trade frequency with/without filters
```

**Check 3: Confidence Threshold**
```
If ConfidenceThreshold = 0.70 and model rarely exceeds this:
   - Lower to 0.60 or 0.55
   - Check prediction_log.csv for actual confidence values
```

**Check 4: Risk Limits**
```
If MaxTradesPerDay = 3 and already taken 3 trades:
   - Wait for next day
   - Or increase limit temporarily

If MaxDailyLoss triggered:
   - Check daily P/L
   - Wait for next day
```

### Accuracy Dropping

**Model Drift Detected**
```
If dashboard shows accuracy < 55%:

1. Check market regime:
   - High volatility events (news, gaps)
   - Unusual market conditions
   → Wait for normalization

2. Review recent trades:
   - Are losses on specific signal types (LONG/SHORT)?
   - Are filters working correctly?
   → May need parameter adjustment

3. Retrain model:
   - If drift persists > 1 week
   - Follow Monthly Maintenance → Model Retraining
```

### Monitoring Dashboard Not Updating

**prediction_log.csv Empty**
```
Solutions:
1. Enable in EA:
   EnablePredictionLog = true

2. Check EA is making predictions:
   - Verify EA attached to chart
   - Check "Experts" tab for prediction messages

3. Check file location:
   - Should be in MQL5/Files/prediction_log.csv
   - Use File → Open Data Folder to locate
```

**MT5 Connection Failed**
```
python monitor_live_performance.py

Error: MT5 initialization failed

Solutions:
1. Ensure MT5 terminal is running
2. Reinstall MetaTrader5 package:
   pip uninstall MetaTrader5
   pip install MetaTrader5
3. Try explicit initialization:
   python -c "import MetaTrader5 as mt5; mt5.initialize('<MT5_PATH>')"
```

---

## Best Practices Summary

### DO's ✅
- ✅ Always test on demo for 2+ weeks before live
- ✅ Enable hybrid validation (research-proven improvement)
- ✅ Start with conservative settings (0.5% risk)
- ✅ Monitor daily via dashboard
- ✅ Log predictions for analysis
- ✅ Retrain model monthly
- ✅ Use VPS for 24/7 operation
- ✅ Keep detailed trading journal
- ✅ Review weekly performance metrics

### DON'Ts ❌
- ❌ Never skip demo testing
- ❌ Never trade with money you can't afford to lose
- ❌ Never disable hybrid validation without testing
- ❌ Never ignore accuracy drops below 55%
- ❌ Never increase risk after losses (revenge trading)
- ❌ Never run multiple instances on same account
- ❌ Never modify EA code without thorough testing
- ❌ Never trust backtest alone (always forward test)

---

## Quick Reference

### File Locations
```
EA:              MQL5/Experts/XAUUSD_NeuralBot_M1.mq5
Compiled EA:     MQL5/Experts/XAUUSD_NeuralBot_M1.ex5
ONNX Model:      MQL5/Files/lightgbm_xauusd.onnx
Prediction Log:  MQL5/Files/prediction_log.csv
Feature Log:     MQL5/Files/feature_log.csv
```

### Key Metrics Targets
```
Win Rate:        70-80%
Profit Factor:   1.8-2.5
Max Drawdown:    < 12%
Sharpe Ratio:    > 1.0
Model Accuracy:  > 65%
High Conf. Acc.: > 80%
```

### Support Contacts
- EA Issues: Check `docs/MT5_IMPLEMENTATION_RESEARCH_REPORT.md`
- Model Issues: Review `src/train_lightgbm.py`
- Monitoring: See `python_monitoring/README.md`

---

**Last Updated:** 2025-12-22
**Version:** 1.0 (Post-Research Implementation)
