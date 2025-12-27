# MT5 Installation & Verification Checklist

## ✅ Pre-Installation

- [ ] MT5 platform installed
- [ ] XAUUSD symbol available in Market Watch
- [ ] M1 (1-minute) historical data downloaded (2022-2024)
- [ ] Files extracted from MT5_XAUBOT folder

## ✅ File Installation

### Step 1: Copy EA File
- [ ] Opened MT5 Data Folder (File → Open Data Folder)
- [ ] Navigated to `MQL5/Experts/` folder
- [ ] Copied `XAUUSD_Neural_Bot_v2.mq5` to Experts folder
- [ ] Verified file exists in: `MQL5/Experts/XAUUSD_Neural_Bot_v2.mq5`

### Step 2: Copy Model File
- [ ] Navigated to `MQL5/Files/` folder
- [ ] Copied `lightgbm_real_26features.onnx` to Files folder
- [ ] Verified file exists: `MQL5/Files/lightgbm_real_26features.onnx`
- [ ] Verified file size: ~302 KB

### Step 3: Compile EA
- [ ] Opened MetaEditor (F4)
- [ ] Found EA in: `Experts/XAUUSD_Neural_Bot_v2.mq5`
- [ ] Clicked Compile (F7)
- [ ] Compilation successful: "0 errors, 0 warnings"
- [ ] Closed MetaEditor

## ✅ Backtest Configuration

### Strategy Tester Settings
- [ ] Opened Strategy Tester (Ctrl+R)
- [ ] Selected EA: `XAUUSD_Neural_Bot_v2`
- [ ] Symbol: `XAUUSD`
- [ ] Period: `M1` (1 minute)
- [ ] Date: `2022.01.01` to `2024.12.31`
- [ ] Model: `Every tick`
- [ ] Deposit: `10000` USD
- [ ] Leverage: `1:500` (or broker default)

### Input Parameters - CRITICAL!
- [ ] `InpRiskPercent`: `0.5`
- [ ] `InpConfidenceThreshold`: `0.35` ⚠️ MUST BE 0.35
- [ ] `InpMaxTradesPerDay`: `10`
- [ ] `InpATRMultiplierSL`: `1.5`
- [ ] `InpRiskRewardRatio`: `2.0`
- [ ] `InpMagicNumber`: `230172`
- [ ] `InpUseValidation`: **`false`** ⚠️ MUST BE FALSE!
- [ ] `InpModelPath`: `lightgbm_real_26features.onnx`

### Visual Verification
- [ ] All parameters match the list above
- [ ] No red errors in Journal tab
- [ ] "Every tick" model selected (not "1 minute OHLC")

## ✅ Running Backtest

- [ ] Clicked **Start** button
- [ ] Journal shows: "XAUUSD Neural Bot v2.0 - Initializing"
- [ ] Journal shows: "✓ ONNX Model loaded"
- [ ] Journal shows: "✓ Indicators initialized"
- [ ] Journal shows: "Validation: DISABLED ✓"
- [ ] No "ERROR" messages in Journal
- [ ] Progress bar advancing
- [ ] Waiting for completion (~10-15 minutes)

## ✅ Results Verification

### Expected Results (approximate)
- [ ] **Total Trades**: ~370-390 trades
- [ ] **Net Profit**: ~$45,000 - $50,000
- [ ] **Return %**: ~450% - 500%
- [ ] **Win Rate**: ~42% - 46%
- [ ] **Profit Factor**: ~1.4 - 1.7
- [ ] **Max Drawdown**: ~15% - 20%
- [ ] **Trades/Year**: ~120-130

### Results Tab Check
- [ ] Net profit is POSITIVE (+$40K+)
- [ ] Total trades > 300
- [ ] Win rate > 40%
- [ ] Profit factor > 1.2
- [ ] Graph shows upward trend

### Graph Tab Check
- [ ] Balance line trending upward
- [ ] Drawdown periods recover
- [ ] No catastrophic drops
- [ ] Final balance ~$55K-$60K

## ✅ Troubleshooting (If Results Are Wrong)

### If 0 Trades Executed:
- [ ] Check `InpUseValidation` = **false** (NOT true!)
- [ ] Check `InpConfidenceThreshold` = 0.35
- [ ] Check Period = M1 (not M5, M15, etc.)
- [ ] Check date range covers 2022-2024
- [ ] Re-download M1 historical data

### If Model Load Failed:
- [ ] Verify file: `MQL5/Files/lightgbm_real_26features.onnx` exists
- [ ] Check file size: ~302 KB
- [ ] No spaces in filename
- [ ] Restart MT5
- [ ] Try absolute path in `InpModelPath`

### If Results Very Different (>50% difference):
- [ ] Verify ALL parameters match exactly
- [ ] Check "Every tick" model selected
- [ ] Review spread settings (should be realistic, ~0.2-0.5 for XAUUSD)
- [ ] Check commission settings (0 for most brokers on demo)

### If Backtest Very Slow:
- [ ] Normal for 3 years of M1 data
- [ ] Should take 10-15 minutes
- [ ] Check CPU usage (should be high)
- [ ] Don't interrupt - let it complete

## ✅ Next Steps After Successful Backtest

- [ ] Results match expected range (±10% variation OK)
- [ ] Screenshots saved for reference
- [ ] Ready to proceed to **Paper Trading**

### Paper Trading Checklist
- [ ] Open demo account
- [ ] Fund with $10,000 virtual
- [ ] Drag EA to XAUUSD M1 chart
- [ ] Use SAME parameters as backtest
- [ ] Enable AutoTrading (Ctrl+E)
- [ ] Allow live trading in EA settings
- [ ] Monitor for 30 days minimum

### Paper Trading Success Criteria (30 days)
- [ ] Win rate stays >40%
- [ ] Profit factor stays >1.2
- [ ] Max drawdown <20%
- [ ] No abnormal behavior
- [ ] Trades match backtest characteristics

## ✅ Live Trading Prerequisites (After Paper Trading)

⚠️ **DO NOT deploy to live until ALL criteria met:**

- [ ] 30+ days successful paper trading
- [ ] Consistent profitability (no losing weeks)
- [ ] Win rate maintained >40%
- [ ] Profit factor maintained >1.2
- [ ] Max drawdown stayed <20%
- [ ] You understand all risks
- [ ] Using only risk capital (money you can afford to lose)
- [ ] Starting with micro lots (0.01)
- [ ] Have stop-loss plan if drawdown exceeds 25%

## 📝 Notes & Observations

Date: _______________

Backtest Results:
- Net Profit: $_____________
- Win Rate: _____________%
- Profit Factor: _____________
- Max Drawdown: _____________%
- Total Trades: _____________

Issues Encountered:
_________________________________________
_________________________________________
_________________________________________

Resolution:
_________________________________________
_________________________________________
_________________________________________

---

**CRITICAL REMINDER**:
- Keep `InpUseValidation = false`
- Use 0.35 confidence threshold
- M1 timeframe only
- Paper trade for 30 days minimum before live!

**Expected Performance**: +472% over 3 years (backtest)
**Your Results**: Match within ±10% = ✅ Success!
