# MT5 Deployment Guide

## Files Ready

✅ **ONNX Model**: `mt5_expert_advisor/Files/lightgbm_xauusd.onnx` (3.2 MB)
✅ **EA Code**: `mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5`
✅ **Features**: 26 features matching Python training

## Deployment Steps

### 1. Copy ONNX Model to MT5

**Windows Path:**
```
C:\Users\[USERNAME]\AppData\Roaming\MetaQuotes\Terminal\[BROKER_ID]\MQL5\Files\
```

**Copy:**
```
mt5_expert_advisor/Files/lightgbm_xauusd.onnx
→ MQL5/Files/lightgbm_xauusd.onnx
```

### 2. Copy EA to MT5

**Copy:**
```
mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5
→ MQL5/Experts/XAUUSD_NeuralBot_M1.mq5
```

### 3. Compile EA

1. Open **MetaEditor** (F4 in MT5)
2. Open `XAUUSD_NeuralBot_M1.mq5`
3. Press **F7** to compile
4. Check for errors in **Toolbox** tab
5. Should see: "0 error(s), 0 warning(s)"

### 4. Strategy Tester Setup

1. Open **Strategy Tester** (Ctrl+R in MT5)
2. **Expert Advisor**: XAUUSD_NeuralBot_M1
3. **Symbol**: XAUUSD
4. **Period**: M5 (chart timeframe)
5. **Date range**: Last 3 months
6. **Model**: Every tick based on real ticks
7. **Optimization**: Disabled (for first test)

**Parameters:**
- RiskPercent: 0.5
- ConfidenceThreshold: 0.60
- MaxTradesPerDay: 5
- MaxDailyLoss: 4.0

### 5. Run Strategy Tester

Click **Start** and monitor:

**Check Logs:**
- "ONNX model loaded successfully"
- "Input: [1, 26], Output: [1, 3]"
- No "ERROR: ONNX inference failed"

**Expected Behavior:**
- Trades only during 12:00-17:00 UTC
- ~500-600 trades per month
- Win rate ~75%
- More SHORT than LONG signals

### 6. Verify Results

**Graph Tab:**
- Check equity curve (should be upward)
- Check drawdown (should be <5%)

**Results Tab:**
- Total trades: ~1500-2000 (3 months)
- Profit factor: >2.0
- Win rate: >70%

**Report Tab:**
- Export HTML report
- Review trade distribution

## Expected Performance (3 months)

Based on backtest:
- **Trades**: ~1,500
- **Win rate**: 75%
- **Profit factor**: 6.29
- **Max drawdown**: <5%
- **SHORT trades**: ~85% of total
- **LONG trades**: ~15% of total

## Troubleshooting

### "ONNX model not found"
- Check file is in `MQL5/Files/lightgbm_xauusd.onnx`
- File name is case-sensitive
- No spaces in filename

### "ONNX inference failed"
- Check input shape is [1, 26]
- Check output shape is [1, 3]
- Verify all 26 features are calculated

### "No trades generated"
- Check session hours (12:00-17:00 UTC)
- Check confidence threshold (try 0.50)
- Check risk limits not breached
- Verify IsNewM1Bar() is triggering

### Compilation errors
- Ensure MQL5 build 3802+ (for ONNX support)
- Check all indicator handles are valid
- Verify Trade.mqh is included

## Live Deployment (After Testing)

### Demo Account (30 days minimum)

1. Attach EA to **M5 chart** (XAUUSD)
2. Enable **AutoTrading** (Ctrl+E)
3. Monitor daily:
   - Check logs for errors
   - Verify trades match backtest behavior
   - Track win rate and P&L

### Live Account (After Demo Success)

1. Start with **minimum capital** ($500-$1000)
2. Use **conservative settings**:
   - RiskPercent: 0.3
   - ConfidenceThreshold: 0.65
   - MaxTradesPerDay: 3
3. Monitor for **2 weeks** before scaling
4. Keep **manual override** ready

## EA Behavior

### M1 Execution on M5 Chart

```
OnTick() called every tick (M5 chart)
    ↓
IsNewM1Bar() checks if new M1 bar closed
    ↓ (if true)
CalculateM1Features() - 26 features
    ↓
PredictSignal() - ONNX inference
    ↓
Returns: 0=SHORT, 1=HOLD, 2=LONG + confidence
    ↓
If confidence >= 0.60 and in session:
    Open trade (40 pip SL, 80 pip TP)
```

### Risk Management

- **Per trade**: 0.5% of equity
- **Daily limit**: 5 trades max
- **Daily loss**: Stop at 4% loss
- **Session**: Only 12:00-17:00 UTC
- **Position**: Max 1 concurrent trade

## Monitoring Checklist

Daily:
- [ ] Check EA is running (green smiley)
- [ ] Verify trades are within session hours
- [ ] Check no ONNX errors in logs
- [ ] Monitor daily P&L vs. limits

Weekly:
- [ ] Review win rate (should be >70%)
- [ ] Check trade frequency (~100/week)
- [ ] Verify SHORT/LONG ratio (~85/15)
- [ ] Compare to backtest metrics

Monthly:
- [ ] Full performance review
- [ ] Consider model retraining
- [ ] Adjust parameters if needed

## Support Files

- **Training results**: `TRAINING_RESULTS.md`
- **Architecture**: `M1_ARCHITECTURE.md`
- **Feature order**: `config/features_order.json`
- **Model metadata**: `python_training/models/model_metadata.json`

## Next Steps

1. ✅ Copy ONNX to MT5 Files folder
2. ✅ Copy EA to MT5 Experts folder
3. ✅ Compile EA in MetaEditor
4. ⏳ Run Strategy Tester (3 months)
5. ⏳ Verify results match backtest
6. ⏳ Deploy to demo account (30 days)
7. ⏳ Deploy to live account (small capital)
