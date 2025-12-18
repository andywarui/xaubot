# MT5 Strategy Tester - Execution Checklist

## Step 1: Copy Files to MT5

### Find Your MT5 Data Folder
Press `Ctrl+Shift+D` in MT5 or:
```
File → Open Data Folder
```

### Copy ONNX Model
**Source:**
```
C:\Users\KRAFTLAB\Desktop\TRADING\xaubot\mt5_expert_advisor\Files\lightgbm_xauusd.onnx
```

**Destination:**
```
[MT5_DATA_FOLDER]\MQL5\Files\lightgbm_xauusd.onnx
```

### Copy EA
**Source:**
```
C:\Users\KRAFTLAB\Desktop\TRADING\xaubot\mt5_expert_advisor\XAUUSD_NeuralBot_M1.mq5
```

**Destination:**
```
[MT5_DATA_FOLDER]\MQL5\Experts\XAUUSD_NeuralBot_M1.mq5
```

---

## Step 2: Compile EA

1. Open **MetaEditor** (press F4 in MT5)
2. Navigate to: `Experts → XAUUSD_NeuralBot_M1.mq5`
3. Press **F7** to compile
4. Check **Toolbox** tab at bottom

**Expected:**
```
0 error(s), 0 warning(s)
Compilation successful
```

**If errors:** Share the error messages

---

## Step 3: Strategy Tester Setup

1. Open **Strategy Tester** (Ctrl+R in MT5)
2. Configure:

```
Expert Advisor:  XAUUSD_NeuralBot_M1
Symbol:          XAUUSD
Period:          M5
Date:            Last 3 months (e.g., 2024.10.01 - 2025.01.01)
Model:           Every tick based on real ticks
Deposit:         50 USD
Leverage:        1:100
Optimization:    Disabled
```

3. **Inputs** tab:
```
RiskPercent:          0.5
ConfidenceThreshold:  0.60
MaxTradesPerDay:      5
MaxDailyLoss:         4.0
```

---

## Step 4: Run Test

1. Click **Start** button
2. Monitor **Journal** tab for:

**Expected messages:**
```
✓ "XAUUSD Neural Bot M1 - Initializing..."
✓ "ONNX model loaded successfully"
✓ "Input: [1, 26], Output: [1, 3]"
✓ "Attached to M5 chart, executes on M1 bars"
```

**Watch for errors:**
```
✗ "ERROR: Failed to load ONNX model"
✗ "ERROR: ONNX inference failed"
✗ "ERROR: Failed to calculate features"
```

3. Wait for test to complete (may take 5-30 minutes)

### Tick-data quality sanity check (highly recommended)

If you see lots of lines like `XAUUSD : YYYY.MM.DD 23:59 - no real ticks within a day`, your run is mostly using synthetic tick generation.

MT5 prints the tester log path at the end of a run (it looks like `...\logs\YYYYMMDD.log`). You can quantify tick coverage from that log:

```
python src\mt5_tester_log_analyze.py "C:\path\to\YYYYMMDD.log"
```

If the script reports that >50% of minutes are missing real ticks, prefer benchmarking on a shorter window where real ticks are mostly present.

---

## Step 5: Record Results

### From "Results" Tab

**Record these metrics:**
```
Total trades:        _______
Profit trades:       _______
Loss trades:         _______
Win rate (%):        _______
Profit factor:       _______
Total net profit:    _______
Max drawdown:        _______
Max drawdown (%):    _______
```

### From "Graph" Tab

**Visual checks:**
- [ ] Equity curve trending upward
- [ ] Drawdown stays below 10%
- [ ] Balance line shows consistent growth

### From "Report" Tab

1. Right-click → **Save as Report**
2. Save as: `strategy_tester_report.html`

---

## Step 6: Compare to Python Backtest

### Python Results (Reference)
```
Total trades:     579,280
Win rate:         75.88%
Profit factor:    6.29
Total P&L:        $295,742
Max drawdown:     $29.60
SHORT trades:     490,993 (77.31% win rate)
LONG trades:      88,287 (67.90% win rate)
```

### Expected MT5 Results (3 months)
```
Total trades:     ~1,500 - 2,000
Win rate:         70% - 80%
Profit factor:    >2.0
Max drawdown:     <10%
SHORT/LONG ratio: ~85/15
```

---

## Troubleshooting

### Issue: "ONNX model not found"
**Fix:**
1. Verify file exists: `MQL5\Files\lightgbm_xauusd.onnx`
2. Check filename (case-sensitive)
3. Restart MT5

### Issue: "ONNX inference failed"
**Fix:**
1. Check MT5 build >= 3802 (for ONNX support)
2. Verify ONNX file not corrupted (3.2 MB size)
3. Check Journal for specific error

### Issue: "No trades generated"
**Fix:**
1. Check date range includes 12:00-17:00 UTC hours
2. Lower ConfidenceThreshold to 0.50
3. Check XAUUSD data is available

### Issue: Compilation errors
**Fix:**
1. Update MT5 to latest version
2. Check Trade.mqh exists in MQL5\Include\Trade\
3. Share error message for specific help

---

## What to Share

After test completes, share:

1. **Journal messages** (first 20 lines)
2. **Results metrics** (from Step 5)
3. **Graph screenshot** (equity curve)
4. **Any errors** encountered

**Format:**
```
=== MT5 Strategy Tester Results ===

Date range: 2024.10.01 - 2025.01.01
Total trades: _______
Win rate: _______
Profit factor: _______
Max drawdown: _______

Journal:
[paste first 20 lines]

Issues:
[any errors or unexpected behavior]
```

---

## Next Steps (After Results)

### If results match Python backtest:
→ Deploy to demo account (30 days)

### If results differ significantly:
→ Debug discrepancies:
  - Check feature calculation
  - Verify ONNX output
  - Compare trade signals

### If errors occur:
→ Fix issues and retest

---

## Quick Command Reference

```
F4  - Open MetaEditor
F7  - Compile EA
Ctrl+R - Open Strategy Tester
Ctrl+Shift+D - Open Data Folder
```

---

**Ready to test!** Follow steps 1-5 and share results.
