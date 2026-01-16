# MT5 Strategy Tester - Model Deployment Guide

**Created**: January 1, 2026
**Purpose**: Deploy ONNX model to MT5 Strategy Tester directory
**Status**: Ready to execute

---

## The Problem

MT5 Strategy Tester uses a **different directory** than live trading:

- **Live Trading**: `Terminal\<ID>\MQL5\Files\`
- **Strategy Tester**: `Tester\<ID>\Agent-127.0.0.1-3000\MQL5\Files\`

The EA successfully compiles, but the model file is not in the Tester-specific directory.

---

## The Solution

### Option 1: Run Batch Script (EASIEST)

**On Windows**:
1. Navigate to: `C:\Users\KRAFTLAB\Documents\xaubot`
2. Double-click: `DEPLOY_TO_TESTER.bat`
3. Wait for "Deployment Complete!" message
4. Press Enter

### Option 2: Run PowerShell Script (RECOMMENDED)

**On Windows**:
1. Right-click `DEPLOY_TO_TESTER.ps1`
2. Select "Run with PowerShell"
3. If execution policy blocks it, run this first:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
4. Then run: `.\DEPLOY_TO_TESTER.ps1`

### Option 3: Manual Copy (FALLBACK)

**If scripts don't work**:

1. Copy this file:
   ```
   C:\Users\KRAFTLAB\Documents\xaubot\MT5_XAUBOT\Files\lightgbm_xauusd.onnx
   ```

2. To this location:
   ```
   C:\Users\KRAFTLAB\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\MQL5\Files\
   ```

3. Create the `Files` folder if it doesn't exist

---

## What the Scripts Do

1. **Verify source file exists** (9,596 bytes)
2. **Create Tester directory** (if missing)
3. **Copy model file** to Tester location
4. **Verify successful copy** (check size and date)

---

## Expected Output

```
========================================
MT5 Strategy Tester Model Deployment
========================================

Source: C:\Users\KRAFTLAB\Documents\xaubot\MT5_XAUBOT\Files\lightgbm_xauusd.onnx
Destination: C:\Users\KRAFTLAB\...\Tester\...\MQL5\Files

[1/3] Checking source file...
       Size: 9596 bytes
       OK

[2/3] Creating Tester directory...
       Created: C:\Users\KRAFTLAB\...\Files

[3/3] Copying model file...
       SUCCESS!

Verification:
  File: lightgbm_xauusd.onnx
  Location: C:\Users\KRAFTLAB\...\Files
  Size: 9596 bytes
  Date: 2026-01-01 08:00:00

========================================
Deployment Complete!
========================================

Next steps:
1. Open MT5 MetaEditor
2. Compile XAUUSD_Neural_Bot_FIXED.mq5
3. Open Strategy Tester
4. Select XAUUSD_Neural_Bot_FIXED
5. Run backtest

The model should now load successfully!
========================================
```

---

## After Deployment

### Test in Strategy Tester

**Settings**:
- EA: `XAUUSD_Neural_Bot_FIXED`
- Symbol: `XAUUSD`
- Period: `M1`
- Dates: `2023.01.01` - `2025.01.01`
- Model: `Every tick (based on real ticks)`
- Optimization: OFF (first run)

### Expected Initialization Log

You should see:

```
========================================
XAUUSD Neural Bot v3.0 (FIXED ONNX)
Following MQL5 Best Practices
========================================
✓ Model loaded from: C:\Users\KRAFTLAB\...\Tester\...\Files\lightgbm_xauusd.onnx
Setting ONNX input/output shapes...
✓ ONNX shapes configured: Input[1,26] → Output[1,3]
Testing ONNX inference with dummy data...
Test prediction output:
  Class 0: 0.334
  Class 1: 0.333
  Class 2: 0.333
  Sum: 1.000 (should be ~1.0 for probabilities)
✓ ONNX model test passed!
✓ All indicators initialized
========================================
Bot initialized successfully!
========================================
```

### If Model STILL Doesn't Load

**Check**:
1. File exists in Tester directory (verify manually)
2. File size is 9,596 bytes (not 0 bytes)
3. EA is using correct model name: `lightgbm_xauusd.onnx`
4. Agent path matches: `Agent-127.0.0.1-3000` (might change)

**Fallback EA Paths** (EA tries 5 locations):
```mql5
string model_paths[] = {
    InpModelPath,                                           // "lightgbm_xauusd.onnx"
    "Files\\" + InpModelPath,                              // "Files\lightgbm_xauusd.onnx"
    "MQL5\\Files\\" + InpModelPath,                        // "MQL5\Files\lightgbm_xauusd.onnx"
    TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + InpModelPath,
    TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + InpModelPath
};
```

---

## Troubleshooting

### Script Error: "Source file not found"

**Cause**: Project folder not at expected location
**Fix**: Edit script and update `$ProjectDir` or `PROJECT_DIR`

### Script Error: "Access denied"

**Cause**: Need admin rights to write to AppData
**Fix**: Right-click script → "Run as Administrator"

### Model Size Shows 0 Bytes

**Cause**: Git LFS not installed or file not pulled
**Fix**:
```bash
cd C:\Users\KRAFTLAB\Documents\xaubot
git lfs install
git lfs pull
```

### Different Agent Path

**Cause**: MT5 uses different agent ID
**Fix**: Check Strategy Tester log for actual path, update script

Example:
```
Path 4 failed (5002): C:\...\Tester\...\Agent-127.0.0.1-3001\MQL5\Files\...
                                            ^^^^^^^^^^^^^^^^
                                            Use this path
```

---

## Next Steps After Successful Deployment

1. ✅ **Run 2-year backtest** (2023-2025)
2. ✅ **Analyze results** (trades, win rate, profit)
3. ✅ **Compare to Python backtest** (+472% target)
4. ⏳ **Optimize parameters** if needed
5. ⏳ **Deploy to demo account** for paper trading
6. ⏳ **Monitor with Python dashboard**

---

## Related Files

- `XAUUSD_Neural_Bot_FIXED.mq5` - The EA (compiles successfully)
- `lightgbm_xauusd.onnx` - Neural network model (9.4KB)
- `MQL5_ARTICLES_INSIGHTS.md` - Why these steps are needed
- `COMPREHENSIVE_PROJECT_ANALYSIS.md` - Full project status

---

**Status**: Ready to deploy
**Action Required**: Run deployment script on Windows machine
**Expected Time**: 30 seconds
**Success Indicator**: Model loads, EA initializes without errors
