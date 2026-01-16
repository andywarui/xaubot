# Python DLL for MT5 - LightGBM Integration

**Created**: January 2, 2026  
**Purpose**: Load original +472% LightGBM model in MT5 via Python DLL  
**Status**: Ready for testing and compilation

---

## Files

| File | Purpose |
|------|---------|
| `lightgbm_predictor.py` | Core DLL wrapper with ctypes interface |
| `test_local.py` | Test suite (run before compilation) |
| `compile_dll.py` | Automated PyInstaller compilation |

---

## Quick Start

### Step 1: Install Dependencies
```powershell
pip install lightgbm numpy pyinstaller
```

### Step 2: Test Locally
```powershell
cd C:\Users\KRAFTLAB\Documents\xaubot\python_dll
python test_local.py
```

Expected output:
```
  [✓ PASS] Model file exists
  [✓ PASS] Model loading
  [✓ PASS] Basic prediction
  ...
  🎉 All tests passed! Ready to compile DLL.
```

### Step 3: Compile DLL
```powershell
python compile_dll.py
```

### Step 4: Deploy to MT5
```powershell
# Copy DLL
Copy-Item "dist\lightgbm_predictor.dll" "C:\Program Files\MetaTrader 5\MQL5\Libraries\"

# Copy model
Copy-Item "..\python_training\models\lightgbm_xauusd.pkl" "C:\Program Files\MetaTrader 5\MQL5\Files\"
```

---

## API Reference

### `init_model(model_path: str) -> int`
Load the LightGBM model from pickle file.

**Parameters:**
- `model_path`: Absolute path to `lightgbm_xauusd.pkl`

**Returns:**
- `1` = Success
- `0` = Failed

### `predict_signal(features: list) -> int`
Predict trading signal from 26 features.

**Parameters:**
- `features`: List of 26 float values in exact order

**Returns:**
- `0` = SHORT
- `1` = HOLD  
- `2` = LONG
- `-1` = Error

### `predict_with_probabilities(features: list) -> tuple`
Get prediction with class probabilities.

**Returns:**
- `(signal, prob_short, prob_hold, prob_long)`

### `get_feature_names() -> list`
Get ordered list of 26 feature names.

### `cleanup_model()`
Free model resources.

---

## Feature Order (26 features)

The features must be passed in this exact order:

| # | Feature | Description |
|---|---------|-------------|
| 1 | close | Current close price |
| 2 | returns | 1-bar return |
| 3 | returns_2 | 2-bar return |
| 4 | returns_5 | 5-bar return |
| 5 | returns_10 | 10-bar return |
| 6 | rsi_14 | RSI(14) |
| 7 | rsi_28 | RSI(28) |
| 8 | atr_14 | ATR(14) |
| 9 | atr_ratio | ATR ratio |
| 10 | ema_ratio_8_21 | EMA(8)/EMA(21) |
| 11 | ema_ratio_13_55 | EMA(13)/EMA(55) |
| 12 | macd | MACD line |
| 13 | macd_signal | MACD signal |
| 14 | macd_hist | MACD histogram |
| 15 | bb_position | Bollinger Band position |
| 16 | bb_width | Bollinger Band width |
| 17 | price_position | Price position in range |
| 18 | high_low_range | High-Low range |
| 19 | close_to_high | Distance to high |
| 20 | close_to_low | Distance to low |
| 21 | hour_sin | Hour sine encoding |
| 22 | hour_cos | Hour cosine encoding |
| 23 | day_sin | Day sine encoding |
| 24 | day_cos | Day cosine encoding |
| 25 | is_london_session | London session flag |
| 26 | is_ny_session | NY session flag |

---

## Model Details

- **Type**: LightGBM Classifier
- **File**: `python_training/models/lightgbm_xauusd.pkl`
- **Size**: 5.1 MB
- **Classes**: 0 (SHORT), 1 (HOLD), 2 (LONG)
- **Backtest Performance**: +472.81% return (2022-2024)
