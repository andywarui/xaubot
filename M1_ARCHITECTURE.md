# M1-Based Architecture

## Overview
- **Training unit**: M1 bars (6.6M bars)
- **Prediction**: Every M1 bar close
- **Chart**: EA attached to M5 chart
- **Execution**: M1-driven (not M5-driven)

## Data Flow

```
JSONL Files (M1, M5, M15, H1, H4, D1)
    ↓
download_or_import_data.py
    ↓
Parquet files (xauusd_M1.parquet, xauusd_M5.parquet, ...)
    ↓
aggregate_to_m1.py
    ↓
xauusd_m1_clean.csv (6.6M M1 bars with time features)
    ↓
build_features_m1.py
    ↓
features_m1_train.parquet (4.6M M1 bars × 26 features)
features_m1_val.parquet (990K M1 bars × 26 features)
features_m1_test.parquet (990K M1 bars × 26 features)
    ↓
train_lightgbm.py (train on M1 bars)
    ↓
lightgbm_xauusd_m1.onnx
    ↓
MT5 EA (XAUUSD_NeuralBot_M1.mq5)
```

## Features (26 total)

### M1 Base Features (16)
1. body
2. body_abs
3. candle_range
4. close_position
5. return_1 (1 minute)
6. return_5 (5 minutes)
7. return_15 (15 minutes)
8. return_60 (1 hour)
9. tr (true range)
10. atr_14
11. rsi_14
12. ema_10
13. ema_20
14. ema_50
15. hour_sin
16. hour_cos

### Higher TF Context (10)
17. M5_trend
18. M5_position
19. M15_trend
20. M15_position
21. H1_trend
22. H1_position
23. H4_trend
24. H4_position
25. D1_trend
26. D1_position

## MT5 EA Behavior

### Chart Setup
- Attach EA to **M5 chart**
- EA internally uses **M1 data**

### Execution Logic
```cpp
void OnTick()
{
    // Called on every tick (M5 chart)
    
    if(!IsNewM1Bar())  // Check if new M1 bar closed
        return;        // Exit if no new M1 bar
    
    // New M1 bar detected:
    // 1. Calculate 26 M1 features (with M5/M15/H1/H4/D1 context)
    CalculateM1Features(features);
    
    // 2. Run ONNX model
    int signal = PredictSignal(features);
    
    // 3. Execute trade if signal is strong
    if(signal == LONG)
        OpenLongPosition();
    else if(signal == SHORT)
        OpenShortPosition();
}
```

### IsNewM1Bar() Helper
```cpp
bool IsNewM1Bar()
{
    datetime currentM1Time = iTime(_Symbol, PERIOD_M1, 0);
    if(currentM1Time != lastM1BarTime)
    {
        lastM1BarTime = currentM1Time;
        return true;  // New M1 bar closed
    }
    return false;  // Same M1 bar
}
```

## Key Points

1. **Training**: Model trained on 6.6M M1 bars
2. **Prediction**: New prediction every M1 bar close (every minute)
3. **Chart**: EA attached to M5 chart (for visual clarity)
4. **Data source**: All calculations use M1 series via `iTime(_Symbol, PERIOD_M1, ...)`
5. **Higher TF**: M5/M15/H1/H4/D1 context merged via nearest-past join
6. **Session filter**: Only trade 12:00-17:00 UTC (London-NY overlap)
7. **Labels**: Per M1 bar (80 pip TP, 40 pip SL, 12 bar forward)

## Pipeline Commands

```bash
# Step 1: Import JSONL to Parquet
python python_training/download_or_import_data.py

# Step 2: Prepare M1 with session filtering
python python_training/aggregate_to_m1.py

# Step 3: Build M1 features with higher TF context
python python_training/build_features_m1.py

# Step 4: Train LightGBM on M1 bars
python python_training/train_lightgbm.py

# Step 5: Export to ONNX
python python_training/export_to_onnx.py
```

## Results

- **M1 bars**: 6,600,530 total
- **Session bars**: 1,519,830 (23%)
- **Features**: 26 (16 M1 + 10 higher TF)
- **Train**: 4,620,320 M1 bars
- **Val**: 990,069 M1 bars
- **Test**: 990,069 M1 bars

## Next Steps

1. Train LightGBM on M1 features
2. Export to ONNX
3. Integrate ONNX in MT5 EA
4. Test on Strategy Tester (M5 chart, M1 execution)
5. Deploy to demo account
