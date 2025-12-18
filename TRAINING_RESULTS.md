# M1 Training Results

## Training Summary

### Dataset
- **Training**: 4,620,320 M1 bars (70%)
- **Validation**: 990,069 M1 bars (15%)
- **Test**: 990,069 M1 bars (15%)
- **Features**: 26 (16 M1 + 10 higher TF context)

### Class Distribution
- **SHORT (0)**: 51.0% train, 57.9% val
- **HOLD (1)**: 26.6% train, 14.0% val
- **LONG (2)**: 22.4% train, 28.1% val

### Model Performance
- **Accuracy**: 66.25%
- **Best iteration**: 468 (early stopping at 20 rounds)
- **Loss**: 0.744547 (validation multi_logloss)

### Per-Class Performance
```
              precision    recall  f1-score   support
SHORT            0.699     0.845     0.765    573,396
HOLD             0.611     0.281     0.385    138,687
LONG             0.568     0.477     0.519    277,986
```

### Confusion Matrix
```
           Predicted
           SHORT    HOLD    LONG
Actual 
SHORT     484,271  14,899  74,226
HOLD       72,927  39,016  26,744
LONG      135,357   9,967 132,662
```

### Top 10 Features (by importance)
1. **atr_14**: 10,575,558
2. **M15_position**: 5,790,479
3. **M5_position**: 2,702,868
4. **return_5**: 1,379,153
5. **H1_position**: 859,181
6. **return_60**: 438,492
7. **ema_10**: 356,113
8. **return_15**: 353,144
9. **hour_cos**: 345,332
10. **ema_50**: 337,315

## Backtest Results (Confidence >= 0.60)

### Overall Metrics
- **Total trades**: 579,280
- **Win rate**: 75.88%
- **Wins**: 439,545
- **Losses**: 139,735
- **Total P&L**: $295,742
- **Profit factor**: 6.29
- **Max drawdown**: $29.60

### Per Signal
- **SHORT**: 490,993 trades, 77.31% win rate
- **LONG**: 88,287 trades, 67.90% win rate

### Trade Statistics
- **Avg win**: $0.80 (80 pips)
- **Avg loss**: -$0.40 (40 pips)
- **Risk/Reward**: 1:2

## ONNX Export

### Model File
- **Path**: `python_training/models/lightgbm_xauusd.onnx`
- **Size**: 3,171.7 KB
- **Trees**: 1,404
- **Input shape**: [None, 26]
- **Output shape**: [None, 3] (probabilities for SHORT/HOLD/LONG)

### MT5 Deployment
- **Copied to**: `mt5_expert_advisor/Files/lightgbm_xauusd.onnx`
- **Feature order**: `config/features_order.json` (26 features)
- **EA file**: `mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5`

## Key Insights

1. **ATR is most important** - Volatility is the strongest predictor
2. **Higher TF position matters** - M15/M5/H1 position in range are top features
3. **SHORT bias** - Model predicts SHORT more often (77% win rate)
4. **High win rate** - 75.88% overall with 60% confidence threshold
5. **Good profit factor** - 6.29 indicates strong edge

## Next Steps

1. ✅ Training complete (66.25% accuracy)
2. ✅ Backtest complete (75.88% win rate, $295K profit)
3. ✅ ONNX export complete (3.2 MB file)
4. ⏳ Integrate ONNX in MT5 EA
5. ⏳ Test on Strategy Tester
6. ⏳ Deploy to demo account

## MT5 Integration Checklist

- [x] Feature order matches (26 features)
- [x] ONNX file copied to MT5 Files folder
- [x] EA calculates exact same features
- [ ] ONNX inference implemented in EA
- [ ] Strategy Tester validation
- [ ] Demo account testing

## Files Created

```
python_training/
├── models/
│   ├── lightgbm_xauusd.pkl (LightGBM model)
│   ├── lightgbm_xauusd.onnx (ONNX export)
│   └── model_metadata.json (metadata)
├── download_or_import_data.py
├── aggregate_to_m1.py
├── build_features_m1.py
├── train_lightgbm.py
├── evaluate_backtest.py
└── export_to_onnx.py

config/
├── paths.yaml
├── model_meta.json
└── features_order.json (26 features)

mt5_expert_advisor/
├── XAUUSD_NeuralBot_M1.mq5 (EA)
└── Files/
    └── lightgbm_xauusd.onnx (ONNX model)
```
