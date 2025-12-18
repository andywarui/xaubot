# XAUBOT Project Status & Next Steps
**Generated:** December 18, 2025  
**Last Update:** Hybrid Features Fixed

---

## 🎯 Project Overview

**XAUBOT** is a multi-timeframe transformer-based trading bot for XAU/USD (Gold) that deploys to MetaTrader 5 via ONNX.

### Architecture Stack
```
┌─────────────────────────────────────────────────┐
│  MT5 Expert Advisor (MQL5)                      │
│  ├─ ONNX Runtime integration                    │
│  ├─ 26 feature calculations per M1 bar          │
│  └─ Risk management & execution                 │
├─────────────────────────────────────────────────┤
│  ONNX Models                                     │
│  ├─ LightGBM Hybrid (classification)            │
│  └─ Multi-TF Transformer (feature extraction)   │
├─────────────────────────────────────────────────┤
│  Python Training Pipeline                        │
│  ├─ Multi-TF Transformer (130 features, 72.9%)  │
│  ├─ Hybrid Feature Generation                   │
│  └─ LightGBM Hybrid Classifier                  │
└─────────────────────────────────────────────────┘
```

---

## ✅ Completed Milestones

### 1. Data Pipeline ✓
- **6.6M M1 bars** (5 years of XAU/USD data)
- Multi-timeframe parquet files (M1, M5, M15, H1, D1)
- Session filtering (London-NY overlap: 12:00-17:00 UTC)
- Train/Val/Test split: 70/15/15

### 2. Multi-TF Transformer ✓
- **Model:** `multi_tf_transformer_price.pth`
- **Architecture:** 130 features (26×5 TFs), 30 timesteps, 128 d_model
- **Performance:** 72.9% direction accuracy
- **Training:** 4.6M bars, regression target (% price change)
- **Status:** ✅ Trained and validated

### 3. Hybrid Feature Preparation ✓ (JUST FIXED)
- **Script:** `prepare_hybrid_features_multi_tf.py`
- **Output:** `hybrid_features_{train,val,test}.parquet`
- **Features:** 27 (26 base + 1 multi_tf_signal from transformer)
- **Labels:** Correct 3-class distribution (SHORT/HOLD/LONG)
  - Train: 2.3M SHORT, 1.2M HOLD, 1.0M LONG
  - Val/Test: Proper distribution maintained
- **Fixed Issues:**
  - ✅ Data loading now uses M1-based alignment (not broken .npy files)
  - ✅ Scaler bug fixed (was GradScaler, now MinMaxScaler)
  - ✅ Labels now have correct 0/1/2 distribution (was all 1s)

---

## 🔄 Current State: Ready for Hybrid LightGBM Training

### What Just Happened
We successfully fixed the hybrid feature preparation pipeline:

**Problem Identified:**
1. Original `build_5tf_from_feature_parquets.py` only used M5 data
2. M5 had placeholder labels (all 1s) from feature generation
3. `prepare_hybrid_features_multi_tf.py` was loading broken .npy files

**Solution Applied:**
1. Rewrote `prepare_hybrid_features_multi_tf.py` to:
   - Load data using same method as `train_multi_tf_transformer.py`
   - Use M1 labels (correct 3-class distribution)
   - Fit fresh MinMaxScaler on training data
   - Generate transformer predictions as `multi_tf_signal` feature
2. Generated hybrid feature parquet files with correct labels

**Files Created:**
```
data/processed/
├── hybrid_features_train.parquet  (4.6M rows, 30 columns)
├── hybrid_features_val.parquet    (990K rows, 30 columns)
└── hybrid_features_test.parquet   (990K rows, 30 columns)
```

**Feature Schema:**
- `time` (timestamp)
- `close` (M1 close price)
- `label` (0=SHORT, 1=HOLD, 2=LONG)
- `multi_tf_signal` (transformer prediction)
- 26 base features (body, atr_14, rsi_14, M5_trend, etc.)

---

## 🚀 Next Steps: Hybrid LightGBM Training

### Immediate Action
Run the hybrid LightGBM training on the pod:

```bash
cd /workspace/xaubot
python python_training/train_lightgbm_hybrid.py
```

### Expected Outcome
This will train a LightGBM classifier using:
- **Input:** 27 features (26 base + multi_tf_signal)
- **Target:** 3-class classification (SHORT/HOLD/LONG)
- **Expected Performance:** ~66-70% accuracy (baseline was 66.25%)
- **Hypothesis:** Transformer signal should boost accuracy

### What the Script Does
1. Loads `hybrid_features_{train,val,test}.parquet`
2. Trains LightGBM with multi_tf_signal as additional feature
3. Evaluates on validation set
4. Saves model to `models/lightgbm_hybrid.pkl`
5. Exports to ONNX: `models/lightgbm_hybrid.onnx`

---

## 📋 Full Project Pipeline

### Phase 1: Data Preparation ✅
- [x] Download/import JSONL data
- [x] Aggregate to M1 clean CSV
- [x] Build features for M1, M5, M15, H1, D1
- [x] Create train/val/test parquet files

### Phase 2: Multi-TF Transformer ✅
- [x] Train transformer on 130 features (72.9% accuracy)
- [x] Save model, scaler, config
- [x] Validate direction accuracy

### Phase 3: Hybrid Feature Generation ✅
- [x] Load trained transformer
- [x] Generate multi_tf_signal predictions
- [x] Merge with base features
- [x] Fix label distribution
- [x] Save hybrid parquet files

### Phase 4: Hybrid LightGBM Training ⏳ NEXT
- [ ] Train LightGBM on 27 hybrid features
- [ ] Validate performance (target: >66.25% baseline)
- [ ] Export to ONNX
- [ ] Validate ONNX parity

### Phase 5: ONNX Integration & MT5 ⏳
- [ ] Fix MT5 EA to handle 2-output ONNX (label + probabilities)
- [ ] Copy hybrid ONNX to `mt5_expert_advisor/Files/`
- [ ] Implement feature parity validation
- [ ] Test on MT5 Strategy Tester

### Phase 6: Backtesting & Validation ⏳
- [ ] Run MT5 Strategy Tester (12 months OOS)
- [ ] Compare MT5 vs Python backtest results
- [ ] Validate win rate, profit factor, drawdown

### Phase 7: Deployment ⏳
- [ ] 30-day demo account testing
- [ ] Monitor performance vs expectations
- [ ] Go live with conservative config

---

## 🔧 Key Technical Details

### Feature Engineering (26 Base Features)
**M1 Base (16):**
- Candle: body, body_abs, candle_range, close_position
- Returns: return_1, return_5, return_15, return_60
- Indicators: tr, atr_14, rsi_14, ema_10, ema_20, ema_50
- Time: hour_sin, hour_cos

**Higher TF Context (10):**
- M5/M15/H1/H4/D1: trend, position

### Multi-TF Transformer Architecture
- **Input:** [batch, 30 timesteps, 130 features]
- **Encoder:** 3-layer transformer (128 d_model, 8 heads)
- **Output:** % price change prediction (regression)
- **Performance:** 72.9% direction accuracy

### Hybrid Approach
- Use transformer predictions as additional feature
- LightGBM learns when to trust transformer vs base features
- Expected benefit: Better generalization, reduced overfitting

---

## 📊 Performance Metrics

### Multi-TF Transformer (Current Best)
- **Direction Accuracy:** 72.9%
- **Validation Loss:** 0.00123 (MSE on % price change)
- **Training:** 30 epochs, early stopping

### LightGBM Baseline (Previous)
- **Accuracy:** 66.25%
- **Win Rate (backtest):** 75.88%
- **Profit Factor:** 6.29
- **Per-Class:** SHORT 77.3%, HOLD 28.1%, LONG 47.7%

### Target for Hybrid Model
- **Accuracy:** >66.25% (should exceed baseline)
- **Win Rate:** >75%
- **Profit Factor:** >6.0
- **Goal:** Improve HOLD and LONG classification

---

## 🐛 Known Issues & Fixes

### Issue 1: Data Leakage (RESOLVED)
- **Problem:** Loss went to 0.000000 during initial training
- **Cause:** M5/M15/H1/D1 parquets had placeholder labels (all 1s)
- **Fix:** Use M1 labels for all data

### Issue 2: Broken .npy Files (RESOLVED)
- **Problem:** `X_5tf_train.npy` had wrong shape (982K, 27) instead of (4.6M, 130)
- **Cause:** `build_5tf_from_feature_parquets.py` only loaded M5
- **Fix:** Rewrote data loading to use M1 base with higher TF merging

### Issue 3: Scaler Type Mismatch (RESOLVED)
- **Problem:** `multi_tf_scaler.pkl` contained GradScaler (PyTorch amp) instead of MinMaxScaler
- **Cause:** Wrong object saved during training
- **Fix:** Refit MinMaxScaler on training data and save correctly

### Issue 4: MT5 ONNX Integration (PENDING)
- **Problem:** MT5 EA expects 1 output, ONNX model has 2 (label + probabilities)
- **Status:** Need to update EA to use `OnnxRunBatch()` or handle 2 outputs
- **Documentation:** See `QUICKFIX_MT5_ONNX.md`

---

## 📁 Critical Files

### Python Training
```
python_training/
├── train_multi_tf_transformer.py      # Multi-TF transformer training (DONE)
├── prepare_hybrid_features_multi_tf.py # Hybrid feature generation (JUST FIXED)
├── train_lightgbm_hybrid.py           # Hybrid LightGBM training (NEXT)
└── export_onnx_mt5.py                 # ONNX export for MT5
```

### Models
```
python_training/models/
├── multi_tf_transformer_price.pth     # Transformer weights (72.9%)
├── multi_tf_scaler.pkl                # MinMaxScaler (FIXED)
├── multi_tf_config.json               # Model config
└── (pending) lightgbm_hybrid.pkl      # Hybrid LightGBM (NEXT)
```

### Data
```
data/processed/
├── features_m1_{train,val,test}.parquet       # M1 base features (4.6M rows)
├── features_m5_{train,val,test}.parquet       # M5 features (982K rows)
├── features_m15_{train,val,test}.parquet      # M15 features
├── features_h1_{train,val,test}.parquet       # H1 features
├── features_d1_{train,val,test}.parquet       # D1 features
└── hybrid_features_{train,val,test}.parquet   # Hybrid features (JUST CREATED)
```

### MT5 Integration
```
mt5_expert_advisor/
├── XAUUSD_NeuralBot_M1.mq5            # EA code (needs ONNX fix)
└── Files/
    ├── lightgbm_xauusd.onnx           # Current model (needs update)
    └── config/
        └── model_config.json          # Feature order
```

---

## 💡 Workflow Best Practices (Established)

### Git Workflow
```bash
# After any meaningful change:
git add -A
git commit -m "Short description of what changed"
git push origin main
```

### Good Commit Message Format
- Start with action verb: Fix, Add, Update, Remove, Refactor
- Keep first line under 50 chars
- Add details after blank line if needed

### Example Commits
```
Fix hybrid feature preparation with correct labels and scaler
Add ONNX export for MT5 integration
Update training hyperparameters for better accuracy
Refactor data loading to use merge_asof alignment
```

---

## 🎯 Success Criteria

### Before Demo Deployment
- [ ] Hybrid LightGBM accuracy >66.25%
- [ ] ONNX export validated (Python vs ONNX match)
- [ ] MT5 EA compiles without errors
- [ ] Feature parity check passes (±2% tolerance)
- [ ] MT5 backtest shows ~70% win rate
- [ ] Profit factor >1.5

### Before Live Deployment
- [ ] 30-day demo account validation
- [ ] Performance matches expectations
- [ ] No unexpected model behavior
- [ ] Risk management tested (max DD, daily loss limits)

---

## 📞 Current Session Summary

**What We Accomplished:**
1. ✅ Identified data leakage (labels all 1s)
2. ✅ Fixed hybrid feature preparation pipeline
3. ✅ Generated correct hybrid features with proper label distribution
4. ✅ Saved corrected scaler for future use
5. ✅ Established Git workflow for the project
6. ✅ Deep dive into project architecture and roadmap

**What's Next:**
1. ⏳ Train hybrid LightGBM model
2. ⏳ Export to ONNX
3. ⏳ Fix MT5 EA for 2-output ONNX
4. ⏳ Run MT5 Strategy Tester validation

**Command to Run Next:**
```bash
python python_training/train_lightgbm_hybrid.py
```

---

## 📚 Documentation Files
- `README.md` - Project overview and quick start
- `M1_ARCHITECTURE.md` - M1-based execution architecture
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `TRAINING_RESULTS.md` - Model performance metrics
- `MT5_ONNX_INTEGRATION.md` - ONNX integration guide
- `QUICKFIX_MT5_ONNX.md` - MT5 EA fix for 2-output ONNX
- `TODO_MT5.md` - MT5 development checklist
- `DEPLOYMENT_CHECKLIST.md` - Go/No-Go deployment criteria
- `docs/ROADMAP.md` - Development roadmap

---

**Status:** 🟢 Ready for Hybrid LightGBM Training  
**Progress:** ~65% complete (Data ✓, Transformer ✓, Hybrid Features ✓)  
**Next Milestone:** Hybrid LightGBM → ONNX → MT5 Integration
