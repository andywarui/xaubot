# 🏆 XAUBOT ULTIMATE DEVELOPMENT PLAN

## Complete Roadmap: From Model to Live Trading

**Project:** xaubot - Neural XAU/USD Trading Bot  
**Created:** December 2024  
**Status:** Phase 1 Ready to Begin

---

## 📋 Executive Summary

| Phase | Name | Duration | Key Deliverable |
|-------|------|----------|-----------------|
| 1 | Model Optimization | 2-3 days | Balanced model with >60% recall all classes |
| 2 | Comprehensive Backtesting | 2-3 days | Full validation report with confidence intervals |
| 3 | MT5 Integration | 3-4 days | Working Expert Advisor |
| 4 | Paper Trading | 1-2 weeks | Live validation on demo account |
| 5 | Go-Live | Ongoing | Production deployment with monitoring |

**Total Estimated Time: 3-4 weeks**

---

## 📊 Current State

### Completed Work
- ✅ Multi-timeframe data pipeline (M1, M5, M15, H1, D1)
- ✅ Transformer model trained (72.9% direction accuracy)
- ✅ Hybrid features generated with correct labels
- ✅ LightGBM hybrid model (66.3% test accuracy)
- ✅ Git LFS setup for large files
- ✅ GitHub repository synced

### Current Model Performance
```
Transformer (Regression):
├── Direction Accuracy: 72.9%
└── Best Val Loss: 0.001227

Hybrid LightGBM (Classification):
├── Test Accuracy: 66.32%
├── SHORT Recall: 89% ✅
├── HOLD Recall:  33% ❌ (needs improvement)
└── LONG Recall:  32% ❌ (needs improvement)

Top Feature: multi_tf_signal (8.4M importance)
```

---

## 🎯 PHASE 1: MODEL OPTIMIZATION

### Goal
Achieve balanced performance across all classes

```
Current State:
├── SHORT Recall: 89% ✅
├── HOLD Recall:  33% ❌
└── LONG Recall:  32% ❌

Target State:
├── SHORT Recall: >70%
├── HOLD Recall:  >55%
└── LONG Recall:  >55%
└── Overall Accuracy: >65%
```

### 1.1 Class Balancing Strategies

| Method | Description | Implementation |
|--------|-------------|----------------|
| Sample Weighting | Inverse frequency weights: w_i = N / (k × n_i) | LightGBM `class_weight` |
| SMOTE Oversampling | Generate synthetic HOLD/LONG samples | `imblearn.over_sampling` |
| Undersampling | Reduce SHORT to match LONG count | `imblearn.under_sampling` |
| Focal Loss | Down-weight easy examples | Custom loss function |
| Threshold Tuning | Optimal probability thresholds per class | Grid search |

### 1.2 Feature Engineering Improvements

```
New Features to Add:
├── Momentum Features
│   ├── ROC (Rate of Change) - 5, 10, 20 periods
│   ├── Momentum oscillator
│   └── Price acceleration
│
├── Volatility Features  
│   ├── Bollinger Band %B
│   ├── Keltner Channel position
│   └── ATR ratio (current/average)
│
├── Market Microstructure
│   ├── Bid-Ask spread proxy
│   ├── Volume profile
│   └── VWAP distance
│
├── Multi-Timeframe Enhancements
│   ├── TF agreement score (how many TFs agree on direction)
│   ├── TF momentum alignment
│   └── Cross-TF divergence signals
│
└── Time-Based Features
    ├── Session indicators (Asian/London/NY)
    ├── Day of week
    └── Distance to major news events
```

### 1.3 Ensemble Architecture

```
   Input Features (130)
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │ Transformer │    │  LightGBM   │    │   XGBoost   │
  │  (72.9%)    │    │   (66.3%)   │    │   (new)     │
  └─────────────┘    └─────────────┘    └─────────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Meta-Learner   │
                   │ (Stacking/Vote) │
                   └─────────────────┘
                            │
                            ▼
                   Final Prediction
                   (SHORT/HOLD/LONG)
```

### 1.4 Phase 1 Deliverables

```
python_training/
├── train_lightgbm_balanced.py      # Class-weighted training
├── train_xgboost_hybrid.py         # XGBoost for ensemble
├── train_ensemble.py               # Meta-learner stacking
├── optimize_thresholds.py          # Per-class threshold tuning
├── feature_engineering_v2.py       # Enhanced features
└── models/
    ├── ensemble_meta.pkl
    ├── lightgbm_balanced.txt
    ├── xgboost_hybrid.json
    └── optimal_thresholds.json
```

### 1.5 Phase 1 Success Criteria

- [ ] All class recalls > 55%
- [ ] Overall accuracy > 65%
- [ ] No single class dominates predictions
- [ ] Feature importance shows multi_tf_signal still top
- [ ] Ensemble outperforms individual models

---

## 🔬 PHASE 2: COMPREHENSIVE BACKTESTING

### Goal
Validate model with statistical rigor

### 2.1 Backtesting Methods Overview

| Method | Purpose | Priority |
|--------|---------|----------|
| Walk-Forward Optimization | Prevent overfitting | 🔴 High |
| Monte Carlo Simulation | Confidence intervals | 🔴 High |
| Historical Stress Test | Crash resilience | 🔴 High |
| Regime-Based Analysis | Understand when it works | 🟡 Medium |
| Reality Gap Testing | Real-world viability | 🟡 Medium |
| CPCV | Academic rigor | 🟢 Nice-to-have |
| Deflated Sharpe | Publication-ready | 🟢 Nice-to-have |

### 2.2 Walk-Forward Optimization

```
Data Timeline (2019-2024):
═══════════════════════════════════════════════════════════════════

Fold 1: Train [2019──────2021] Test [2021H1] → Metrics₁
Fold 2: Train [2019────────2021H1] Test [2021H2] → Metrics₂  
Fold 3: Train [2019──────────2021] Test [2022H1] → Metrics₃
Fold 4: Train [2020────────2022H1] Test [2022H2] → Metrics₄
Fold 5: Train [2020──────────2022] Test [2023H1] → Metrics₅
Fold 6: Train [2021────────2023H1] Test [2023H2] → Metrics₆
Fold 7: Train [2021──────────2023] Test [2024H1] → Metrics₇

Final WFO Score = Mean(Metrics₁...₇) ± Std
═══════════════════════════════════════════════════════════════════
```

### 2.3 Monte Carlo Simulation

```
Parameters:
├── Simulations:        10,000 paths
├── Methods:
│   ├── Trade shuffling (random order)
│   ├── Bootstrap resampling (with replacement)
│   ├── Return perturbation (±5% noise)
│   └── Drawdown path simulation
│
└── Metrics Calculated:
    ├── Total Return:     Mean, 5th%, 95th%
    ├── Max Drawdown:     Mean, 5th%, 95th%
    ├── Sharpe Ratio:     Mean, 5th%, 95th%
    ├── Win Rate:         Mean, 5th%, 95th%
    ├── Profit Factor:    Mean, 5th%, 95th%
    ├── Recovery Factor:  Mean, 5th%, 95th%
    └── Risk of Ruin:     P(Drawdown > 50%)
```

### 2.4 Historical Stress Testing

| Event | Date | Gold Movement | Test Criteria |
|-------|------|---------------|---------------|
| COVID Crash | Mar 2020 | -$200 then +$400 | Survive |
| Gold ATH Run | Aug 2020 | $1700 → $2075 (+22%) | Capture upside |
| Flash Crash | Aug 2021 | -$100 in minutes | Limit losses |
| Ukraine Invasion | Feb 2022 | +$150 in days | Capture spike |
| Fed Rate Hikes | 2022 | $2050 → $1620 (-21%) | Survive drawdown |
| Banking Crisis | Mar 2023 | +$200 in 2 weeks | Capture move |
| Israel-Hamas | Oct 2023 | +$150 spike | React to news |
| 2024 ATH | Mar 2024 | New highs >$2200 | Participate |

**Criteria:** Survive all events with <30% drawdown

### 2.5 Regime-Based Analysis

```
Regime Detection (using ADX, ATR, Trend):

┌─────────────────┬─────────────────┬─────────────────┐
│   TRENDING UP   │  TRENDING DOWN  │    RANGING      │
│   ADX > 25      │   ADX > 25      │   ADX < 20      │
│   Price > EMA   │   Price < EMA   │   Choppy        │
└─────────────────┴─────────────────┴─────────────────┘

┌─────────────────────┬─────────────────────┐
│   HIGH VOLATILITY   │   LOW VOLATILITY    │
│   ATR > 1.5x avg    │   ATR < 0.5x avg    │
└─────────────────────┴─────────────────────┘

Report Metrics PER REGIME:
├── Win Rate
├── Profit Factor
├── Average Trade
└── Trade Count
```

### 2.6 Reality Gap Testing

| Level | Added Friction | XAUUSD Typical |
|-------|---------------|----------------|
| 0 | Baseline (perfect) | - |
| 1 | + Spread | $0.20-0.30 |
| 2 | + Slippage | $0.00-0.10 random |
| 3 | + Commission | $7 per lot RT |
| 4 | + Swap | Overnight costs |
| 5 | + Partial Fills | 80% fill rate |
| 6 | + Latency | 100-500ms |
| 7 | + Weekend Gaps | Fri→Mon gaps |

**Criteria:** Still profitable at Level 5

### 2.7 Phase 2 Deliverables

```
python_training/backtesting/
├── __init__.py
├── base_backtest.py            # Core backtesting engine
├── walk_forward.py             # Walk-forward optimization
├── monte_carlo.py              # Monte Carlo simulation
├── stress_test.py              # Historical stress testing
├── regime_analysis.py          # Regime-based performance
├── reality_gap.py              # Execution cost simulation
├── metrics.py                  # All metric calculations
├── visualization.py            # Charts and plots
├── comprehensive_report.py     # Generate full PDF report
└── results/
    ├── wfo_results.json
    ├── monte_carlo_results.json
    ├── stress_test_results.json
    ├── regime_analysis_results.json
    ├── reality_gap_results.json
    └── comprehensive_report.pdf
```

### 2.8 Phase 2 Success Criteria

- [ ] Walk-Forward: Consistent positive returns across all folds
- [ ] Monte Carlo: 95% CI for Sharpe > 0.5
- [ ] Stress Test: Survive all events with <30% drawdown
- [ ] Regime: Profitable in at least 3/5 regimes
- [ ] Reality Gap: Still profitable at Level 5
- [ ] Risk of Ruin: < 5% chance of 50% drawdown

---

## 🖥️ PHASE 3: MT5 INTEGRATION

### Goal
Deploy validated model to MetaTrader 5

### 3.1 Architecture Overview

```
                      MetaTrader 5 Terminal
  ┌───────────────────────────────────────────────────────────┐
  │                                                            │
  │   XAUUSD_NeuralBot.mq5 (Expert Advisor)                   │
  │   ├── OnInit()     → Load ONNX models, config             │
  │   ├── OnTick()     → Main logic loop                      │
  │   ├── OnTimer()    → Periodic model inference             │
  │   └── OnDeinit()   → Cleanup                              │
  │                                                            │
  │   ┌─────────────────────────────────────────────────┐     │
  │   │           Feature Calculator Module              │     │
  │   │  ┌─────────────────────────────────────────┐    │     │
  │   │  │ Get OHLCV from 5 timeframes (M1-D1)     │    │     │
  │   │  │ Calculate 26 technical indicators       │    │     │
  │   │  │ Apply MinMax scaling                    │    │     │
  │   │  │ Format input tensor                     │    │     │
  │   │  └─────────────────────────────────────────┘    │     │
  │   └─────────────────────────────────────────────────┘     │
  │                         │                                  │
  │                         ▼                                  │
  │   ┌─────────────────────────────────────────────────┐     │
  │   │              ONNX Inference Pipeline             │     │
  │   │  ┌──────────────────┐   ┌──────────────────┐    │     │
  │   │  │ Transformer ONNX │──▶│ LightGBM ONNX    │    │     │
  │   │  │ (multi_tf_signal)│   │ (final predict)  │    │     │
  │   │  └──────────────────┘   └──────────────────┘    │     │
  │   └─────────────────────────────────────────────────┘     │
  │                         │                                  │
  │                         ▼                                  │
  │   ┌─────────────────────────────────────────────────┐     │
  │   │              Trade Execution Module              │     │
  │   │  ├── Position sizing (risk management)          │     │
  │   │  ├── Entry logic (signal thresholds)            │     │
  │   │  ├── Exit logic (TP/SL/trailing)                │     │
  │   │  └── Order management                           │     │
  │   └─────────────────────────────────────────────────┘     │
  │                                                            │
  └───────────────────────────────────────────────────────────┘
```

### 3.2 ONNX Export Pipeline

```
Python Side:
═══════════════════════════════════════════════════════════════════

1. Export Transformer to ONNX
   ├── Input: [batch, 30, 130] float32
   ├── Output: [batch, 1] float32 (multi_tf_signal)
   └── Optimizations: fp16, graph optimization

2. Export LightGBM to ONNX  
   ├── Input: [batch, 27] float32
   ├── Output: [batch, 3] float32 (class probabilities)
   └── Using: onnxmltools + lightgbm

3. Export Scaler Parameters
   ├── min_values: [130] float32
   ├── max_values: [130] float32
   └── Format: JSON for MQL5 parsing

4. Export Feature Config
   ├── feature_names: ["body", "body_abs", ...]
   ├── feature_order: [0, 1, 2, ...]
   └── indicator_params: {atr_period: 14, ...}

═══════════════════════════════════════════════════════════════════
```

### 3.3 MQL5 File Structure

```
MQL5/
├── Experts/
│   └── XAUUSD_NeuralBot/
│       ├── XAUUSD_NeuralBot.mq5       # Main EA file
│       └── README.md
│
├── Include/
│   └── NeuralBot/
│       ├── FeatureCalculator.mqh     # Technical indicators
│       ├── MultiTimeframe.mqh        # MTF data handling
│       ├── ONNXInference.mqh         # ONNX model wrapper
│       ├── RiskManager.mqh           # Position sizing
│       ├── TradeManager.mqh          # Order execution
│       └── Config.mqh                # Configuration
│
├── Files/
│   └── NeuralBot/
│       ├── transformer.onnx          # Transformer model
│       ├── lightgbm.onnx             # LightGBM model
│       ├── scaler_params.json        # Scaling parameters
│       ├── feature_config.json       # Feature configuration
│       └── model_config.json         # Model hyperparameters
│
└── Scripts/
    └── NeuralBot/
        ├── TestFeatures.mq5          # Validate features match Python
        ├── TestONNX.mq5              # Test ONNX inference
        └── ValidateParity.mq5        # Full parity check
```

### 3.4 EA Input Parameters

```cpp
// Model Settings
input string   ModelPath        = "NeuralBot/lightgbm.onnx";
input string   TransformerPath  = "NeuralBot/transformer.onnx";
input double   SignalThreshold  = 0.6;      // Minimum confidence

// Risk Management
input double   RiskPercent      = 1.0;      // Risk per trade (%)
input double   MaxDrawdown      = 20.0;     // Max DD before stop (%)
input int      MaxOpenTrades    = 3;        // Maximum concurrent trades
input double   MaxLotSize       = 1.0;      // Maximum lot size

// Trade Settings
input int      TakeProfit       = 800;      // TP in points (80 pips)
input int      StopLoss         = 400;      // SL in points (40 pips)
input bool     UseTrailingStop  = true;     // Enable trailing stop
input int      TrailingStart    = 300;      // Trailing activation (30 pips)
input int      TrailingStep     = 100;      // Trailing step (10 pips)

// Session Filter
input bool     TradeAsian       = true;     // Trade Asian session
input bool     TradeLondon      = true;     // Trade London session  
input bool     TradeNewYork     = true;     // Trade New York session
input bool     TradeNewsEvents  = false;    // Trade during news
```

### 3.5 Feature Parity Validation

```
Step 1: Export Python features to CSV
        python export_features_for_validation.py

Step 2: Calculate same features in MQL5
        Run TestFeatures.mq5 script

Step 3: Compare feature values
        ├── Acceptable difference: < 0.001 (0.1%)
        ├── Check all 130 features
        └── Test across 1000+ bars

Step 4: Compare model predictions
        ├── Python prediction vs MQL5 prediction
        ├── Must match exactly (same ONNX model)
        └── Test edge cases

Step 5: Sign-off
        ├── All features within tolerance ✓
        ├── All predictions match ✓
        └── Ready for paper trading ✓
```

### 3.6 Phase 3 Deliverables

```
mt5_expert_advisor/
├── MQL5/
│   ├── Experts/XAUUSD_NeuralBot/
│   ├── Include/NeuralBot/
│   ├── Files/NeuralBot/
│   └── Scripts/NeuralBot/
│
├── python_export/
│   ├── export_transformer_onnx.py
│   ├── export_lightgbm_onnx.py
│   ├── export_scaler_params.py
│   └── validate_parity.py
│
└── docs/
    ├── INSTALLATION.md
    ├── CONFIGURATION.md
    ├── TROUBLESHOOTING.md
    └── PARITY_REPORT.md
```

### 3.7 Phase 3 Success Criteria

- [ ] Both ONNX models load successfully in MT5
- [ ] Feature parity < 0.1% difference from Python
- [ ] Prediction parity: 100% match with Python
- [ ] EA compiles without errors
- [ ] EA runs on demo without crashes for 24h
- [ ] Trades execute correctly (entry/exit/SL/TP)

---

## 📊 PHASE 4: PAPER TRADING

### Goal
Validate in real market conditions (demo account)

### 4.1 Paper Trading Protocol

```
Duration: 2 weeks minimum (cover different market conditions)

Week 1: Conservative Settings
├── Risk: 0.5% per trade
├── Max trades: 2 concurrent
└── Log everything

Week 2: Normal Settings
├── Risk: 1.0% per trade
├── Max trades: 3 concurrent
└── Compare to backtest expectations

Daily Monitoring:
├── Trade count vs expected
├── Win rate vs backtest
├── Average P&L per trade
├── Max drawdown
├── Feature values (sanity check)
└── Any errors or warnings
```

### 4.2 Red Flags (Stop & Investigate)

| Issue | Threshold | Action |
|-------|-----------|--------|
| Low win rate | < 40% over 50+ trades | Review model |
| High drawdown | > 15% | Reduce risk |
| Session losses | Consistent losses in specific session | Disable session |
| Feature errors | Any calculation errors | Fix immediately |
| Execution issues | Slippage > 5 pips average | Review broker |

### 4.3 Phase 4 Deliverables

```
paper_trading/
├── daily_logs/
│   ├── day_01_log.csv
│   ├── day_02_log.csv
│   └── ...
│
├── analysis/
│   ├── performance_vs_backtest.py
│   ├── trade_analysis.py
│   └── issue_tracker.md
│
└── reports/
    ├── week_1_report.pdf
    ├── week_2_report.pdf
    └── go_live_recommendation.pdf
```

### 4.4 Phase 4 Success Criteria

- [ ] 100+ trades executed
- [ ] Win rate within 10% of backtest
- [ ] No critical bugs or crashes
- [ ] Drawdown < 15%
- [ ] Positive P&L (even if small)
- [ ] All sessions covered (Asian/London/NY)
- [ ] No execution issues

---

## 🚀 PHASE 5: GO-LIVE

### Goal
Production deployment with monitoring

### 5.1 Launch Protocol

```
Pre-Launch:
☐ All Phase 1-4 criteria met
☐ VPS/Server setup (low latency)
☐ Broker account funded
☐ Risk parameters finalized
☐ Emergency stop procedures documented
☐ Monitoring alerts configured

Launch (Week 1):
├── 25% of intended capital
├── 0.5% risk per trade
└── Daily manual review

Scale Up (Week 2-4):
├── 50% capital if Week 1 positive
├── 0.75% risk per trade
└── Twice daily review

Full Operation (Month 2+):
├── 100% capital if profitable
├── 1.0% risk per trade
└── Weekly review
```

### 5.2 Monitoring & Alerts

```
Real-Time Monitoring:
├── Current P&L (today/week/month)
├── Open positions
├── Current drawdown
└── Last signal & prediction

Alerts (Telegram/Email):
├── Trade opened/closed
├── Daily P&L summary
├── Drawdown > 10% warning
├── Drawdown > 15% critical
├── No trades in 24h (check if running)
└── Error/exception occurred

Weekly Report:
├── Performance vs backtest
├── Trade breakdown by class
├── Regime performance
└── Model drift detection
```

### 5.3 Ongoing Maintenance

```
Monthly:
├── Compare live performance to backtest
├── Check for model drift (prediction distribution)
└── Review feature importance changes

Quarterly:
├── Retrain on latest data
├── Walk-forward validation update
├── Review and adjust risk parameters
└── Update stress test with new events

Triggers for Immediate Review:
├── 3 consecutive losing weeks
├── Drawdown > 20%
├── Win rate drops > 15% from backtest
└── Major market structure change
```

---

## 📅 TIMELINE & MILESTONES

```
═══════════════════════════════════════════════════════════════════
                        PROJECT TIMELINE
═══════════════════════════════════════════════════════════════════

Week 1: PHASE 1 - Model Optimization
├── Day 1-2: Class balancing implementation
├── Day 3-4: Feature engineering v2
├── Day 5-6: Ensemble training
└── Day 7: Threshold optimization & validation

Week 2: PHASE 2 - Backtesting Suite
├── Day 1-2: Walk-forward optimization
├── Day 3: Monte Carlo simulation
├── Day 4: Stress testing
├── Day 5: Regime analysis
├── Day 6: Reality gap testing
└── Day 7: Comprehensive report generation

Week 3: PHASE 3 - MT5 Integration  
├── Day 1-2: ONNX export pipeline
├── Day 3-4: MQL5 EA development
├── Day 5: Feature parity validation
├── Day 6: Integration testing
└── Day 7: Bug fixes & optimization

Week 4-5: PHASE 4 - Paper Trading
├── Week 4: Conservative settings
└── Week 5: Normal settings

Week 6+: PHASE 5 - Go-Live
├── Week 6: 25% capital deployment
├── Week 7-8: Scale up to 50%
└── Month 2+: Full operation

═══════════════════════════════════════════════════════════════════
```

---

## 📁 FINAL PROJECT STRUCTURE

```
xaubot/
├── README.md
├── XAUBOT_DEVELOPMENT_PLAN.md      # This file
├── requirements.txt
├── config/
│   ├── model_config.yaml
│   ├── backtest_config.yaml
│   └── trading_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── python_training/
│   ├── features/
│   │   ├── build_features_v2.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train_transformer.py
│   │   ├── train_lightgbm_balanced.py
│   │   ├── train_xgboost.py
│   │   ├── train_ensemble.py
│   │   └── optimize_thresholds.py
│   │
│   ├── backtesting/
│   │   ├── base_backtest.py
│   │   ├── walk_forward.py
│   │   ├── monte_carlo.py
│   │   ├── stress_test.py
│   │   ├── regime_analysis.py
│   │   ├── reality_gap.py
│   │   └── comprehensive_report.py
│   │
│   ├── export/
│   │   ├── export_transformer_onnx.py
│   │   ├── export_lightgbm_onnx.py
│   │   └── validate_parity.py
│   │
│   └── models/  (saved models)
│       ├── transformer.pth
│       ├── transformer.onnx
│       ├── lightgbm_balanced.txt
│       ├── lightgbm.onnx
│       ├── ensemble_meta.pkl
│       └── scaler.pkl
│
├── mt5_expert_advisor/
│   ├── MQL5/
│   │   ├── Experts/
│   │   ├── Include/
│   │   ├── Files/
│   │   └── Scripts/
│   └── docs/
│
├── monitoring/
│   ├── dashboard.py
│   ├── alerts.py
│   └── reports/
│
└── docs/
    ├── MODEL_DOCUMENTATION.md
    ├── BACKTEST_REPORT.md
    ├── MT5_INTEGRATION.md
    ├── DEPLOYMENT_GUIDE.md
    └── MAINTENANCE_GUIDE.md
```

---

## 🔗 REFERENCE LINKS

- **MT5 ONNX Documentation**: https://www.metatrader5.com/en/metaeditor/help/machine_learning
- **LightGBM ONNX**: https://onnx.ai/sklearn-onnx/
- **PyTorch ONNX Export**: https://pytorch.org/docs/stable/onnx.html

---

## ✅ NEXT ACTION

**Start Phase 1: Model Optimization**

```bash
cd /workspace/xaubot
python python_training/train_lightgbm_balanced.py
```

---

*Document Version: 1.0*  
*Last Updated: December 2024*
