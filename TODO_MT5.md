# TODO: MT5 Expert Advisor Development

## Session Goal: Build and Test MT5 EA with ONNX Model

### Prerequisites (Before Starting):
- [ ] Install MT5 (if not already installed)
- [ ] Download ONNX Runtime for MT5: https://github.com/microsoft/onnxruntime
- [ ] Review MQL5 documentation: https://www.mql5.com/en/docs

---

## Phase 6: MT5 EA Development

### Part 1: Environment Setup (15 min)
- [ ] Install MT5 ONNX library
- [ ] Create new EA project in MetaEditor
- [ ] Copy ONNX model to MT5 `Files/models/` folder
- [ ] Verify model loads in MT5

### Part 2: Feature Calculation in MQL5 (60 min)
- [ ] Implement 21 technical indicators (ATR, RSI, EMA, etc.)
- [ ] Implement 10 market structure features (FVG, OB, swings)
- [ ] Implement 8 orderflow features (CVD approximation)
- [ ] Implement 6 time features (session position)
- [ ] Implement 8 volatility features
- [ ] Implement 6 price action features
- [ ] Implement SMC quality score
- [ ] Test: Compare MQL5 features vs Python features

### Part 3: EA Core Logic (45 min)
- [ ] Create input parameters (confidence threshold, risk %, etc.)
- [ ] Implement ONNX model inference
- [ ] Add confidence filtering (>= 55%)
- [ ] Implement signal generation (LONG/SHORT/HOLD)
- [ ] Add trade execution logic

### Part 4: Risk Management (30 min)
- [ ] Implement 5% equity risk per trade
- [ ] Add ATR-based stop loss
- [ ] Add trailing stop (activate at +10%)
- [ ] Add max drawdown kill switch (25%)
- [ ] Add max trades per day limit (5)

### Part 5: Validation Mode (30 min)
- [ ] Add dual-mode: Validation vs ML
- [ ] Implement manual EMA/RSI confirmation (validation mode)
- [ ] Add trade logging
- [ ] Add equity tracking

### Part 6: Testing (45 min)
- [ ] Compile EA (fix any errors)
- [ ] Load on XAUUSD M1 chart
- [ ] Run MT5 Strategy Tester (2023-2025)
- [ ] Compare results: MT5 vs Python backtest
- [ ] Validate win rate ~66%
- [ ] Check profit factor ~1.96

---

## Expected Outputs:
1. `XAUUSD_AI_Bot.mq5` - Main EA file
2. `FeatureEngine.mqh` - Feature calculation library
3. `RiskManager.mqh` - Risk management module
4. Backtest report (HTML) from MT5
5. Trade log (CSV) for analysis

---

## Success Criteria:
- ✅ EA compiles without errors
- ✅ Model loads and predicts correctly
- ✅ Features match Python calculations (±2%)
- ✅ MT5 backtest shows ~60-70% win rate
- ✅ Profit factor >= 1.5
- ✅ Max drawdown < 25%

---

## Estimated Time: 3-4 hours
## Recommended: Split into 2 sessions (Setup + Features, then Logic + Testing)
