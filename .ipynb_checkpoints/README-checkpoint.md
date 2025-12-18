# 🤖 XAUUSD AI Trading Bot

> Industrial-grade MetaTrader 5 Expert Advisor powered by LightGBM and Smart Money Concepts (SMC)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](https://github.com/andywarui/xauusd-ai-trading-bot)

**Validated Performance**: 66.2% win rate | 1.96 profit factor | 3,780% return (7-month backtest)

---

## 📊 Performance Summary

| Metric | Result | Status |
|--------|--------|--------|
| **Win Rate** | 66.2% | ✅ **Excellent** |
| **Profit Factor** | 1.96 | ✅ **Profitable** |
| **Max Drawdown** | 19.5% | ✅ **Under Control** |
| **LONG Accuracy** | 64.1% | ⭐ **Strong** |
| **SHORT Accuracy** | 73.0% | ⭐⭐ **Very Strong** |
| **Trades/Day** | 15.7 | ⚠️ *Can optimize to 4-5* |
| **Test Period** | 7 months | Apr-Nov 2025 |

---

## 🎯 Features

### AI-Powered Predictions
- LightGBM classifier with 68 features
- 55% confidence threshold for trade filtering
- ONNX format for MT5 integration
- Smart Money Concepts (SMC) quality scoring

### Advanced Feature Engineering
- **21** Technical indicators (ATR, RSI, MACD, Bollinger Bands, etc.)
- **10** Market structure features (FVG, Order Blocks, Liquidity Sweeps)
- **8** Orderflow metrics (CVD approximation, delta divergence)
- **8** Volatility context features
- **6** Time-based features (session positioning)
- **1** SMC quality score (4-step validation)

### Risk Management
- 5% equity risk per trade
- ATR-based dynamic stop-loss
- 25% max drawdown kill switch
- Trailing stop activation at +10%
- London-NY overlap trading only (13:00-16:59 UTC)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- MetaTrader 5
- 16GB RAM

### Installation

Clone and setup:
git clone https://github.com/andywarui/xauusd-ai-trading-bot.git
cd xauusd-ai-trading-bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

text

### Data Pipeline
python src/merge_yearly_data.py
python src/validate_merged_data.py
python src/filter_overlap.py
python src/feature_engineering.py
python src/create_labels.py

text

### Model Training
python src/train_lightgbm.py
python src/analyze_confidence.py
python src/backtest_simple.py

text

### ONNX Export (Python 3.11)
py -3.11 -m venv .venv_onnx
.venv_onnx\Scripts\activate
pip install lightgbm onnx onnxmltools
python src/export_to_onnx_simple.py

text

---

## 📁 Project Structure

xauusd-ai-trading-bot/
├── data/
│ ├── raw/ # Raw XAUUSD M1 data
│ └── processed/ # Filtered and labeled
├── src/
│ ├── merge_yearly_data.py
│ ├── filter_overlap.py
│ ├── feature_engineering.py
│ ├── train_lightgbm.py
│ └── backtest_simple.py
├── python_training/models/ # Trained models
├── mt5_expert_advisor/Files/ # ONNX model (222 KB)
└── docs/ # Documentation

text

---

## 🧪 Backtest Results

**Configuration**: $50 capital | 5% risk | 55% confidence | 149 days

Total Trades: 2,332
Win Rate: 66.2%
Profit Factor: 1.96
Net Profit: $1,890 (3,780%)
Max Drawdown: 19.5%

LONG: 1,787 trades | 64.1% WR | $1,262 profit
SHORT: 545 trades | 73.0% WR | $627 profit

text

---

## 📈 Model Details

**LightGBM Classifier**
- Trees: 102
- Features: 68
- Training: 142,511 samples
- Testing: 35,628 samples
- Classes: SHORT | HOLD | LONG

**Feature Categories**: Technical (21) | Market Structure (10) | Orderflow (8) | Time (6) | Volatility (8) | Price Action (6) | SMC Score (1)

---

## 🎯 Roadmap

### ✅ Completed (62.5%)
- [x] Data acquisition (178k bars)
- [x] Feature engineering (61 features)
- [x] Model training (LightGBM)
- [x] ONNX export
- [x] Backtest validation

### ⏳ In Progress
- [ ] MT5 EA development
- [ ] Risk management in MQL5
- [ ] Strategy Tester validation

### 🔮 Future
- [ ] Shadow testing (30 days)
- [ ] Live deployment
- [ ] Auto-retraining pipeline

---

## 🔧 Planned v2 Enhancements

- News filter (ForexFactory API)
- Time filters (avoid session opens/closes)
- Sentiment features (COT, DXY)
- Multi-timeframe confirmation
- Ensemble models

---

## ⚠️ Disclaimer

**Educational purposes only**. Trading carries substantial risk. Past performance ≠ future results. Test on demo accounts first.

---

## 📧 Contact

**Author**: Andy Warui  
**Repository**: https://github.com/andywarui/xauusd-ai-trading-bot

---

## 📜 License

MIT License

---

**⭐ Star this repo if you find it useful!**
