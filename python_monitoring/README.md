# MT5 Python Monitoring System

Real-time monitoring and analysis of XAUUSD Neural Bot performance using the MetaTrader5 Python package.

## Features

### 1. Live Performance Monitor (`monitor_live_performance.py`)
- **Real-time accuracy tracking** - Calculate model accuracy by comparing predictions with actual outcomes
- **Position monitoring** - Track open trades, P/L, and risk metrics
- **Account metrics** - Monitor balance, equity, margin level
- **Model drift detection** - Alert when accuracy drops below acceptable thresholds
- **Automated logging** - Save metrics history to JSON for analysis

### 2. Streamlit Dashboard (`dashboard.py`)
- **Interactive visualizations** - Real-time charts of prediction distributions, confidence levels
- **Accuracy trends** - Track model performance over time
- **Recent predictions** - Table view of latest signals
- **Auto-refresh** - Configurable refresh intervals (10-300 seconds)

## Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify MT5 Connection:**
   Ensure MetaTrader 5 terminal is running on your system.

## Usage

### Live Monitoring (Command Line)

**Basic usage (connect to current MT5 terminal):**
```bash
python monitor_live_performance.py
```

**With specific account:**
```bash
python monitor_live_performance.py \
    --account 12345678 \
    --password "YourPassword" \
    --server "YourBroker-Demo"
```

**Custom check interval:**
```bash
python monitor_live_performance.py --interval 30  # Check every 30 seconds
```

**Run for specific duration:**
```bash
python monitor_live_performance.py --duration 24  # Run for 24 hours
```

**Example Output:**
```
✅ Connected to MT5 (using current terminal login)
   Terminal: MetaTrader 5
   Build: 3770
   Connected: True

🔍 LIVE MONITORING STARTED
======================================================================
Start time: 2025-12-22 10:30:00
Check interval: 60s
Duration: Indefinite
======================================================================

[2025-12-22 10:31:00] Checking...
✅ Loaded 543 predictions from MQL5/Files/prediction_log.csv

📊 MODEL PERFORMANCE (last 528 predictions):
   Overall Accuracy: 72.3%
   LONG signals: 83.5%
   HOLD signals: 68.2%
   SHORT signals: 64.7%
   High confidence (≥65%): 85.1%

💼 OPEN POSITIONS (1):
   #12345678 LONG 0.1 lots @ 2045.32 | P/L: +$8.50 | M1 LONG 0.72

💰 ACCOUNT:
   Balance: $10000.00
   Equity: $10008.50
   Profit: $8.50
   Margin Level: 8542.3%
```

### Dashboard (Web Interface)

1. **Start Dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

2. **Access in Browser:**
   Open http://localhost:8501

3. **Features:**
   - **Metrics Overview**: Total predictions, average confidence, signal distribution
   - **Prediction Trends**: Hourly probability charts for LONG/HOLD/SHORT
   - **Class Distribution**: Bar chart of signal types
   - **Confidence Histogram**: Distribution of model confidence levels
   - **Recent Predictions**: Last 20 predictions with full details
   - **Accuracy Tracking**: Model performance over time (when monitoring is running)

4. **Sidebar Controls:**
   - Adjust refresh rate (10-300 seconds)
   - Set lookback period (1-168 hours)
   - Manual refresh button

## How It Works

### Accuracy Calculation

The monitor calculates actual outcomes by:

1. **Loading EA predictions** from `MQL5/Files/prediction_log.csv`
2. **Fetching historical prices** from MT5 using `copy_rates_range()`
3. **Calculating future returns** (default: 15 minutes ahead)
4. **Classifying outcomes**:
   - `LONG (2)` if price moved up >$0.50
   - `SHORT (0)` if price moved down <-$0.50
   - `HOLD (1)` otherwise
5. **Comparing predictions vs actuals** to compute accuracy

### Model Drift Detection

The system alerts when:
- Overall accuracy < 55% (below acceptable threshold)
- Significant drop in high-confidence predictions
- Unusual prediction distributions (e.g., 90%+ HOLD signals)

When alerts occur, consider:
1. Checking market conditions (high volatility, news events)
2. Reviewing filter parameters
3. Retraining model with recent data

## File Locations

The monitor searches for logs in these locations:
1. `MQL5/Files/prediction_log.csv` (relative to script)
2. `<MT5_DATA_PATH>/MQL5/Files/prediction_log.csv`
3. `<MT5_COMMON_DATA_PATH>/Files/prediction_log.csv`

**To find your MT5 data path:**
- Open MT5 Terminal
- File → Open Data Folder
- Look for `MQL5/Files/` subdirectory

## Monitoring Best Practices

1. **Run monitoring continuously** in the background during live trading
2. **Check dashboard daily** to review performance trends
3. **Set alerts** for accuracy drops below 55%
4. **Review rejected signals** in EA logs to tune hybrid filters
5. **Compare filtered vs unfiltered** performance using EnableHybridValidation flag

## Troubleshooting

### "MT5 initialization failed"
- Ensure MT5 terminal is running
- Check if Python can find MT5 installation
- Try reinstalling MetaTrader5 package: `pip install --upgrade MetaTrader5`

### "Prediction log not found"
- Enable `EnablePredictionLog = true` in EA settings
- Check EA is actually running and making predictions
- Verify log file exists in MT5 Files folder

### "Failed to fetch price data"
- Check MT5 terminal is connected to broker
- Ensure XAUUSD symbol is in Market Watch
- Verify historical data is available (Tools → History Center)

### Dashboard not updating
- Check refresh rate setting in sidebar
- Manually click "Refresh Now" button
- Restart Streamlit if needed: Ctrl+C, then `streamlit run dashboard.py`

## Integration with EA

The EA automatically logs predictions when:
```mql5
input bool EnablePredictionLog = true;  // Enable in EA settings
```

Log format:
```csv
time,p_short,p_hold,p_long,best_class,best_prob
2025-12-22 10:30:00,0.125,0.243,0.632,2,0.632
```

## Advanced Usage

### Automated Retraining

Create a cron job (Linux/Mac) or Task Scheduler task (Windows) to:
1. Run monitoring for 24 hours
2. Check if accuracy < 55%
3. Trigger retraining pipeline if needed

Example:
```bash
# monitor_and_retrain.sh
python monitor_live_performance.py --duration 24
python check_performance_and_retrain.py  # Custom script
```

### Custom Metrics

Extend `MT5PerformanceMonitor` class to add:
- Win rate calculations
- Profit factor tracking
- Sharpe ratio computation
- Custom alert conditions

## Related Files

- **EA:** `mt5_expert_advisor/XAUUSD_NeuralBot_M1.mq5`
- **Research Report:** `docs/MT5_IMPLEMENTATION_RESEARCH_REPORT.md`
- **Training Scripts:** `src/train_lightgbm.py`

## Support

For issues or questions:
1. Check logs in `monitoring_results.json`
2. Review EA logs in MT5 Experts tab
3. Consult research report for implementation details
