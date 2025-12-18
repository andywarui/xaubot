# Live Trading Configuration

## Conservative Settings for Real Deployment

Based on stress testing and sensitivity analysis, use these parameters for live trading:

### Risk Parameters
- **Risk per Trade**: 0.5% (max 1.0%)
- **Initial Capital**: $500-1000 minimum
- **Daily Loss Limit**: 3% of starting equity
- **Weekly Loss Limit**: 5% of starting equity
- **Max Trades per Day**: 3-5
- **Max Open Positions**: 1

### Entry Rules
- **Confidence Threshold**: 0.70 (higher = fewer but better trades)
- **Session Filter**: Only 12:00-17:00 UTC (London-NY overlap)
- **Spread Filter**: Skip if spread > 4 pips
- **Volatility Filter**: Skip if ATR > 3× normal (avoid news spikes)

### Exit Rules
- **Take Profit**: 80 pips
- **Stop Loss**: 40 pips
- **Trailing Stop**: Activate at +10% (move SL to breakeven at +40 pips)
- **Time Stop**: Close position after 12 hours if not hit TP/SL

### Circuit Breakers
1. **Daily Kill Switch**: Stop trading if down 3% in a day
2. **Losing Streak**: Halt after 3 consecutive losses, resume next day
3. **Max Drawdown**: Stop all trading if account drops 15% from peak
4. **Weekend Closure**: Close all positions 1 hour before Friday market close

### Monitoring Checklist

**Daily:**
- [ ] Check EA is running (green indicator)
- [ ] Verify trades within session hours
- [ ] Review P&L vs limits
- [ ] Check for ONNX errors in logs

**Weekly:**
- [ ] Calculate actual win rate (compare to backtest 76%)
- [ ] Verify profit factor >2.0
- [ ] Check drawdown <10%
- [ ] Review trade distribution (LONG:SHORT ratio)

**Monthly:**
- [ ] Full performance review
- [ ] Compare to backtest metrics
- [ ] Decide on risk adjustment
- [ ] Consider model retraining if drift detected

### Gradual Scale-Up Plan

**Phase 1 (Months 1-2): Demo Account**
- Risk: 0.5%
- Capital: $500 demo
- Goal: Validate EA behavior matches backtest

**Phase 2 (Month 3): Micro Live**
- Risk: 0.5%
- Capital: $500-1000 real
- Goal: Verify real execution (slippage, spreads, latency)

**Phase 3 (Months 4-6): Small Live**
- Risk: 0.75%
- Capital: $2000-5000
- Goal: Prove consistency over 3+ months

**Phase 4 (Month 7+): Full Live**
- Risk: 1.0% (max)
- Capital: $10,000+
- Goal: Scale if metrics hold

### Red Flags to Watch
- Win rate drops below 65% for 2+ weeks
- Profit factor drops below 1.5
- Drawdown exceeds 15%
- Trade frequency spikes >10 trades/day
- Unusual LONG:SHORT ratio (should be ~1:5-6)

### Emergency Actions
If red flags appear:
1. Reduce risk to 0.25%
2. Increase confidence threshold to 0.75
3. Reduce max trades to 2/day
4. Collect 2 weeks of logs for analysis
5. Consider model retraining

---

**Last Updated**: December 12, 2025
**Backtest Period**: 2022-2025 (3+ years)
**OOS Test**: Last 12 months
**Expected Metrics**: 76% WR, 5.0+ PF, <10% DD
