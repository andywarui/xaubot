# XAUUSD Neural Bot - Parameter Optimization Guide

This guide provides recommended parameter ranges and optimization strategies based on research findings and best practices.

## Table of Contents

1. [Parameter Reference](#parameter-reference)
2. [Optimization Strategies](#optimization-strategies)
3. [Session-Specific Configurations](#session-specific-configurations)
4. [Risk Profiles](#risk-profiles)
5. [A/B Testing Framework](#ab-testing-framework)

---

## Parameter Reference

### Core Trading Parameters

| Parameter | Default | Safe Range | Aggressive Range | Description | Impact |
|-----------|---------|------------|------------------|-------------|--------|
| **RiskPercent** | 0.5 | 0.25-0.75 | 0.75-2.0 | Risk per trade (% of equity) | Higher = larger positions, more risk |
| **ConfidenceThreshold** | 0.60 | 0.65-0.75 | 0.50-0.65 | Min ML confidence to trade | Higher = fewer trades, higher quality |
| **MaxTradesPerDay** | 5 | 2-5 | 5-10 | Daily trade limit | Limits overtrading |
| **MaxDailyLoss** | 4.0 | 3.0-5.0 | 5.0-8.0 | Daily loss limit (% of equity) | Protection against bad days |
| **StopLossUSD** | 4.0 | 3.0-6.0 | 2.0-3.0 | Stop loss in USD | Smaller = tighter risk |
| **TakeProfitUSD** | 8.0 | 6.0-12.0 | 4.0-6.0 | Take profit in USD | Affects win rate vs profit size |
| **MaxMarginPercent** | 50.0 | 30.0-50.0 | 50.0-80.0 | Max margin usage per trade | Safety buffer |

### Hybrid Validation Filters (Research-Based)

| Parameter | Default | Conservative | Aggressive | Description | Effect When Stricter |
|-----------|---------|--------------|------------|-------------|---------------------|
| **EnableHybridValidation** | true | true | false/true | Enable ML+Technical filters | Fewer but higher quality trades |
| **MaxSpreadUSD** | 2.0 | 1.5-2.0 | 2.5-3.0 | Max acceptable spread | Fewer trades during wide spreads |
| **RSI_OverboughtLevel** | 70.0 | 65.0-70.0 | 70.0-75.0 | RSI overbought threshold | Fewer LONG trades near peaks |
| **RSI_OversoldLevel** | 30.0 | 30.0-35.0 | 25.0-30.0 | RSI oversold threshold | Fewer SHORT trades near bottoms |
| **ATR_MinLevel** | 1.5 | 2.0-3.0 | 1.0-1.5 | Min volatility filter | Avoids choppy markets |
| **ATR_MaxLevel** | 8.0 | 6.0-8.0 | 8.0-12.0 | Max volatility filter | Avoids extreme volatility |
| **ADX_MinStrength** | 20.0 | 22.0-25.0 | 15.0-20.0 | Min trend strength | Requires stronger trends |
| **RequireMTFAlignment** | true | true | false | Multi-timeframe EMA check | Requires higher TF confirmation |

---

## Optimization Strategies

### Strategy 1: Win Rate Optimization (Recommended Start)

**Goal:** Maximize accuracy while maintaining reasonable trade frequency

**Parameters to Optimize:**
```
ConfidenceThreshold: 0.55 to 0.75, step 0.05
RSI_OverboughtLevel: 65 to 75, step 5
RSI_OversoldLevel: 25 to 35, step 5
ADX_MinStrength: 15 to 25, step 5

Fixed:
RiskPercent = 0.5
EnableHybridValidation = true
RequireMTFAlignment = true
```

**Expected Outcomes:**
- Higher confidence → Fewer trades, higher win rate
- Stricter RSI → Better entry timing
- Higher ADX → Only trending markets

**Optimization Target:** Maximize **Profit Factor** or **Sharpe Ratio**

**Validation:**
- Forward test period: 25%
- Min acceptable win rate: 65%
- Min acceptable trades: 30 over test period

### Strategy 2: Trade Frequency Optimization

**Goal:** Find balance between signal frequency and quality

**Parameters to Optimize:**
```
ConfidenceThreshold: 0.50 to 0.70, step 0.05
ATR_MinLevel: 1.0 to 3.0, step 0.5
ATR_MaxLevel: 6.0 to 12.0, step 2.0
MaxTradesPerDay: 3 to 10, step 1

Fixed:
EnableHybridValidation = true
RequireMTFAlignment = true
```

**Expected Outcomes:**
- Lower confidence → More trades
- Wider ATR range → More volatility regimes accepted
- Higher daily limit → More opportunities

**Optimization Target:** Maximize **Total Net Profit** with constraint **PF > 1.5**

### Strategy 3: Risk-Reward Optimization

**Goal:** Optimize position sizing and trade management

**Parameters to Optimize:**
```
RiskPercent: 0.25 to 1.0, step 0.25
StopLossUSD: 2.0 to 6.0, step 1.0
TakeProfitUSD: 4.0 to 12.0, step 2.0
MaxMarginPercent: 30.0 to 70.0, step 10.0

Fixed:
ConfidenceThreshold = 0.65
EnableHybridValidation = true
```

**Expected Outcomes:**
- Smaller SL → Tighter risk but more stops hit
- Larger TP → Bigger wins but lower hit rate
- Higher risk % → Faster growth but larger drawdowns

**Optimization Target:** Maximize **Sharpe Ratio** (risk-adjusted returns)

**Important Constraints:**
- Max Drawdown < 15%
- Profit Factor > 1.3

### Strategy 4: Filter Threshold Optimization

**Goal:** Fine-tune hybrid validation filters

**Parameters to Optimize:**
```
MaxSpreadUSD: 1.5 to 3.0, step 0.5
RSI_OverboughtLevel: 65 to 75, step 2
RSI_OversoldLevel: 25 to 35, step 2
ATR_MinLevel: 1.0 to 2.5, step 0.5
ATR_MaxLevel: 6.0 to 10.0, step 1.0
ADX_MinStrength: 15 to 25, step 2

Fixed:
EnableHybridValidation = true
ConfidenceThreshold = 0.65
RequireMTFAlignment = true
```

**Expected Outcomes:**
- Find optimal filter balance
- Reject bad signals while keeping good ones
- Maximize filter effectiveness

**Optimization Target:** Maximize **Win Rate** while maintaining **Min Trades > 50**

---

## Session-Specific Configurations

### London Session (08:00-17:00 UTC)

**Characteristics:**
- High liquidity
- Moderate volatility
- Tight spreads
- Best for XAUUSD trading

**Recommended Settings:**
```mql5
// Core
ConfidenceThreshold = 0.60        // Standard threshold
MaxTradesPerDay = 5               // Good opportunities
RiskPercent = 0.5                 // Standard risk

// Filters
MaxSpreadUSD = 1.5                // Tight spreads expected
ATR_MinLevel = 2.0                // Higher volatility
ATR_MaxLevel = 8.0                // Normal max
ADX_MinStrength = 20.0            // Standard trend strength
RSI_OverboughtLevel = 70.0        // Standard
RSI_OversoldLevel = 30.0          // Standard
RequireMTFAlignment = true        // Keep confirmation

// Risk Management
StopLossUSD = 4.0                 // Standard stops
TakeProfitUSD = 8.0               // 2:1 R:R
```

### NY Session (13:00-22:00 UTC)

**Characteristics:**
- Highest volatility (news releases)
- Wide swings
- More false signals
- Overlap with London (13:00-17:00)

**Recommended Settings:**
```mql5
// Core
ConfidenceThreshold = 0.65        // Higher threshold for volatility
MaxTradesPerDay = 5               // Many opportunities but selective
RiskPercent = 0.5                 // Keep standard (volatility compensates)

// Filters
MaxSpreadUSD = 2.0                // Allow slightly wider spreads
ATR_MinLevel = 2.5                // Higher volatility expected
ATR_MaxLevel = 10.0               // Allow extreme moves
ADX_MinStrength = 22.0            // Require stronger trends
RSI_OverboughtLevel = 68.0        // Slightly tighter (avoid blowoffs)
RSI_OversoldLevel = 32.0          // Slightly tighter
RequireMTFAlignment = true        // CRITICAL for volatility

// Risk Management
StopLossUSD = 5.0                 // Wider stops for volatility
TakeProfitUSD = 10.0              // Wider targets
```

### Asian Session (00:00-09:00 UTC)

**Characteristics:**
- Lower liquidity
- Wider spreads
- Range-bound behavior
- Fewer opportunities

**Recommended Settings:**
```mql5
// Core
ConfidenceThreshold = 0.70        // Higher threshold (fewer but better)
MaxTradesPerDay = 2               // Limited opportunities
RiskPercent = 0.5                 // Standard risk

// Filters
MaxSpreadUSD = 2.5                // Wider spreads expected
ATR_MinLevel = 1.0                // Lower volatility
ATR_MaxLevel = 6.0                // Avoid rare spikes
ADX_MinStrength = 25.0            // Require strong trends (rare but best)
RSI_OverboughtLevel = 72.0        // Tighter (clearer signals)
RSI_OversoldLevel = 28.0          // Tighter
RequireMTFAlignment = true        // Essential for ranging markets

// Risk Management
StopLossUSD = 3.5                 // Tighter (smaller moves)
TakeProfitUSD = 7.0               // Tighter targets
```

### 24/7 Balanced (All Sessions)

**Characteristics:**
- Mix of all market conditions
- Adaptive to changing volatility
- Balanced approach

**Recommended Settings:**
```mql5
// Core
ConfidenceThreshold = 0.65        // Balanced threshold
MaxTradesPerDay = 5               // Reasonable across all sessions
RiskPercent = 0.5                 // Standard

// Filters
MaxSpreadUSD = 2.0                // Balanced
ATR_MinLevel = 1.5                // Accept moderate vol
ATR_MaxLevel = 8.0                // Reject extreme vol
ADX_MinStrength = 20.0            // Standard trend requirement
RSI_OverboughtLevel = 70.0        // Standard
RSI_OversoldLevel = 30.0          // Standard
RequireMTFAlignment = true        // Always use confirmation

// Risk Management
StopLossUSD = 4.0                 // Standard
TakeProfitUSD = 8.0               // 2:1 R:R
```

---

## Risk Profiles

### Ultra-Conservative (Capital Preservation)

**Goal:** Minimize drawdown, very high win rate, slow growth

```mql5
// Core
RiskPercent = 0.25                // Minimal risk
ConfidenceThreshold = 0.75        // Only highest confidence
MaxTradesPerDay = 2               // Very selective
MaxDailyLoss = 2.0                // Tight daily limit

// Filters (Strict)
EnableHybridValidation = true
MaxSpreadUSD = 1.5
RSI_OverboughtLevel = 65.0
RSI_OversoldLevel = 35.0
ATR_MinLevel = 2.0
ATR_MaxLevel = 6.0
ADX_MinStrength = 25.0
RequireMTFAlignment = true

// Risk Management
StopLossUSD = 3.0                 // Tight stops
TakeProfitUSD = 9.0               // 3:1 R:R
MaxMarginPercent = 30.0           // Conservative margin

// Expected Performance
// - Win rate: 80-85%
// - Profit factor: 2.5-3.0
// - Trades/month: 20-30
// - Max DD: 5-8%
// - Sharpe: 1.5-2.0
```

### Conservative (Standard - Recommended)

**Goal:** Good balance of safety and growth

```mql5
// Core
RiskPercent = 0.5                 // Standard risk
ConfidenceThreshold = 0.65        // High quality signals
MaxTradesPerDay = 3               // Moderate frequency
MaxDailyLoss = 4.0                // Reasonable limit

// Filters (Balanced)
EnableHybridValidation = true
MaxSpreadUSD = 2.0
RSI_OverboughtLevel = 70.0
RSI_OversoldLevel = 30.0
ATR_MinLevel = 1.5
ATR_MaxLevel = 8.0
ADX_MinStrength = 20.0
RequireMTFAlignment = true

// Risk Management
StopLossUSD = 4.0                 // Standard stops
TakeProfitUSD = 8.0               // 2:1 R:R
MaxMarginPercent = 50.0           // Balanced

// Expected Performance
// - Win rate: 70-75%
// - Profit factor: 1.8-2.2
// - Trades/month: 40-60
// - Max DD: 8-12%
// - Sharpe: 1.0-1.5
```

### Moderate (Growth-Oriented)

**Goal:** Higher returns with moderate risk

```mql5
// Core
RiskPercent = 0.75                // Increased risk
ConfidenceThreshold = 0.60        // More signals
MaxTradesPerDay = 5               // More opportunities
MaxDailyLoss = 5.0                // Higher tolerance

// Filters (Relaxed)
EnableHybridValidation = true
MaxSpreadUSD = 2.5
RSI_OverboughtLevel = 72.0
RSI_OversoldLevel = 28.0
ATR_MinLevel = 1.0
ATR_MaxLevel = 10.0
ADX_MinStrength = 18.0
RequireMTFAlignment = true        // Keep this protection

// Risk Management
StopLossUSD = 4.0                 // Standard stops
TakeProfitUSD = 8.0               // 2:1 R:R
MaxMarginPercent = 60.0           // More margin usage

// Expected Performance
// - Win rate: 65-70%
// - Profit factor: 1.5-1.8
// - Trades/month: 60-90
// - Max DD: 12-15%
// - Sharpe: 0.8-1.2
```

### Aggressive (High Risk/Reward)

**Goal:** Maximum returns, accept higher drawdowns

```mql5
// Core
RiskPercent = 1.0                 // High risk
ConfidenceThreshold = 0.55        // Many signals
MaxTradesPerDay = 8               // High frequency
MaxDailyLoss = 8.0                // High tolerance

// Filters (Minimal)
EnableHybridValidation = true     // Keep some protection
MaxSpreadUSD = 3.0
RSI_OverboughtLevel = 75.0
RSI_OversoldLevel = 25.0
ATR_MinLevel = 0.5
ATR_MaxLevel = 12.0
ADX_MinStrength = 15.0
RequireMTFAlignment = false       // Trade more aggressively

// Risk Management
StopLossUSD = 3.0                 // Tighter stops
TakeProfitUSD = 6.0               // 2:1 R:R
MaxMarginPercent = 70.0           // High margin usage

// Expected Performance
// - Win rate: 60-65%
// - Profit factor: 1.3-1.6
// - Trades/month: 100-150
// - Max DD: 15-25%
// - Sharpe: 0.5-0.9
```

**⚠️ WARNING: Aggressive profile not recommended without extensive testing!**

---

## A/B Testing Framework

### Test 1: Hybrid Validation Impact

**Question:** Does hybrid validation improve performance?

**Setup:**
```
Variant A (Control):
  EnableHybridValidation = false
  ConfidenceThreshold = 0.60
  All other settings = default

Variant B (Treatment):
  EnableHybridValidation = true
  ConfidenceThreshold = 0.60
  All hybrid filters = default

Duration: 2 weeks each (same market conditions)
```

**Metrics to Track:**
- Win rate (expect B higher by 10-15%)
- Trade frequency (expect B lower by 30-40%)
- Profit factor (expect B higher by 20-30%)
- Max drawdown (expect B lower by 20-30%)

**Expected Result:** B should significantly outperform A

### Test 2: Confidence Threshold Sweet Spot

**Question:** What's the optimal confidence threshold?

**Setup:**
```
Variant A: ConfidenceThreshold = 0.55
Variant B: ConfidenceThreshold = 0.60
Variant C: ConfidenceThreshold = 0.65
Variant D: ConfidenceThreshold = 0.70

All other settings = conservative defaults
Duration: 2 weeks each
```

**Metrics to Track:**
- Trades per week
- Win rate
- Profit factor
- Risk-adjusted returns (Sharpe)

**Analysis:**
- Plot win rate vs trade frequency
- Find inflection point
- Consider your preference: more trades or higher quality?

### Test 3: MTF Alignment Necessity

**Question:** Is multi-timeframe alignment worth the trade-offs?

**Setup:**
```
Variant A:
  RequireMTFAlignment = false
  EnableHybridValidation = true
  All other filters = default

Variant B:
  RequireMTFAlignment = true
  EnableHybridValidation = true
  All other filters = default

Duration: 2 weeks each
```

**Metrics to Track:**
- Trade count (expect A higher)
- Win rate (expect B higher)
- False signal rate
- Trending vs ranging market performance

**Expected Result:** B should have fewer trades but better quality

### Test 4: ATR Range Optimization

**Question:** What's the optimal volatility range?

**Setup:**
```
Variant A (Tight):
  ATR_MinLevel = 2.0
  ATR_MaxLevel = 6.0

Variant B (Moderate):
  ATR_MinLevel = 1.5
  ATR_MaxLevel = 8.0

Variant C (Wide):
  ATR_MinLevel = 1.0
  ATR_MaxLevel = 10.0

Duration: 2 weeks each
```

**Metrics to Track:**
- Market coverage (% of time with valid ATR)
- Performance in different volatility regimes
- Filter rejection rate

**Analysis:**
- Consider your trading hours
- London session needs wider range
- Asian session needs tighter range

---

## Parameter Tuning Workflow

### Step 1: Baseline Test (2 weeks)

Run with conservative defaults:
```mql5
RiskPercent = 0.5
ConfidenceThreshold = 0.65
EnableHybridValidation = true
All filters = default
```

**Record:**
- Total trades
- Win rate
- Profit factor
- Max drawdown
- Sharpe ratio

### Step 2: Identify Bottleneck

**Low trade count (<20/week)?**
→ Relax filters or lower confidence

**Low win rate (<65%)?**
→ Tighten filters or raise confidence

**High drawdown (>12%)?**
→ Reduce risk or tighten stops

**Low profit factor (<1.5)?**
→ Improve R:R or filter quality

### Step 3: Single Parameter Change

Change ONE parameter at a time.

**Example: If low trade count:**
```
Week 3: ConfidenceThreshold = 0.60 (was 0.65)
Week 4: Analyze results
```

**Compare:**
- Did trades increase?
- Did win rate change?
- Net impact on P/L?

### Step 4: Validate Change

If improvement confirmed:
- Keep new parameter
- Test for another 2 weeks
- Consider next optimization

If no improvement:
- Revert to previous value
- Try different parameter

### Step 5: Compound Changes

After finding improvements, test combinations:

```
Original: Conf=0.65, ADX=20
After individual tests: Conf=0.60, ADX=22

Test combined: Conf=0.60, ADX=22
```

Ensure improvements compound positively.

---

## Optimization DON'Ts

❌ **Don't over-optimize**
- Perfectly fitted to historical data
- Unrealistic metrics (PF > 4.0, Win rate > 90%)
- Too many parameters optimized at once

❌ **Don't ignore forward period**
- Always validate on unseen data
- 25% forward period minimum
- Parameters should work across both in-sample and out-of-sample

❌ **Don't chase best single result**
- Review top 10 results
- Look for parameter stability
- Avoid outliers

❌ **Don't optimize during news events**
- Results won't generalize
- Test on normal market conditions

❌ **Don't change parameters too frequently**
- Give each setting 2+ weeks
- Market conditions vary day-to-day
- Need sufficient sample size

❌ **Don't ignore practical constraints**
- Execution delays
- Spread widening
- Slippage
- Commissions

---

## Quick Parameter Recommendations

### "I want maximum safety"
```
→ Use Ultra-Conservative profile
→ Start with minimal capital allocation
→ Focus on Sharpe ratio
```

### "I want balanced performance"
```
→ Use Conservative profile (recommended)
→ Follow research-based defaults
→ Optimize win rate first
```

### "I want more trades"
```
→ Lower ConfidenceThreshold to 0.55-0.60
→ Relax ATR range (1.0-10.0)
→ Increase MaxTradesPerDay to 7-8
→ Keep hybrid validation enabled!
```

### "I want higher win rate"
```
→ Raise ConfidenceThreshold to 0.70-0.75
→ Tighten RSI levels (68/32)
→ Increase ADX to 22-25
→ Accept fewer trades
```

### "My accuracy is dropping"
```
→ Raise ConfidenceThreshold +0.05
→ Enable all hybrid filters
→ Require MTF alignment
→ Consider model retraining
```

---

**Last Updated:** 2025-12-22
**Version:** 1.0
**Recommended Review:** Monthly or when performance degrades
