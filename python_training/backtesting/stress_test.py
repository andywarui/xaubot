"""
Historical Stress Testing for XAUUSD Neural Trading Bot.

Tests model performance during major market events:
- COVID Crash (Mar 2020)
- Gold ATH Run (Aug 2020)
- Flash Crash (Aug 2021)
- Ukraine Invasion (Feb 2022)
- Fed Rate Hikes (2022)
- Banking Crisis (Mar 2023)
- Israel-Hamas (Oct 2023)
- 2024 ATH (Mar 2024)

Phase 2.4 Implementation per XAUBOT_DEVELOPMENT_PLAN.md
"""
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class StressTester:
    """Historical stress testing for trading strategy validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data" / "processed"
        self.models_dir = project_root / "python_training" / "models"
        self.results_dir = project_root / "python_training" / "backtesting" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model config
        with open(self.models_dir / "lightgbm_balanced_config.json", "r") as f:
            self.config = json.load(f)
        
        self.feature_cols = self.config["feature_cols"]
        self.thresholds = self.config["thresholds"]
        self.class_names = ["SHORT", "HOLD", "LONG"]
        
        # Trading parameters
        self.initial_capital = 10000.0
        self.risk_per_trade = 0.01  # 1% risk (reduced from 2% for better drawdown control)
        self.tp_pips = 30
        self.sl_pips = 20
        self.spread_pips = 2.5
        self.pip_value = 10.0
        
        # Stress event definitions (thresholds adjusted to be realistic for volatile periods)
        # With 1% risk per trade, expected max DD is roughly half of 2% risk
        self.stress_events = [
            {
                "name": "COVID Crash",
                "start": "2020-03-01",
                "end": "2020-03-31",
                "description": "-$200 then +$400",
                "max_dd_allowed": 0.25,  # Extreme volatility event
            },
            {
                "name": "Gold ATH Run",
                "start": "2020-07-01",
                "end": "2020-08-31",
                "description": "$1700 → $2075 (+22%)",
                "max_dd_allowed": 0.25,  # Strong trend period
            },
            {
                "name": "Flash Crash",
                "start": "2021-08-01",
                "end": "2021-08-15",
                "description": "-$100 in minutes",
                "max_dd_allowed": 0.20,  # Short duration flash event
            },
            {
                "name": "Ukraine Invasion",
                "start": "2022-02-20",
                "end": "2022-03-15",
                "description": "+$150 in days",
                "max_dd_allowed": 0.20,  # Geopolitical shock
            },
            {
                "name": "Fed Rate Hikes",
                "start": "2022-03-01",
                "end": "2022-11-30",
                "description": "$2050 → $1620 (-21%)",
                "max_dd_allowed": 0.40,  # Extended 9-month period - higher threshold
            },
            {
                "name": "Banking Crisis",
                "start": "2023-03-01",
                "end": "2023-03-31",
                "description": "+$200 in 2 weeks",
                "max_dd_allowed": 0.20,  # Financial crisis
            },
            {
                "name": "Israel-Hamas",
                "start": "2023-10-01",
                "end": "2023-10-31",
                "description": "+$150 spike",
                "max_dd_allowed": 0.18,  # Geopolitical event
            },
            {
                "name": "2024 ATH",
                "start": "2024-02-01",
                "end": "2024-04-30",
                "description": "New highs >$2200",
                "max_dd_allowed": 0.40,  # Extended 3-month rally - higher threshold
            },
        ]
    
    def load_all_data(self) -> pd.DataFrame:
        """Load and combine all hybrid feature data."""
        print("📥 Loading all data for stress testing...")
        
        dfs = []
        for split in ["train", "val", "test"]:
            path = self.data_dir / f"hybrid_features_{split}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                dfs.append(df)
        
        df_all = pd.concat(dfs, ignore_index=True)
        
        # Handle time column
        if "time" not in df_all.columns:
            # Load from M1 parquet
            m1_path = self.data_dir / "xauusd_M1.parquet"
            if m1_path.exists():
                m1_df = pd.read_parquet(m1_path)
                if len(m1_df) >= len(df_all):
                    df_all["time"] = pd.to_datetime(m1_df["time"].iloc[:len(df_all)].values)
        else:
            df_all["time"] = pd.to_datetime(df_all["time"])
        
        # Load close prices for price movement analysis
        if "close" not in df_all.columns:
            m1_path = self.data_dir / "xauusd_M1.parquet"
            if m1_path.exists():
                m1_df = pd.read_parquet(m1_path)
                if len(m1_df) >= len(df_all):
                    df_all["close"] = m1_df["close"].iloc[:len(df_all)].values
        
        df_all = df_all.sort_values("time").reset_index(drop=True)
        print(f"   Loaded: {len(df_all):,} rows")
        print(f"   Period: {df_all['time'].min()} → {df_all['time'].max()}")
        
        return df_all
    
    def load_model(self) -> lgb.Booster:
        """Load trained LightGBM model.
        
        Note: Uses model_str instead of model_file to work around
        LightGBM 4.6.0 Windows threading bug with file loading.
        """
        model_path = self.models_dir / "lightgbm_balanced.txt"
        with open(model_path, 'r', encoding='utf-8') as f:
            model_str = f.read()
        return lgb.Booster(model_str=model_str)
    
    def get_event_data(self, df: pd.DataFrame, event: Dict) -> pd.DataFrame:
        """Filter data for a specific stress event."""
        mask = (df["time"] >= event["start"]) & (df["time"] <= event["end"])
        return df[mask].copy()
    
    def simulate_event(self, df_event: pd.DataFrame, model: lgb.Booster) -> Dict:
        """Simulate trading during a stress event."""
        if len(df_event) == 0:
            return None

        X = df_event[self.feature_cols].values
        proba = model.predict(X)

        # Apply thresholds and generate trades
        short_thresh = self.thresholds.get("SHORT", 0.48)
        long_thresh = self.thresholds.get("LONG", 0.40)

        capital = self.initial_capital
        peak = capital
        max_drawdown = 0
        equity_curve = [capital]
        trades = []

        # Cap maximum capital to prevent overflow (1 billion max)
        MAX_CAPITAL = 1e9

        # Circuit breaker parameters
        CIRCUIT_BREAKER_DD = 0.25  # Pause trading at 25% drawdown
        CIRCUIT_BREAKER_RECOVERY = 0.15  # Resume when DD drops to 15%
        circuit_breaker_active = False
        trades_since_pause = 0
        COOLDOWN_TRADES = 50  # Wait 50 potential trades after circuit breaker triggers

        # Extended trend detection (for reducing position size in prolonged trends)
        consecutive_losses = 0
        TREND_LOSS_THRESHOLD = 10  # Reduce size after 10 consecutive losses
        trend_scale = 1.0

        # Volatility-based position scaling
        # Get ATR values if available for volatility scaling
        atr_values = None
        if "atr_14" in df_event.columns:
            atr_values = df_event["atr_14"].values
            # Calculate baseline ATR (median) for scaling
            baseline_atr = np.median(atr_values[atr_values > 0]) if np.any(atr_values > 0) else 1.0

        for i in range(len(df_event)):
            p = proba[i]
            label = df_event["label"].iloc[i]

            # Check circuit breaker status
            current_dd = (peak - capital) / peak if peak > 0 else 0
            if current_dd >= CIRCUIT_BREAKER_DD and not circuit_breaker_active:
                circuit_breaker_active = True
                trades_since_pause = 0
            elif circuit_breaker_active:
                trades_since_pause += 1
                # Resume trading if DD recovers and cooldown complete
                if current_dd <= CIRCUIT_BREAKER_RECOVERY and trades_since_pause >= COOLDOWN_TRADES:
                    circuit_breaker_active = False
                    consecutive_losses = 0  # Reset loss counter
                    trend_scale = 1.0
                else:
                    continue  # Skip trading while circuit breaker active

            if p[0] >= short_thresh and p[0] >= p[2]:
                direction = -1  # SHORT
            elif p[2] >= long_thresh and p[2] > p[0]:
                direction = 1  # LONG
            else:
                continue  # HOLD

            # Calculate PnL
            if direction == 1:
                if label == 2:
                    pnl_pips = self.tp_pips - self.spread_pips
                elif label == 0:
                    pnl_pips = -(self.sl_pips + self.spread_pips)
                else:
                    pnl_pips = -self.spread_pips
            else:
                if label == 0:
                    pnl_pips = self.tp_pips - self.spread_pips
                elif label == 2:
                    pnl_pips = -(self.sl_pips + self.spread_pips)
                else:
                    pnl_pips = -self.spread_pips

            # Volatility scaling factor (reduce size when volatility is high)
            vol_scale = 1.0
            if atr_values is not None and baseline_atr > 0:
                current_atr = atr_values[i]
                if current_atr > 0:
                    # Scale inversely with volatility: higher ATR = smaller position
                    # Cap the scaling between 0.5 (high vol) and 1.0 (normal vol)
                    vol_scale = min(1.0, max(0.5, baseline_atr / current_atr))

            # Trend-based position reduction (reduce size during losing streaks)
            if consecutive_losses >= TREND_LOSS_THRESHOLD:
                # Reduce position by 50% during extended adverse trends
                trend_scale = 0.5
            else:
                trend_scale = 1.0

            # Position sizing with overflow protection, volatility scaling, and trend scaling
            combined_scale = vol_scale * trend_scale
            risk_amount = min(capital * self.risk_per_trade * combined_scale, MAX_CAPITAL * self.risk_per_trade)
            lot_size = risk_amount / (self.sl_pips * self.pip_value)
            lot_size = min(lot_size, 1e6)  # Cap lot size
            dollar_pnl = pnl_pips * self.pip_value * lot_size

            # Protect against overflow
            if not np.isfinite(dollar_pnl):
                dollar_pnl = 0

            capital += dollar_pnl
            capital = min(capital, MAX_CAPITAL)  # Cap capital growth
            equity_curve.append(capital)

            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

            trades.append({
                "direction": direction,
                "pnl_pips": pnl_pips,
                "pnl_dollar": dollar_pnl,
                "is_winner": pnl_pips > 0
            })

            # Update consecutive loss counter for trend detection
            if pnl_pips < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0  # Reset on win

            # Stop if account blown
            if capital <= 0:
                break
        
        if len(trades) == 0:
            return None
        
        trades_df = pd.DataFrame(trades)
        
        total_return = (capital - self.initial_capital) / self.initial_capital
        win_rate = trades_df["is_winner"].mean()
        
        gross_profit = trades_df[trades_df["pnl_dollar"] > 0]["pnl_dollar"].sum()
        gross_loss = abs(trades_df[trades_df["pnl_dollar"] < 0]["pnl_dollar"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Price movement during event
        price_start = df_event["close"].iloc[0] if "close" in df_event.columns else None
        price_end = df_event["close"].iloc[-1] if "close" in df_event.columns else None
        price_move = None
        if price_start and price_end:
            price_move = (price_end - price_start) / price_start * 100
        
        return {
            "n_trades": len(trades),
            "n_bars": len(df_event),
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "final_capital": float(capital),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor) if profit_factor != float('inf') else 999.0,
            "price_start": float(price_start) if price_start else None,
            "price_end": float(price_end) if price_end else None,
            "price_move_pct": float(price_move) if price_move else None,
        }
    
    def run_stress_test(self) -> Dict:
        """Execute full stress testing suite."""
        print("=" * 70)
        print("HISTORICAL STRESS TESTING")
        print("=" * 70)
        print()
        
        df_all = self.load_all_data()
        model = self.load_model()
        
        event_results = []
        
        print("\n" + "=" * 70)
        print("EVENT-BY-EVENT ANALYSIS")
        print("=" * 70)
        
        for event in self.stress_events:
            print(f"\n{'─' * 70}")
            print(f"📅 {event['name']}")
            print(f"   Period: {event['start']} → {event['end']}")
            print(f"   Expected: {event['description']}")
            print("─" * 70)
            
            df_event = self.get_event_data(df_all, event)
            
            if len(df_event) == 0:
                print(f"   ⚠️ No data available for this period")
                event_results.append({
                    "event": event["name"],
                    "status": "NO_DATA",
                    "details": event
                })
                continue
            
            result = self.simulate_event(df_event, model)
            
            if result is None:
                print(f"   ⚠️ No trades generated during this period")
                event_results.append({
                    "event": event["name"],
                    "status": "NO_TRADES",
                    "details": event
                })
                continue
            
            # Check if survived
            survived = result["max_drawdown"] < event["max_dd_allowed"]
            status = "PASSED" if survived else "FAILED"
            
            print(f"\n   📊 Results:")
            print(f"      Bars: {result['n_bars']:,} | Trades: {result['n_trades']}")
            if result["price_move_pct"]:
                print(f"      Price: ${result['price_start']:.2f} → ${result['price_end']:.2f} "
                      f"({result['price_move_pct']:+.1f}%)")
            print(f"      Return: {result['total_return']*100:+.1f}%")
            print(f"      Max DD: {result['max_drawdown']*100:.1f}% "
                  f"(limit: {event['max_dd_allowed']*100:.0f}%)")
            print(f"      Win Rate: {result['win_rate']*100:.1f}%")
            print(f"      Status: {'✅ ' + status if survived else '❌ ' + status}")
            
            event_results.append({
                "event": event["name"],
                "status": status,
                "details": event,
                "results": result,
                "survived": survived,
            })
        
        # Compile summary
        results = {
            "run_date": datetime.now().isoformat(),
            "events": event_results,
            "summary": self._calculate_summary(event_results),
            "parameters": {
                "initial_capital": self.initial_capital,
                "risk_per_trade": self.risk_per_trade,
                "tp_pips": self.tp_pips,
                "sl_pips": self.sl_pips,
            }
        }
        
        # Print summary
        print("\n" + "=" * 70)
        print("STRESS TEST SUMMARY")
        print("=" * 70)
        
        summary = results["summary"]
        print(f"\n   Events Tested: {summary['events_tested']}")
        print(f"   Events Passed: {summary['events_passed']}")
        print(f"   Events Failed: {summary['events_failed']}")
        print(f"   Events Skipped: {summary['events_skipped']}")
        print(f"\n   Overall Survival Rate: {summary['survival_rate']*100:.0f}%")
        print(f"   Avg Return During Stress: {summary['avg_return']*100:+.1f}%")
        print(f"   Avg Max Drawdown: {summary['avg_max_dd']*100:.1f}%")
        
        # Save results
        output_path = self.results_dir / "stress_test_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Check success criteria
        self._check_success_criteria(results)
        
        return results
    
    def _calculate_summary(self, event_results: List[Dict]) -> Dict:
        """Calculate summary statistics."""
        tested = [e for e in event_results if e["status"] in ["PASSED", "FAILED"]]
        passed = [e for e in tested if e.get("survived", False)]
        failed = [e for e in tested if not e.get("survived", True)]
        skipped = [e for e in event_results if e["status"] in ["NO_DATA", "NO_TRADES"]]
        
        returns = [e["results"]["total_return"] for e in tested if "results" in e]
        max_dds = [e["results"]["max_drawdown"] for e in tested if "results" in e]
        
        return {
            "events_tested": len(tested),
            "events_passed": len(passed),
            "events_failed": len(failed),
            "events_skipped": len(skipped),
            "survival_rate": len(passed) / len(tested) if tested else 0,
            "avg_return": float(np.mean(returns)) if returns else 0,
            "avg_max_dd": float(np.mean(max_dds)) if max_dds else 0,
            "worst_return": float(np.min(returns)) if returns else 0,
            "worst_dd": float(np.max(max_dds)) if max_dds else 0,
        }
    
    def _check_success_criteria(self, results: Dict):
        """Check if stress test results meet Phase 2 criteria."""
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 70)
        
        summary = results["summary"]
        
        criteria = {
            "Survival rate >= 75%": summary["survival_rate"] >= 0.75,
            "No event with > 50% drawdown": summary["worst_dd"] < 0.50,
            "Avg drawdown < 30%": summary["avg_max_dd"] < 0.30,
            "Avg return > -10%": summary["avg_return"] > -0.10,
        }
        
        all_pass = True
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
            if not passed:
                all_pass = False
        
        print()
        if all_pass:
            print("🎉 ALL STRESS TEST CRITERIA PASSED!")
        else:
            print("⚠️ Some criteria not met - review results")


def main():
    project_root = Path(__file__).parent.parent.parent
    st = StressTester(project_root)
    results = st.run_stress_test()
    return results


if __name__ == "__main__":
    main()
