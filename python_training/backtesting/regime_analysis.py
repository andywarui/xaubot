"""
Regime-Based Analysis for XAUUSD Neural Trading Bot.

Analyzes model performance across different market regimes:
- Trending Up (ADX > 25, Price > EMA)
- Trending Down (ADX > 25, Price < EMA)
- Ranging (ADX < 20)
- High Volatility (ATR > 1.5x avg)
- Low Volatility (ATR < 0.5x avg)

Phase 2.5 Implementation per XAUBOT_DEVELOPMENT_PLAN.md
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


class RegimeAnalyzer:
    """Regime-based performance analysis for trading strategies."""
    
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
        
        # Trading parameters
        self.tp_pips = 30
        self.sl_pips = 20
        self.spread_pips = 2.5
        
        # Regime thresholds
        self.adx_trend_threshold = 25
        self.adx_range_threshold = 20
        self.atr_high_mult = 1.5
        self.atr_low_mult = 0.5
    
    def load_data_with_indicators(self) -> pd.DataFrame:
        """Load hybrid features and calculate regime indicators."""
        print("📥 Loading data for regime analysis...")
        
        # Load hybrid features
        dfs = []
        for split in ["train", "val", "test"]:
            path = self.data_dir / f"hybrid_features_{split}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                dfs.append(df)
        
        df_all = pd.concat(dfs, ignore_index=True)
        
        # Load M1 data for additional indicators
        m1_path = self.data_dir / "xauusd_M1.parquet"
        if m1_path.exists():
            m1_df = pd.read_parquet(m1_path)
            if len(m1_df) >= len(df_all):
                df_all["time"] = pd.to_datetime(m1_df["time"].iloc[:len(df_all)].values)
                df_all["close"] = m1_df["close"].iloc[:len(df_all)].values
                df_all["high"] = m1_df["high"].iloc[:len(df_all)].values
                df_all["low"] = m1_df["low"].iloc[:len(df_all)].values
        
        df_all = df_all.sort_values("time").reset_index(drop=True)
        
        # Calculate regime indicators
        df_all = self._calculate_regime_indicators(df_all)
        
        print(f"   Loaded: {len(df_all):,} rows")
        
        return df_all
    
    def _calculate_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX, ATR and trend indicators for regime detection."""
        
        # True Range (use existing or calculate)
        if "tr" not in df.columns and all(c in df.columns for c in ["high", "low", "close"]):
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR (14-period)
        if "atr_14" not in df.columns and "tr" in df.columns:
            df["atr_14"] = df["tr"].rolling(14).mean()
        
        # EMA 50 for trend direction
        if "ema_50" not in df.columns and "close" in df.columns:
            df["ema_50"] = df["close"].ewm(span=50).mean()
        
        # ADX calculation
        if all(c in df.columns for c in ["high", "low", "close"]):
            df = self._calculate_adx(df, period=14)
        else:
            # Use ATR as proxy for ADX if we can't calculate
            if "atr_14" in df.columns:
                atr_sma = df["atr_14"].rolling(14).mean()
                df["adx"] = (df["atr_14"] / atr_sma).fillna(1) * 25  # Scale to ADX-like range
            else:
                df["adx"] = 20  # Default to ranging
        
        # Calculate ATR percentile for volatility regime
        if "atr_14" in df.columns:
            atr_mean = df["atr_14"].rolling(100).mean()
            df["atr_ratio"] = df["atr_14"] / atr_mean
            df["atr_ratio"] = df["atr_ratio"].fillna(1.0)
        else:
            df["atr_ratio"] = 1.0
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX indicator."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        
        # Directional movement
        up_move = np.diff(high, prepend=high[0])
        down_move = -np.diff(low, prepend=low[0])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        tr = df["tr"].values if "tr" in df.columns else np.maximum(
            high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        )
        
        # Using exponential smoothing
        def smooth(arr, period):
            result = np.zeros_like(arr)
            result[0] = arr[0]
            alpha = 1 / period
            for i in range(1, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
            return result
        
        atr_smooth = smooth(tr, period)
        plus_dm_smooth = smooth(plus_dm, period)
        minus_dm_smooth = smooth(minus_dm, period)
        
        # Directional indicators
        plus_di = 100 * plus_dm_smooth / np.where(atr_smooth > 0, atr_smooth, 1)
        minus_di = 100 * minus_dm_smooth / np.where(atr_smooth > 0, atr_smooth, 1)
        
        # DX and ADX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * di_diff / np.where(di_sum > 0, di_sum, 1)
        
        adx = smooth(dx, period)
        df["adx"] = adx
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        
        return df
    
    def classify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each bar into a regime."""
        
        # Trend regime based on ADX and price vs EMA
        conditions = [
            (df["adx"] > self.adx_trend_threshold) & (df["close"] > df["ema_50"]),
            (df["adx"] > self.adx_trend_threshold) & (df["close"] < df["ema_50"]),
            df["adx"] < self.adx_range_threshold,
        ]
        choices = ["TREND_UP", "TREND_DOWN", "RANGING"]
        df["trend_regime"] = np.select(conditions, choices, default="MIXED")
        
        # Volatility regime
        vol_conditions = [
            df["atr_ratio"] > self.atr_high_mult,
            df["atr_ratio"] < self.atr_low_mult,
        ]
        vol_choices = ["HIGH_VOL", "LOW_VOL"]
        df["vol_regime"] = np.select(vol_conditions, vol_choices, default="NORMAL_VOL")
        
        # Combined regime
        df["regime"] = df["trend_regime"] + "_" + df["vol_regime"]
        
        return df
    
    def load_model(self) -> lgb.Booster:
        """Load trained LightGBM model."""
        model_path = self.models_dir / "lightgbm_balanced.txt"
        return lgb.Booster(model_file=str(model_path))
    
    def analyze_regime(self, df_regime: pd.DataFrame, model: lgb.Booster) -> Dict:
        """Analyze model performance for a specific regime."""
        if len(df_regime) == 0:
            return None
        
        X = df_regime[self.feature_cols].values
        proba = model.predict(X)
        
        short_thresh = self.thresholds.get("SHORT", 0.48)
        long_thresh = self.thresholds.get("LONG", 0.40)
        
        trades = []
        for i in range(len(df_regime)):
            p = proba[i]
            label = df_regime["label"].iloc[i]
            
            if p[0] >= short_thresh and p[0] >= p[2]:
                direction = -1
            elif p[2] >= long_thresh and p[2] > p[0]:
                direction = 1
            else:
                continue
            
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
            
            trades.append({
                "direction": direction,
                "pnl_pips": pnl_pips,
                "is_winner": pnl_pips > 0
            })
        
        if len(trades) == 0:
            return None
        
        trades_df = pd.DataFrame(trades)
        
        win_rate = trades_df["is_winner"].mean()
        avg_trade = trades_df["pnl_pips"].mean()
        total_pnl = trades_df["pnl_pips"].sum()
        
        gross_profit = trades_df[trades_df["pnl_pips"] > 0]["pnl_pips"].sum()
        gross_loss = abs(trades_df[trades_df["pnl_pips"] < 0]["pnl_pips"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        
        return {
            "n_bars": len(df_regime),
            "n_trades": len(trades),
            "trade_frequency": len(trades) / len(df_regime) * 100,
            "win_rate": float(win_rate),
            "avg_trade_pips": float(avg_trade),
            "total_pnl_pips": float(total_pnl),
            "profit_factor": float(profit_factor),
            "long_trades": sum(1 for t in trades if t["direction"] == 1),
            "short_trades": sum(1 for t in trades if t["direction"] == -1),
        }
    
    def run_regime_analysis(self) -> Dict:
        """Execute full regime-based analysis."""
        print("=" * 70)
        print("REGIME-BASED ANALYSIS")
        print("=" * 70)
        print()
        
        df_all = self.load_data_with_indicators()
        df_all = self.classify_regime(df_all)
        model = self.load_model()
        
        # Print regime distribution
        print("\n📊 Regime Distribution:")
        trend_dist = df_all["trend_regime"].value_counts()
        vol_dist = df_all["vol_regime"].value_counts()
        
        print("\n   Trend Regimes:")
        for regime, count in trend_dist.items():
            print(f"      {regime}: {count:,} ({count/len(df_all)*100:.1f}%)")
        
        print("\n   Volatility Regimes:")
        for regime, count in vol_dist.items():
            print(f"      {regime}: {count:,} ({count/len(df_all)*100:.1f}%)")
        
        # Analyze each regime
        print("\n" + "=" * 70)
        print("PERFORMANCE BY REGIME")
        print("=" * 70)
        
        regime_results = {}
        
        # Trend regimes
        for regime in ["TREND_UP", "TREND_DOWN", "RANGING", "MIXED"]:
            df_regime = df_all[df_all["trend_regime"] == regime]
            result = self.analyze_regime(df_regime, model)
            
            if result:
                regime_results[regime] = result
                print(f"\n{'─' * 50}")
                print(f"📈 {regime}")
                print(f"   Bars: {result['n_bars']:,} | Trades: {result['n_trades']:,} "
                      f"({result['trade_frequency']:.2f}% freq)")
                print(f"   Win Rate: {result['win_rate']*100:.1f}%")
                print(f"   Avg Trade: {result['avg_trade_pips']:+.1f} pips")
                print(f"   Profit Factor: {result['profit_factor']:.2f}")
                print(f"   Total PnL: {result['total_pnl_pips']:+,.0f} pips")
        
        # Volatility regimes
        print("\n" + "=" * 70)
        print("PERFORMANCE BY VOLATILITY")
        print("=" * 70)
        
        for regime in ["HIGH_VOL", "NORMAL_VOL", "LOW_VOL"]:
            df_regime = df_all[df_all["vol_regime"] == regime]
            result = self.analyze_regime(df_regime, model)
            
            if result:
                regime_results[f"VOL_{regime}"] = result
                print(f"\n{'─' * 50}")
                print(f"📊 {regime}")
                print(f"   Bars: {result['n_bars']:,} | Trades: {result['n_trades']:,} "
                      f"({result['trade_frequency']:.2f}% freq)")
                print(f"   Win Rate: {result['win_rate']*100:.1f}%")
                print(f"   Avg Trade: {result['avg_trade_pips']:+.1f} pips")
                print(f"   Profit Factor: {result['profit_factor']:.2f}")
                print(f"   Total PnL: {result['total_pnl_pips']:+,.0f} pips")
        
        # Compile results
        results = {
            "run_date": datetime.now().isoformat(),
            "total_bars": len(df_all),
            "regime_distribution": {
                "trend": trend_dist.to_dict(),
                "volatility": vol_dist.to_dict(),
            },
            "regime_results": regime_results,
            "recommendations": self._generate_recommendations(regime_results),
        }
        
        # Print recommendations
        print("\n" + "=" * 70)
        print("REGIME RECOMMENDATIONS")
        print("=" * 70)
        
        for rec in results["recommendations"]:
            print(f"\n   {rec['icon']} {rec['regime']}: {rec['recommendation']}")
        
        # Save results
        output_path = self.results_dir / "regime_analysis_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Check success criteria
        self._check_success_criteria(results)
        
        return results
    
    def _generate_recommendations(self, regime_results: Dict) -> List[Dict]:
        """Generate trading recommendations per regime."""
        recommendations = []
        
        for regime, result in regime_results.items():
            if result["profit_factor"] >= 1.5 and result["win_rate"] >= 0.55:
                rec = "AGGRESSIVE - Increase position size"
                icon = "🟢"
            elif result["profit_factor"] >= 1.0 and result["win_rate"] >= 0.50:
                rec = "NORMAL - Standard trading"
                icon = "🟡"
            elif result["profit_factor"] >= 0.8:
                rec = "CAUTIOUS - Reduce position size"
                icon = "🟠"
            else:
                rec = "AVOID - Consider skipping trades"
                icon = "🔴"
            
            recommendations.append({
                "regime": regime,
                "recommendation": rec,
                "icon": icon,
                "profit_factor": result["profit_factor"],
                "win_rate": result["win_rate"],
            })
        
        return recommendations
    
    def _check_success_criteria(self, results: Dict):
        """Check if regime analysis meets success criteria."""
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 70)
        
        regime_results = results["regime_results"]
        
        # Check that we have profitable regimes
        profitable_regimes = sum(1 for r in regime_results.values() if r["profit_factor"] > 1.0)
        total_regimes = len(regime_results)
        
        criteria = {
            "At least 50% regimes profitable": profitable_regimes >= total_regimes / 2,
            "No regime with PF < 0.5": all(r["profit_factor"] >= 0.5 for r in regime_results.values()),
            "Average win rate > 45%": np.mean([r["win_rate"] for r in regime_results.values()]) > 0.45,
        }
        
        # Check trend regimes specifically
        for trend in ["TREND_UP", "TREND_DOWN"]:
            if trend in regime_results:
                criteria[f"{trend} profitable (PF > 1.0)"] = regime_results[trend]["profit_factor"] > 1.0
        
        all_pass = True
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
            if not passed:
                all_pass = False
        
        print()
        if all_pass:
            print("🎉 ALL REGIME CRITERIA PASSED!")
        else:
            print("⚠️ Some criteria not met - review results")


def main():
    project_root = Path(__file__).parent.parent.parent
    ra = RegimeAnalyzer(project_root)
    results = ra.run_regime_analysis()
    return results


if __name__ == "__main__":
    main()
