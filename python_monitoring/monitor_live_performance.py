"""
Real-time MT5 EA Performance Monitor
Connects to MT5 via MetaTrader5 package and monitors:
- Prediction accuracy
- Filter rejection rates
- Model performance metrics
- Trade outcomes
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from pathlib import Path


class MT5PerformanceMonitor:
    """Monitor live EA performance and model drift"""

    def __init__(self):
        self.running = False
        self.metrics_history = []

    def initialize_mt5(self, account=None, password=None, server=None):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False

        if account and password and server:
            if not mt5.login(account, password, server):
                print(f"❌ Login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            print(f"✅ Connected to account {account} on {server}")
        else:
            print("✅ Connected to MT5 (using current terminal login)")

        # Check terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"   Terminal: {terminal_info.name}")
            print(f"   Build: {terminal_info.build}")
            print(f"   Connected: {terminal_info.connected}")

        return True

    def load_prediction_log(self, log_file="prediction_log.csv"):
        """Load EA prediction log from MT5 Files folder"""
        try:
            # Try multiple common locations
            possible_paths = [
                Path("MQL5/Files") / log_file,
                Path(mt5.terminal_info().data_path) / "MQL5" / "Files" / log_file,
                Path(mt5.terminal_info().commondata_path) / "Files" / log_file,
            ]

            for path in possible_paths:
                if path.exists():
                    df = pd.read_csv(path, sep=';')
                    df['time'] = pd.to_datetime(df['time'])
                    print(f"✅ Loaded {len(df)} predictions from {path}")
                    return df

            print(f"⚠️  Prediction log not found. Checked:")
            for path in possible_paths:
                print(f"   - {path}")
            return None

        except Exception as e:
            print(f"❌ Error loading prediction log: {e}")
            return None

    def calculate_actual_outcomes(self, predictions_df, lookback_bars=15):
        """
        Calculate actual market outcomes for predictions

        Args:
            predictions_df: DataFrame with predictions
            lookback_bars: Bars to look ahead for outcome (default 15 = 15 minutes)
        """
        if predictions_df is None or len(predictions_df) == 0:
            return None

        # Get historical prices
        symbol = "XAUUSD"
        start_time = predictions_df['time'].min()
        end_time = datetime.now()

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_time, end_time)
        if rates is None:
            print(f"❌ Failed to fetch price data: {mt5.last_error()}")
            return None

        rates_df = pd.DataFrame(rates)
        rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')

        # Merge predictions with prices
        merged = pd.merge_asof(
            predictions_df.sort_values('time'),
            rates_df[['time', 'close']].sort_values('time'),
            on='time',
            direction='nearest',
            tolerance=pd.Timedelta('2min')
        )

        # Calculate future price
        merged['future_close'] = merged['close'].shift(-lookback_bars)
        merged['actual_return'] = merged['future_close'] - merged['close']

        # Classify actual outcome (similar to training labels)
        merged['actual_label'] = np.where(
            merged['actual_return'] > 0.5, 2,  # LONG
            np.where(merged['actual_return'] < -0.5, 0, 1)  # SHORT or HOLD
        )

        # Remove last N rows (no future data)
        merged = merged[:-lookback_bars]

        return merged

    def analyze_model_performance(self, predictions_with_outcomes):
        """Analyze model accuracy and performance"""
        if predictions_with_outcomes is None or len(predictions_with_outcomes) == 0:
            return None

        df = predictions_with_outcomes.dropna()

        # Overall accuracy
        accuracy = (df['best_class'] == df['actual_label']).mean()

        # Per-class accuracy
        class_names = ['SHORT', 'HOLD', 'LONG']
        class_accuracies = {}
        for cls, name in enumerate(class_names):
            mask = df['best_class'] == cls
            if mask.sum() > 0:
                class_acc = (df[mask]['best_class'] == df[mask]['actual_label']).mean()
                class_accuracies[name] = class_acc
            else:
                class_accuracies[name] = 0.0

        # Confidence analysis
        high_conf = df[df['best_prob'] >= 0.65]
        high_conf_accuracy = (
            (high_conf['best_class'] == high_conf['actual_label']).mean()
            if len(high_conf) > 0 else 0.0
        )

        # Prediction distribution
        pred_dist = df['best_class'].value_counts().to_dict()

        return {
            'overall_accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'high_confidence_accuracy': high_conf_accuracy,
            'prediction_distribution': pred_dist,
            'total_predictions': len(df),
            'timestamp': datetime.now().isoformat()
        }

    def check_positions(self, symbol="XAUUSD"):
        """Get current open positions"""
        positions = mt5.positions_get(symbol=symbol)

        if positions is None:
            return []

        position_data = []
        for pos in positions:
            position_data.append({
                'ticket': pos.ticket,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'sl': pos.sl,
                'tp': pos.tp,
                'time': datetime.fromtimestamp(pos.time).isoformat(),
                'comment': pos.comment
            })

        return position_data

    def get_account_metrics(self):
        """Get account performance metrics"""
        account_info = mt5.account_info()
        if account_info is None:
            return None

        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'profit': account_info.profit,
            'margin': account_info.margin,
            'margin_free': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'timestamp': datetime.now().isoformat()
        }

    def monitor_loop(self, interval_seconds=60, duration_hours=None):
        """
        Main monitoring loop

        Args:
            interval_seconds: Check interval (default: 60s)
            duration_hours: Run duration in hours (None = indefinite)
        """
        self.running = True
        start_time = datetime.now()

        print("\n" + "=" * 70)
        print("🔍 LIVE MONITORING STARTED")
        print("=" * 70)
        print(f"Start time: {start_time}")
        print(f"Check interval: {interval_seconds}s")
        print(f"Duration: {'Indefinite' if duration_hours is None else f'{duration_hours}h'}")
        print("=" * 70)
        print()

        try:
            while self.running:
                # Check if duration exceeded
                if duration_hours and (datetime.now() - start_time).total_seconds() / 3600 > duration_hours:
                    print(f"\n⏰ Monitoring duration reached ({duration_hours}h)")
                    break

                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking...")

                # 1. Load predictions
                pred_df = self.load_prediction_log()

                # 2. Calculate outcomes
                if pred_df is not None:
                    pred_with_outcomes = self.calculate_actual_outcomes(pred_df)

                    # 3. Analyze performance
                    if pred_with_outcomes is not None:
                        metrics = self.analyze_model_performance(pred_with_outcomes)

                        if metrics:
                            print(f"\n📊 MODEL PERFORMANCE (last {metrics['total_predictions']} predictions):")
                            print(f"   Overall Accuracy: {metrics['overall_accuracy']:.1%}")
                            print(f"   HIGH signals: {metrics['class_accuracies'].get('LONG', 0):.1%}")
                            print(f"   HOLD signals: {metrics['class_accuracies'].get('HOLD', 0):.1%}")
                            print(f"   SHORT signals: {metrics['class_accuracies'].get('SHORT', 0):.1%}")
                            print(f"   High confidence (≥65%): {metrics['high_confidence_accuracy']:.1%}")

                            # Check for degradation
                            if metrics['overall_accuracy'] < 0.55:
                                print(f"\n⚠️  WARNING: Model accuracy dropped below 55%!")
                                print(f"   Consider retraining or adjusting parameters")

                            self.metrics_history.append(metrics)

                # 4. Check positions
                positions = self.check_positions()
                if positions:
                    print(f"\n💼 OPEN POSITIONS ({len(positions)}):")
                    for pos in positions:
                        profit_sign = "+" if pos['profit'] >= 0 else ""
                        print(f"   #{pos['ticket']} {pos['type']} {pos['volume']} lots "
                              f"@ {pos['price_open']} | P/L: {profit_sign}${pos['profit']:.2f} | {pos['comment']}")
                else:
                    print("\n💼 No open positions")

                # 5. Account metrics
                account = self.get_account_metrics()
                if account:
                    print(f"\n💰 ACCOUNT:")
                    print(f"   Balance: ${account['balance']:.2f}")
                    print(f"   Equity: ${account['equity']:.2f}")
                    print(f"   Profit: ${account['profit']:.2f}")
                    print(f"   Margin Level: {account['margin_level']:.1f}%")

                # Wait for next check
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
        finally:
            self.running = False
            mt5.shutdown()
            print("\n✅ MT5 connection closed")

            # Save metrics history
            if self.metrics_history:
                self.save_metrics_history()

    def save_metrics_history(self):
        """Save collected metrics to JSON"""
        output_file = Path("monitoring_results.json")
        with open(output_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        print(f"✅ Metrics saved to {output_file}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor MT5 EA performance in real-time")
    parser.add_argument('--account', type=int, help='MT5 account number')
    parser.add_argument('--password', type=str, help='MT5 password')
    parser.add_argument('--server', type=str, help='MT5 server')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    parser.add_argument('--duration', type=float, help='Duration in hours (default: indefinite)')

    args = parser.parse_args()

    # Create monitor
    monitor = MT5PerformanceMonitor()

    # Initialize MT5
    if not monitor.initialize_mt5(args.account, args.password, args.server):
        print("❌ Failed to initialize MT5")
        return

    # Start monitoring
    try:
        monitor.monitor_loop(
            interval_seconds=args.interval,
            duration_hours=args.duration
        )
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
