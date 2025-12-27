"""
XAUUSD Neural Bot - Python Backtesting Engine
Replicates MT5 EA logic for comprehensive A/B testing

This backtesting engine implements the same trading logic as the MT5 Expert Advisors
to enable scientific comparison of Single vs Ensemble models in a Python environment.
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

class XAUUSDBacktester:
    """
    Backtesting engine that replicates MT5 EA trading logic

    Features:
    - 26-feature calculation for LightGBM
    - 130-feature calculation for Transformer (5 timeframes × 26 features)
    - 6-layer hybrid validation system
    - Position sizing with risk management
    - Comprehensive performance metrics
    """

    def __init__(self,
                 initial_balance: float = 10000.0,
                 risk_percent: float = 0.5,
                 confidence_threshold: float = 0.60,
                 max_trades_per_day: int = 5,
                 atr_multiplier_sl: float = 1.5,
                 risk_reward_ratio: float = 2.0):
        """
        Initialize backtester

        Args:
            initial_balance: Starting account balance ($)
            risk_percent: Risk per trade (%)
            confidence_threshold: Minimum ML confidence (0-1)
            max_trades_per_day: Maximum trades per day
            atr_multiplier_sl: ATR multiplier for stop loss (default 1.5)
            risk_reward_ratio: TP/SL ratio (default 2.0 for 2:1 RR)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.risk_percent = risk_percent
        self.confidence_threshold = confidence_threshold
        self.max_trades_per_day = max_trades_per_day
        self.atr_multiplier_sl = atr_multiplier_sl
        self.risk_reward_ratio = risk_reward_ratio

        # Hybrid validation parameters
        self.max_spread_usd = 2.0
        self.rsi_overbought = 70.0
        self.rsi_oversold = 30.0
        self.atr_min = 1.5
        self.atr_max = 8.0
        self.adx_min_strength = 20.0
        self.require_mtf_alignment = True

        # Trading state
        self.trades = []
        self.open_position = None
        self.daily_trades = 0
        self.current_date = None

        # Models
        self.lightgbm_session = None
        self.transformer_session = None

        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': initial_balance,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }

    def load_lightgbm_model(self, model_path: str):
        """Load LightGBM ONNX model"""
        print(f"Loading LightGBM model from {model_path}")
        self.lightgbm_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        print("✓ LightGBM model loaded")

    def load_transformer_model(self, model_path: str, scaler_path: str):
        """Load Transformer ONNX model and scaler parameters"""
        print(f"Loading Transformer model from {model_path}")
        self.transformer_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )

        # Load scaler parameters
        with open(scaler_path, 'r') as f:
            self.scaler_params = json.load(f)

        print("✓ Transformer model and scaler loaded")

    def calculate_27_features(self, data: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Calculate 27 features for real data LightGBM model

        Features (matching trained model):
        - multi_tf_signal (placeholder: 0.0)
        - body, body_abs, candle_range, close_position
        - return_1, return_5, return_15, return_60
        - tr, atr_14, rsi_14, ema_10, ema_20, ema_50
        - hour_sin, hour_cos
        - M5_trend, M5_position, M15_trend, M15_position
        - H1_trend, H1_position, H4_trend, H4_position
        - D1_trend, D1_position

        Args:
            data: DataFrame with OHLCV data and indicators
            idx: Current index in dataframe

        Returns:
            numpy array of 27 features
        """
        features = np.zeros(27, dtype=np.float32)

        if idx < 60:  # Need at least 60 bars for returns
            return features

        row = data.iloc[idx]

        # Feature 0: multi_tf_signal (placeholder - set to 0.0 for now)
        features[0] = 0.0

        # Price features
        body = row['close'] - row['open']
        features[1] = body  # body
        features[2] = abs(body)  # body_abs
        features[3] = row['high'] - row['low']  # candle_range
        features[4] = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-8)  # close_position

        # Returns
        features[5] = (data.iloc[idx]['close'] / data.iloc[idx-1]['close']) - 1.0  # return_1
        features[6] = (data.iloc[idx]['close'] / data.iloc[idx-5]['close']) - 1.0  # return_5
        features[7] = (data.iloc[idx]['close'] / data.iloc[idx-15]['close']) - 1.0  # return_15
        features[8] = (data.iloc[idx]['close'] / data.iloc[idx-60]['close']) - 1.0  # return_60

        # Technical indicators (pre-calculated in data)
        features[9] = row.get('tr', 0.0)  # tr
        features[10] = row.get('atr_14', 0.0)  # atr_14
        features[11] = row.get('rsi_14', 50.0)  # rsi_14
        features[12] = row.get('ema_10', row['close'])  # ema_10
        features[13] = row.get('ema_20', row['close'])  # ema_20
        features[14] = row.get('ema_50', row['close'])  # ema_50

        # Time features
        hour = row['time'].hour if 'time' in row.index else 0
        features[15] = np.sin(2 * np.pi * hour / 24)  # hour_sin
        features[16] = np.cos(2 * np.pi * hour / 24)  # hour_cos

        # Multi-timeframe features (M5, M15, H1, H4, D1)
        # These would be calculated from aggregated data
        # For simplicity, using placeholder values (0.0)
        for i in range(17, 27):
            features[i] = 0.0

        return features

    def calculate_26_features(self, data: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Calculate 26 features for LightGBM model

        Features (matching MT5 EA):
        - body, body_abs, candle_range, close_position
        - return_1, return_5, return_15, return_60
        - tr, atr_14, rsi_14, ema_10, ema_20, ema_50
        - hour_sin, hour_cos
        - M5_trend, M5_position, M15_trend, M15_position
        - H1_trend, H1_position, H4_trend, H4_position
        - D1_trend, D1_position

        Args:
            data: DataFrame with OHLCV data and indicators
            idx: Current index in dataframe

        Returns:
            numpy array of 26 features
        """
        features = np.zeros(26, dtype=np.float32)

        if idx < 60:  # Need at least 60 bars for returns
            return features

        row = data.iloc[idx]

        # Price features
        body = row['close'] - row['open']
        features[0] = body  # body
        features[1] = abs(body)  # body_abs
        features[2] = row['high'] - row['low']  # candle_range
        features[3] = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-8)  # close_position

        # Returns
        features[4] = (data.iloc[idx]['close'] / data.iloc[idx-1]['close']) - 1.0  # return_1
        features[5] = (data.iloc[idx]['close'] / data.iloc[idx-5]['close']) - 1.0  # return_5
        features[6] = (data.iloc[idx]['close'] / data.iloc[idx-15]['close']) - 1.0  # return_15
        features[7] = (data.iloc[idx]['close'] / data.iloc[idx-60]['close']) - 1.0  # return_60

        # Technical indicators (pre-calculated in data)
        features[8] = row.get('tr', 0.0)  # tr
        features[9] = row.get('atr_14', 0.0)  # atr_14
        features[10] = row.get('rsi_14', 50.0)  # rsi_14
        features[11] = row.get('ema_10', row['close'])  # ema_10
        features[12] = row.get('ema_20', row['close'])  # ema_20
        features[13] = row.get('ema_50', row['close'])  # ema_50

        # Time features
        hour = row['time'].hour if 'time' in row.index else 0
        features[14] = np.sin(2 * np.pi * hour / 24)  # hour_sin
        features[15] = np.cos(2 * np.pi * hour / 24)  # hour_cos

        # Multi-timeframe features (M5, M15, H1, H4, D1)
        # These would be calculated from aggregated data
        # For simplicity, using placeholder values (0.0)
        # In full implementation, aggregate M1 to other timeframes
        for i in range(16, 26):
            features[i] = 0.0

        return features

    def predict_lightgbm(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Get LightGBM prediction

        Args:
            features: 26-element feature array

        Returns:
            (signal, confidence) where signal is 0=SHORT, 1=HOLD, 2=LONG
        """
        if self.lightgbm_session is None:
            return (1, 0.0)  # HOLD with 0 confidence

        # Reshape for ONNX: [1, 26]
        input_data = features.reshape(1, -1).astype(np.float32)

        # Get model output
        outputs = self.lightgbm_session.run(None, {'input': input_data})

        # LightGBM ONNX format:
        # outputs[0] = label (predicted class)
        # outputs[1] = list of dicts with probabilities
        prob_dict = outputs[1][0]  # Get probability dictionary

        # Find class with highest probability
        predicted_class = max(prob_dict.keys(), key=lambda k: prob_dict[k])
        confidence = float(prob_dict[predicted_class])

        return (predicted_class, confidence)

    def validate_long_signal(self, data: pd.DataFrame, idx: int, ml_confidence: float) -> bool:
        """
        6-layer hybrid validation for LONG signals

        Layers:
        1. Spread filter
        2. RSI filter (avoid overbought)
        3. MACD alignment
        4. ADX trend strength
        5. ATR volatility filter
        6. Multi-timeframe EMA alignment

        Args:
            data: DataFrame with market data and indicators
            idx: Current index
            ml_confidence: ML model confidence

        Returns:
            True if signal passes all validation layers
        """
        row = data.iloc[idx]

        # Layer 1: Spread filter
        spread = row.get('spread', 0.5)
        if spread > self.max_spread_usd:
            return False

        # Layer 2: RSI filter (avoid overbought)
        rsi = row.get('rsi_14', 50.0)
        if rsi > self.rsi_overbought:
            return False

        # Layer 3: MACD alignment (bullish)
        macd = row.get('macd', 0.0)
        macd_signal = row.get('macd_signal', 0.0)
        if macd <= macd_signal:
            return False

        # Layer 4: ADX trend strength
        adx = row.get('adx', 0.0)
        if adx < self.adx_min_strength:
            return False

        # Layer 5: ATR volatility filter
        atr = row.get('atr_14', 0.0)
        if atr < self.atr_min or atr > self.atr_max:
            return False

        # Layer 6: Multi-timeframe EMA alignment (bullish)
        if self.require_mtf_alignment:
            close = row['close']
            ema_20_m15 = row.get('ema_20_m15', close)
            ema_20_h1 = row.get('ema_20_h1', close)

            # Price should be above both EMAs
            if close <= ema_20_m15 or close <= ema_20_h1:
                return False

        return True

    def validate_short_signal(self, data: pd.DataFrame, idx: int, ml_confidence: float) -> bool:
        """
        6-layer hybrid validation for SHORT signals

        Same layers as LONG but with bearish conditions
        """
        row = data.iloc[idx]

        # Layer 1: Spread filter
        spread = row.get('spread', 0.5)
        if spread > self.max_spread_usd:
            return False

        # Layer 2: RSI filter (avoid oversold)
        rsi = row.get('rsi_14', 50.0)
        if rsi < self.rsi_oversold:
            return False

        # Layer 3: MACD alignment (bearish)
        macd = row.get('macd', 0.0)
        macd_signal = row.get('macd_signal', 0.0)
        if macd >= macd_signal:
            return False

        # Layer 4: ADX trend strength
        adx = row.get('adx', 0.0)
        if adx < self.adx_min_strength:
            return False

        # Layer 5: ATR volatility filter
        atr = row.get('atr_14', 0.0)
        if atr < self.atr_min or atr > self.atr_max:
            return False

        # Layer 6: Multi-timeframe EMA alignment (bearish)
        if self.require_mtf_alignment:
            close = row['close']
            ema_20_m15 = row.get('ema_20_m15', close)
            ema_20_h1 = row.get('ema_20_h1', close)

            # Price should be below both EMAs
            if close >= ema_20_m15 or close >= ema_20_h1:
                return False

        return True

    def calculate_position_size(self, price: float, stop_loss_usd: float) -> float:
        """
        Calculate position size based on risk management

        Args:
            price: Current price
            stop_loss_usd: Stop loss in USD

        Returns:
            Position size in lots (0.01 = 1 troy ounce)
        """
        # Calculate risk amount in dollars
        risk_amount = self.balance * (self.risk_percent / 100.0)

        # Calculate position size
        # For XAUUSD, 1 lot = 100 oz, 0.01 lot = 1 oz
        # Position size = risk amount / (stop loss in USD / price) / 100
        position_size = risk_amount / stop_loss_usd

        # Convert to lots (1 oz = 0.01 lot)
        lots = position_size * price / 100.0 / 100.0  # Simplified calculation

        # Ensure minimum/maximum lot size
        lots = max(0.01, min(lots, 1.0))

        return lots

    def open_trade(self, signal: int, entry_price: float, entry_time: datetime, atr: float = 3.0):
        """
        Open a new trade with dynamic ATR-based TP/SL

        Args:
            signal: 0=SHORT, 2=LONG
            entry_price: Entry price
            entry_time: Entry timestamp
            atr: Current ATR value for dynamic SL calculation
        """
        if self.open_position is not None:
            return  # Already have an open position

        # Calculate dynamic SL distance based on ATR
        sl_distance = atr * self.atr_multiplier_sl

        # Calculate TP distance using risk-reward ratio
        tp_distance = sl_distance * self.risk_reward_ratio

        # Calculate position size based on SL distance
        lots = self.calculate_position_size(entry_price, sl_distance)

        # Set SL/TP based on direction
        if signal == 2:  # LONG
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT (signal == 0)
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        self.open_position = {
            'signal': signal,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'lots': lots,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance
        }

        self.daily_trades += 1

    def update_position(self, current_price: float, current_time: datetime):
        """Update open position and check for exit"""
        if self.open_position is None:
            return

        pos = self.open_position
        signal = pos['signal']
        entry_price = pos['entry_price']
        lots = pos['lots']

        # Check for SL/TP hit
        hit_sl = False
        hit_tp = False

        if signal == 2:  # LONG
            if current_price <= pos['stop_loss']:
                hit_sl = True
            elif current_price >= pos['take_profit']:
                hit_tp = True
        else:  # SHORT
            if current_price >= pos['stop_loss']:
                hit_sl = True
            elif current_price <= pos['take_profit']:
                hit_tp = True

        if hit_sl or hit_tp:
            # Close position
            exit_price = pos['stop_loss'] if hit_sl else pos['take_profit']

            # Calculate P/L
            if signal == 2:  # LONG
                profit = (exit_price - entry_price) * lots * 100  # Simplified
            else:  # SHORT
                profit = (entry_price - exit_price) * lots * 100

            # Update balance
            self.balance += profit
            self.equity = self.balance

            # Update statistics
            self.stats['total_trades'] += 1
            if profit > 0:
                self.stats['winning_trades'] += 1
                self.stats['total_profit'] += profit
                self.stats['consecutive_wins'] += 1
                self.stats['consecutive_losses'] = 0
                self.stats['max_consecutive_wins'] = max(
                    self.stats['max_consecutive_wins'],
                    self.stats['consecutive_wins']
                )
            else:
                self.stats['losing_trades'] += 1
                self.stats['total_loss'] += abs(profit)
                self.stats['consecutive_losses'] += 1
                self.stats['consecutive_wins'] = 0
                self.stats['max_consecutive_losses'] = max(
                    self.stats['max_consecutive_losses'],
                    self.stats['consecutive_losses']
                )

            # Update peak equity and drawdown
            if self.equity > self.stats['peak_equity']:
                self.stats['peak_equity'] = self.equity
            else:
                drawdown = (self.stats['peak_equity'] - self.equity) / self.stats['peak_equity']
                self.stats['max_drawdown'] = max(self.stats['max_drawdown'], drawdown)

            # Record trade
            self.trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': current_time,
                'signal': 'LONG' if signal == 2 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'lots': lots,
                'profit': profit,
                'exit_reason': 'SL' if hit_sl else 'TP'
            })

            # Clear position
            self.open_position = None

    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        stats = self.stats.copy()

        # Calculate derived metrics
        total_trades = stats['total_trades']
        if total_trades > 0:
            stats['win_rate'] = (stats['winning_trades'] / total_trades) * 100
            stats['avg_win'] = stats['total_profit'] / max(1, stats['winning_trades'])
            stats['avg_loss'] = stats['total_loss'] / max(1, stats['losing_trades'])

            if stats['total_loss'] > 0:
                stats['profit_factor'] = stats['total_profit'] / stats['total_loss']
            else:
                stats['profit_factor'] = float('inf') if stats['total_profit'] > 0 else 0.0

            stats['net_profit'] = stats['total_profit'] - stats['total_loss']
            stats['return_pct'] = ((self.balance - self.initial_balance) / self.initial_balance) * 100

            if stats['max_drawdown'] > 0:
                stats['recovery_factor'] = stats['net_profit'] / (stats['max_drawdown'] * self.initial_balance)
            else:
                stats['recovery_factor'] = float('inf') if stats['net_profit'] > 0 else 0.0
        else:
            stats['win_rate'] = 0.0
            stats['avg_win'] = 0.0
            stats['avg_loss'] = 0.0
            stats['profit_factor'] = 0.0
            stats['net_profit'] = 0.0
            stats['return_pct'] = 0.0
            stats['recovery_factor'] = 0.0

        stats['final_balance'] = self.balance
        stats['max_drawdown_pct'] = stats['max_drawdown'] * 100

        return stats

    def print_summary(self):
        """Print backtest summary"""
        metrics = self.get_performance_metrics()

        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"Initial Balance:        ${self.initial_balance:,.2f}")
        print(f"Final Balance:          ${metrics['final_balance']:,.2f}")
        print(f"Net Profit:             ${metrics['net_profit']:,.2f}")
        print(f"Return:                 {metrics['return_pct']:.2f}%")
        print()
        print(f"Total Trades:           {metrics['total_trades']}")
        print(f"Winning Trades:         {metrics['winning_trades']} ({metrics['win_rate']:.2f}%)")
        print(f"Losing Trades:          {metrics['losing_trades']}")
        print()
        print(f"Profit Factor:          {metrics['profit_factor']:.2f}")
        print(f"Average Win:            ${metrics['avg_win']:.2f}")
        print(f"Average Loss:           ${metrics['avg_loss']:.2f}")
        print()
        print(f"Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
        print(f"Recovery Factor:        {metrics['recovery_factor']:.2f}")
        print(f"Max Consecutive Wins:   {metrics['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        print("="*70)
