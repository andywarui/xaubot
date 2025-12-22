"""
Intra-Bar Feature Engineering for XAUUSD Neural Trading Bot.

Implements advanced features computed from higher-frequency data within each bar:
1. Hurst Exponent - Market regime indicator (trending vs mean-reverting)
2. Intra-bar Volatility - Volatility within each candle
3. Intra-bar Momentum - Price momentum within candles
4. Microstructure Features - Order flow patterns

Based on Quantreo's intra-bar feature engineering approach.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import hurst library, provide fallback if not available
try:
    from hurst import compute_Hc
    HURST_AVAILABLE = True
except ImportError:
    HURST_AVAILABLE = False
    print("Warning: 'hurst' library not installed. Install with: pip install hurst")


def compute_hurst_exponent(series: pd.Series, min_samples: int = 100) -> float:
    """
    Compute Hurst exponent for a price series.

    Returns:
        H value:
        - H > 0.5: Trending (persistent) behavior
        - H = 0.5: Random walk
        - H < 0.5: Mean-reverting behavior
    """
    if not HURST_AVAILABLE:
        return 0.5  # Return neutral if library not available

    if len(series) < min_samples:
        return np.nan

    try:
        # Remove NaN and zero values
        clean_series = series.dropna()
        clean_series = clean_series[clean_series > 0]

        if len(clean_series) < min_samples:
            return np.nan

        H, c, data = compute_Hc(clean_series.values, kind='price', simplified=True)
        return H
    except Exception:
        return np.nan


def compute_intra_bar_volatility(high_freq_df: pd.DataFrame,
                                  resample_rule: str = '4H') -> pd.Series:
    """
    Compute volatility of returns within each higher timeframe bar.

    Uses standard deviation of 1-minute returns within each 4H bar.
    """
    # Calculate 1-minute returns
    returns = high_freq_df['close'].pct_change()

    # Resample and compute std of returns within each bar
    intra_vol = returns.resample(resample_rule).std()

    return intra_vol


def compute_intra_bar_momentum(high_freq_df: pd.DataFrame,
                                resample_rule: str = '4H') -> pd.Series:
    """
    Compute momentum within each higher timeframe bar.

    Measures how much of the bar's range was achieved early vs late.
    Positive = bullish momentum (price rose throughout)
    Negative = bearish momentum (price fell throughout)
    """
    def momentum_within_bar(group):
        if len(group) < 2:
            return 0

        closes = group['close'].values

        # Split bar into first half and second half
        mid = len(closes) // 2
        first_half_return = (closes[mid] - closes[0]) / (closes[0] + 1e-10)
        second_half_return = (closes[-1] - closes[mid]) / (closes[mid] + 1e-10)

        # Momentum: positive if second half stronger than first
        return second_half_return - first_half_return

    momentum = high_freq_df.resample(resample_rule).apply(momentum_within_bar)

    return momentum['close'] if isinstance(momentum, pd.DataFrame) else momentum


def compute_intra_bar_efficiency(high_freq_df: pd.DataFrame,
                                  resample_rule: str = '4H') -> pd.Series:
    """
    Compute price efficiency within each bar.

    Efficiency = Net move / Total distance traveled
    - High efficiency (close to 1): Strong trend within bar
    - Low efficiency (close to 0): Choppy, ranging within bar
    """
    def efficiency_within_bar(group):
        if len(group) < 2:
            return 0.5

        closes = group['close'].values

        # Net move (absolute)
        net_move = abs(closes[-1] - closes[0])

        # Total distance traveled
        total_distance = np.sum(np.abs(np.diff(closes)))

        if total_distance == 0:
            return 0.5

        return net_move / total_distance

    efficiency = high_freq_df.resample(resample_rule).apply(efficiency_within_bar)

    return efficiency['close'] if isinstance(efficiency, pd.DataFrame) else efficiency


def compute_intra_bar_volume_profile(high_freq_df: pd.DataFrame,
                                      resample_rule: str = '4H') -> Tuple[pd.Series, pd.Series]:
    """
    Compute volume distribution within each bar.

    Returns:
        volume_skew: Whether volume concentrated at start or end
        volume_kurtosis: Whether volume had spikes or was uniform
    """
    def volume_skew_within_bar(group):
        if len(group) < 4:
            return 0

        volumes = group['volume'].values
        n = len(volumes)

        # Compare first half vs second half volume
        mid = n // 2
        first_half = volumes[:mid].sum()
        second_half = volumes[mid:].sum()
        total = first_half + second_half + 1e-10

        # Positive = more volume in second half
        return (second_half - first_half) / total

    def volume_concentration(group):
        if len(group) < 4:
            return 0

        volumes = group['volume'].values

        # Normalized volumes
        total = volumes.sum() + 1e-10
        normalized = volumes / total

        # Entropy-based concentration (low entropy = concentrated)
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        max_entropy = np.log(len(volumes))

        return 1 - (entropy / (max_entropy + 1e-10))  # 0=uniform, 1=concentrated

    skew = high_freq_df.resample(resample_rule).apply(volume_skew_within_bar)
    concentration = high_freq_df.resample(resample_rule).apply(volume_concentration)

    skew_series = skew['volume'] if isinstance(skew, pd.DataFrame) else skew
    conc_series = concentration['volume'] if isinstance(concentration, pd.DataFrame) else concentration

    return skew_series, conc_series


def build_intra_bar_features(m1_df: pd.DataFrame,
                              target_tf: str = '4H',
                              compute_hurst: bool = True) -> pd.DataFrame:
    """
    Build all intra-bar features from M1 data.

    Args:
        m1_df: DataFrame with M1 OHLCV data (must have 'time' column)
        target_tf: Target timeframe for aggregation (default '4H')
        compute_hurst: Whether to compute Hurst exponent (slow)

    Returns:
        DataFrame with intra-bar features indexed by target timeframe
    """
    print(f"Building intra-bar features for {target_tf} from M1 data...")

    # Ensure time index
    if 'time' in m1_df.columns:
        df = m1_df.set_index('time').copy()
    else:
        df = m1_df.copy()

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    features = {}

    # 1. Standard OHLCV aggregation
    print("  Computing OHLCV aggregation...")
    ohlcv = df.resample(target_tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # 2. Hurst exponent (market regime)
    if compute_hurst and HURST_AVAILABLE:
        print("  Computing Hurst exponent...")
        features['hurst_exponent'] = df.resample(target_tf)['close'].apply(
            lambda x: compute_hurst_exponent(x)
        )
    else:
        print("  Skipping Hurst exponent (library not available or disabled)")
        features['hurst_exponent'] = pd.Series(0.5, index=ohlcv.index)

    # 3. Intra-bar volatility
    print("  Computing intra-bar volatility...")
    features['intra_volatility'] = compute_intra_bar_volatility(df, target_tf)

    # 4. Intra-bar momentum
    print("  Computing intra-bar momentum...")
    features['intra_momentum'] = compute_intra_bar_momentum(df, target_tf)

    # 5. Price efficiency
    print("  Computing price efficiency...")
    features['price_efficiency'] = compute_intra_bar_efficiency(df, target_tf)

    # 6. Volume profile
    print("  Computing volume profile...")
    vol_skew, vol_conc = compute_intra_bar_volume_profile(df, target_tf)
    features['volume_skew'] = vol_skew
    features['volume_concentration'] = vol_conc

    # 7. Derived features
    print("  Computing derived features...")

    # Hurst regime classification
    features['hurst_regime'] = features['hurst_exponent'].apply(
        lambda h: 1 if h > 0.55 else (-1 if h < 0.45 else 0)
    )

    # Volatility regime (z-score)
    vol_mean = features['intra_volatility'].rolling(50).mean()
    vol_std = features['intra_volatility'].rolling(50).std()
    features['volatility_zscore'] = (features['intra_volatility'] - vol_mean) / (vol_std + 1e-10)

    # Combine into DataFrame
    result = ohlcv.copy()
    for name, series in features.items():
        result[name] = series

    # Clean NaN
    result = result.dropna()

    print(f"  Built {len(features)} intra-bar features")
    print(f"  Output shape: {result.shape}")

    return result


def add_intra_bar_to_m1(m1_df: pd.DataFrame,
                         intra_bar_df: pd.DataFrame,
                         feature_cols: list = None) -> pd.DataFrame:
    """
    Merge intra-bar features back to M1 data using forward fill.

    This ensures no look-ahead bias - each M1 bar gets the intra-bar
    features from the PREVIOUS completed higher timeframe bar.
    """
    print("Merging intra-bar features to M1 data...")

    if feature_cols is None:
        feature_cols = [
            'hurst_exponent', 'hurst_regime',
            'intra_volatility', 'volatility_zscore',
            'intra_momentum', 'price_efficiency',
            'volume_skew', 'volume_concentration'
        ]

    # Ensure time index on both
    if 'time' in m1_df.columns:
        m1 = m1_df.set_index('time').copy()
    else:
        m1 = m1_df.copy()

    # Select only the features we want
    intra_features = intra_bar_df[feature_cols].copy()

    # Shift by 1 to avoid look-ahead bias (use PREVIOUS bar's features)
    intra_features_shifted = intra_features.shift(1)

    # Reindex to M1 frequency with forward fill
    intra_reindexed = intra_features_shifted.reindex(m1.index, method='ffill')

    # Merge
    result = m1.copy()
    for col in feature_cols:
        result[col] = intra_reindexed[col]

    # Reset index if original had time as column
    if 'time' in m1_df.columns:
        result = result.reset_index()

    print(f"  Added {len(feature_cols)} intra-bar features to M1 data")

    return result


# ============================================================================
# AUTONOMOUS LEARNING COMPONENTS (Future Enhancement)
# ============================================================================

class AdaptiveThresholds:
    """
    Adaptive threshold adjustment based on recent performance.

    Concept: If model performance degrades, adjust thresholds to be more
    conservative. If performance is strong, can be more aggressive.
    """

    def __init__(self,
                 initial_thresholds: dict,
                 adaptation_rate: float = 0.1,
                 lookback_trades: int = 100):
        self.thresholds = initial_thresholds.copy()
        self.adaptation_rate = adaptation_rate
        self.lookback = lookback_trades
        self.performance_history = []

    def record_trade(self, win: bool, confidence: float):
        """Record a trade result."""
        self.performance_history.append({
            'win': win,
            'confidence': confidence
        })

        # Keep only recent history
        if len(self.performance_history) > self.lookback:
            self.performance_history.pop(0)

    def update_thresholds(self) -> dict:
        """Update thresholds based on recent performance."""
        if len(self.performance_history) < 20:
            return self.thresholds

        recent = self.performance_history[-self.lookback:]
        win_rate = sum(t['win'] for t in recent) / len(recent)

        # If win rate dropping, increase thresholds (more conservative)
        # If win rate high, decrease thresholds (more aggressive)
        target_win_rate = 0.55
        adjustment = (win_rate - target_win_rate) * self.adaptation_rate

        for key in self.thresholds:
            self.thresholds[key] = max(0.3, min(0.7,
                self.thresholds[key] - adjustment
            ))

        return self.thresholds


class RegimeDetector:
    """
    Automatic market regime detection for model switching.

    Concept: Detect current market regime and use appropriate model
    or parameters for that regime.
    """

    REGIMES = ['trending_up', 'trending_down', 'ranging', 'volatile']

    def __init__(self, lookback: int = 240):
        self.lookback = lookback
        self.current_regime = 'ranging'

    def detect_regime(self,
                      close: pd.Series,
                      atr: pd.Series,
                      hurst: Optional[float] = None) -> str:
        """
        Detect current market regime.

        Returns one of: trending_up, trending_down, ranging, volatile
        """
        if len(close) < self.lookback:
            return 'ranging'

        recent_close = close.iloc[-self.lookback:]
        recent_atr = atr.iloc[-self.lookback:]

        # Trend detection
        start_price = recent_close.iloc[0]
        end_price = recent_close.iloc[-1]
        price_change_pct = (end_price - start_price) / start_price

        # Volatility detection
        current_atr = recent_atr.iloc[-1]
        avg_atr = recent_atr.mean()
        vol_ratio = current_atr / (avg_atr + 1e-10)

        # Use Hurst if available
        if hurst is not None:
            is_trending = hurst > 0.55
            is_ranging = hurst < 0.45
        else:
            # Fallback: use price efficiency
            returns = recent_close.pct_change().dropna()
            efficiency = abs(price_change_pct) / (returns.abs().sum() + 1e-10)
            is_trending = efficiency > 0.3
            is_ranging = efficiency < 0.15

        # Determine regime
        if vol_ratio > 1.5:
            self.current_regime = 'volatile'
        elif is_trending and price_change_pct > 0.02:
            self.current_regime = 'trending_up'
        elif is_trending and price_change_pct < -0.02:
            self.current_regime = 'trending_down'
        elif is_ranging:
            self.current_regime = 'ranging'
        else:
            # Keep previous regime
            pass

        return self.current_regime


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("INTRA-BAR FEATURE ENGINEERING")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"

    # Load M1 data
    m1_path = data_dir / "xauusd_M1.parquet"
    if m1_path.exists():
        print(f"\nLoading M1 data from {m1_path}...")
        m1_df = pd.read_parquet(m1_path)
        print(f"Loaded {len(m1_df):,} M1 bars")

        # Build intra-bar features for 4H timeframe
        intra_4h = build_intra_bar_features(m1_df, target_tf='4H', compute_hurst=True)

        # Save
        output_path = data_dir / "intra_bar_features_4H.parquet"
        intra_4h.to_parquet(output_path)
        print(f"\nSaved to {output_path}")

        # Show sample
        print("\nSample features:")
        print(intra_4h[['hurst_exponent', 'intra_volatility', 'price_efficiency']].tail(10))
    else:
        print(f"M1 data not found at {m1_path}")
        print("Run data preparation scripts first.")
