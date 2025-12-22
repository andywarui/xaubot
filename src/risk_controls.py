"""
Advanced Risk Controls for XAUBOT Trading System.

Implements production-ready risk management based on Phase 2 backtesting findings:
1. Volatility-based position sizing (reduce in HIGH_VOL regimes)
2. Execution quality monitoring (detect slippage issues)
3. Extended adverse trend circuit breaker
4. Regime-aware risk adjustments

Author: XAUBOT Development Team
Version: 1.0.0
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW_VOL = "LOW_VOL"
    NORMAL_VOL = "NORMAL_VOL"
    HIGH_VOL = "HIGH_VOL"


class TrendRegime(Enum):
    """Trend regime classifications."""
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGING = "RANGING"
    MIXED = "MIXED"


@dataclass
class TradeExecution:
    """Records trade execution details for quality monitoring."""
    timestamp: datetime
    expected_price: float
    actual_price: float
    direction: int  # 1 for long, -1 for short
    lot_size: float
    slippage_pips: float = 0.0

    def __post_init__(self):
        # Calculate slippage: positive = adverse, negative = favorable
        pip_size = 0.01  # For XAU/USD
        price_diff = self.actual_price - self.expected_price
        self.slippage_pips = (price_diff * self.direction) / pip_size


@dataclass
class RiskState:
    """Current state of risk controls."""
    # Position sizing
    base_risk_per_trade: float = 0.01  # 1%
    current_risk_multiplier: float = 1.0

    # Circuit breaker state
    circuit_breaker_active: bool = False
    circuit_breaker_triggered_at: Optional[datetime] = None
    circuit_breaker_reason: str = ""

    # Trend tracking
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    trend_loss_streak: int = 0  # Extended trend losses

    # Drawdown tracking
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown: float = 0.0

    # Execution quality
    recent_slippage: List[float] = field(default_factory=list)
    slippage_alert_active: bool = False

    # Regime state
    current_vol_regime: VolatilityRegime = VolatilityRegime.NORMAL_VOL
    current_trend_regime: TrendRegime = TrendRegime.MIXED


class RiskControlManager:
    """
    Production risk control manager implementing backtesting recommendations.

    Key Features:
    - Volatility regime position sizing
    - Execution quality monitoring
    - Extended adverse trend protection
    - Multi-level circuit breakers
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        base_risk_per_trade: float = 0.01,
        tp_pips: float = 30.0,
        sl_pips: float = 20.0,
    ):
        self.initial_capital = initial_capital
        self.base_risk_per_trade = base_risk_per_trade
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.pip_value = 10.0  # USD per pip per standard lot

        # Initialize risk state
        self.state = RiskState(
            base_risk_per_trade=base_risk_per_trade,
            peak_equity=initial_capital,
            current_equity=initial_capital,
        )

        # Configuration thresholds
        self.config = {
            # Circuit breaker settings
            "circuit_breaker_dd": 0.25,  # 25% drawdown triggers pause
            "circuit_breaker_recovery": 0.15,  # Need 15% recovery to resume

            # Trend loss settings
            "trend_loss_threshold": 10,  # Consecutive losses before reduction
            "extended_trend_loss_threshold": 20,  # Extended streak circuit breaker
            "trend_loss_reduction": 0.5,  # 50% position reduction after threshold

            # Volatility position sizing
            "high_vol_reduction": 0.5,  # 50% reduction in HIGH_VOL
            "low_vol_boost": 1.25,  # 25% increase in LOW_VOL (capped)
            "max_risk_multiplier": 1.5,  # Maximum boost ever
            "min_risk_multiplier": 0.25,  # Minimum reduction ever

            # Execution quality
            "slippage_window": 50,  # Trades to track
            "slippage_alert_threshold": 3.0,  # Avg slippage pips for alert
            "slippage_reduction": 0.7,  # 30% reduction on poor execution

            # ATR thresholds for regime detection
            "atr_high_mult": 1.5,  # ATR > 1.5x avg = HIGH_VOL
            "atr_low_mult": 0.5,  # ATR < 0.5x avg = LOW_VOL
        }

        # Trade history for execution quality
        self.execution_history: List[TradeExecution] = []

    def update_regime(
        self,
        atr_current: float,
        atr_average: float,
        adx: float,
        price: float,
        ema_50: float,
    ) -> Tuple[VolatilityRegime, TrendRegime]:
        """
        Update current market regime based on indicators.

        Args:
            atr_current: Current ATR(14) value
            atr_average: Rolling average ATR (100 periods)
            adx: Current ADX value
            price: Current close price
            ema_50: 50-period EMA

        Returns:
            Tuple of (volatility_regime, trend_regime)
        """
        # Volatility regime
        atr_ratio = atr_current / atr_average if atr_average > 0 else 1.0

        if atr_ratio > self.config["atr_high_mult"]:
            vol_regime = VolatilityRegime.HIGH_VOL
        elif atr_ratio < self.config["atr_low_mult"]:
            vol_regime = VolatilityRegime.LOW_VOL
        else:
            vol_regime = VolatilityRegime.NORMAL_VOL

        # Trend regime
        if adx > 25:
            if price > ema_50:
                trend_regime = TrendRegime.TREND_UP
            else:
                trend_regime = TrendRegime.TREND_DOWN
        elif adx < 20:
            trend_regime = TrendRegime.RANGING
        else:
            trend_regime = TrendRegime.MIXED

        self.state.current_vol_regime = vol_regime
        self.state.current_trend_regime = trend_regime

        return vol_regime, trend_regime

    def record_execution(
        self,
        expected_price: float,
        actual_price: float,
        direction: int,
        lot_size: float,
    ) -> float:
        """
        Record trade execution for quality monitoring.

        Args:
            expected_price: Price at signal generation
            actual_price: Actual fill price
            direction: 1 for long, -1 for short
            lot_size: Trade size

        Returns:
            Slippage in pips (positive = adverse)
        """
        execution = TradeExecution(
            timestamp=datetime.now(),
            expected_price=expected_price,
            actual_price=actual_price,
            direction=direction,
            lot_size=lot_size,
        )

        self.execution_history.append(execution)

        # Keep only recent executions
        if len(self.execution_history) > self.config["slippage_window"]:
            self.execution_history.pop(0)

        # Update slippage tracking
        self.state.recent_slippage.append(execution.slippage_pips)
        if len(self.state.recent_slippage) > self.config["slippage_window"]:
            self.state.recent_slippage.pop(0)

        # Check for slippage alert
        if len(self.state.recent_slippage) >= 10:
            avg_slippage = np.mean(self.state.recent_slippage)
            self.state.slippage_alert_active = (
                avg_slippage > self.config["slippage_alert_threshold"]
            )

        return execution.slippage_pips

    def update_equity(self, new_equity: float, trade_won: bool) -> None:
        """
        Update equity and loss streak tracking after a trade.

        Args:
            new_equity: Current account equity
            trade_won: Whether the trade was profitable
        """
        self.state.current_equity = new_equity

        # Update peak
        if new_equity > self.state.peak_equity:
            self.state.peak_equity = new_equity

        # Calculate drawdown
        if self.state.peak_equity > 0:
            self.state.current_drawdown = (
                (self.state.peak_equity - new_equity) / self.state.peak_equity
            )

        # Update consecutive win/loss tracking
        if trade_won:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
            self.state.trend_loss_streak = max(0, self.state.trend_loss_streak - 1)
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
            self.state.trend_loss_streak += 1

        # Check circuit breakers
        self._check_circuit_breakers()

    def _check_circuit_breakers(self) -> None:
        """Check and update circuit breaker state."""
        # Check drawdown circuit breaker
        if self.state.current_drawdown >= self.config["circuit_breaker_dd"]:
            if not self.state.circuit_breaker_active:
                self.state.circuit_breaker_active = True
                self.state.circuit_breaker_triggered_at = datetime.now()
                self.state.circuit_breaker_reason = (
                    f"Drawdown {self.state.current_drawdown*100:.1f}% exceeded "
                    f"{self.config['circuit_breaker_dd']*100:.0f}% threshold"
                )

        # Check extended trend loss circuit breaker
        if self.state.trend_loss_streak >= self.config["extended_trend_loss_threshold"]:
            if not self.state.circuit_breaker_active:
                self.state.circuit_breaker_active = True
                self.state.circuit_breaker_triggered_at = datetime.now()
                self.state.circuit_breaker_reason = (
                    f"Extended loss streak: {self.state.trend_loss_streak} consecutive losses"
                )

        # Check for recovery (only if circuit breaker was DD-triggered)
        if self.state.circuit_breaker_active:
            recovery = (
                (self.state.current_equity - (self.state.peak_equity * (1 - self.config["circuit_breaker_dd"])))
                / (self.state.peak_equity * self.config["circuit_breaker_dd"])
            )
            if recovery >= self.config["circuit_breaker_recovery"]:
                # Also require loss streak to be broken
                if self.state.trend_loss_streak < self.config["trend_loss_threshold"]:
                    self.state.circuit_breaker_active = False
                    self.state.circuit_breaker_reason = ""

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
    ) -> Tuple[float, Dict]:
        """
        Calculate risk-adjusted position size.

        Applies all risk adjustments:
        1. Volatility regime adjustment
        2. Trend loss streak reduction
        3. Execution quality adjustment
        4. Circuit breaker check

        Args:
            entry_price: Planned entry price
            stop_loss_price: Stop loss price

        Returns:
            Tuple of (lot_size, adjustment_details)
        """
        adjustments = {
            "base_risk": self.base_risk_per_trade,
            "volatility_adjustment": 1.0,
            "trend_loss_adjustment": 1.0,
            "execution_quality_adjustment": 1.0,
            "circuit_breaker_blocked": False,
            "final_multiplier": 1.0,
            "reason": [],
        }

        # Check circuit breaker first
        if self.state.circuit_breaker_active:
            adjustments["circuit_breaker_blocked"] = True
            adjustments["reason"].append(
                f"Circuit breaker active: {self.state.circuit_breaker_reason}"
            )
            return 0.0, adjustments

        # 1. Volatility regime adjustment
        if self.state.current_vol_regime == VolatilityRegime.HIGH_VOL:
            adjustments["volatility_adjustment"] = self.config["high_vol_reduction"]
            adjustments["reason"].append(
                f"HIGH_VOL regime: {self.config['high_vol_reduction']*100:.0f}% position"
            )
        elif self.state.current_vol_regime == VolatilityRegime.LOW_VOL:
            adjustments["volatility_adjustment"] = self.config["low_vol_boost"]
            adjustments["reason"].append(
                f"LOW_VOL regime: {self.config['low_vol_boost']*100:.0f}% position"
            )

        # 2. Trend loss adjustment
        if self.state.consecutive_losses >= self.config["trend_loss_threshold"]:
            adjustments["trend_loss_adjustment"] = self.config["trend_loss_reduction"]
            adjustments["reason"].append(
                f"Loss streak ({self.state.consecutive_losses}): "
                f"{self.config['trend_loss_reduction']*100:.0f}% position"
            )

        # 3. Execution quality adjustment
        if self.state.slippage_alert_active:
            adjustments["execution_quality_adjustment"] = self.config["slippage_reduction"]
            avg_slippage = np.mean(self.state.recent_slippage) if self.state.recent_slippage else 0
            adjustments["reason"].append(
                f"Poor execution (avg slip {avg_slippage:.1f} pips): "
                f"{self.config['slippage_reduction']*100:.0f}% position"
            )

        # Calculate final multiplier
        final_mult = (
            adjustments["volatility_adjustment"]
            * adjustments["trend_loss_adjustment"]
            * adjustments["execution_quality_adjustment"]
        )

        # Apply caps
        final_mult = max(
            self.config["min_risk_multiplier"],
            min(self.config["max_risk_multiplier"], final_mult)
        )
        adjustments["final_multiplier"] = final_mult

        # Calculate position size
        effective_risk = self.base_risk_per_trade * final_mult
        risk_amount = self.state.current_equity * effective_risk

        # Calculate SL distance in pips
        sl_distance_pips = abs(entry_price - stop_loss_price) / 0.01  # XAU/USD pip = 0.01

        if sl_distance_pips > 0:
            lot_size = risk_amount / (sl_distance_pips * self.pip_value)
        else:
            lot_size = 0.0

        # Cap lot size to reasonable maximum
        lot_size = min(lot_size, 100.0)  # Max 100 lots

        adjustments["effective_risk"] = effective_risk
        adjustments["lot_size"] = lot_size

        return lot_size, adjustments

    def get_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is currently allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        if self.state.circuit_breaker_active:
            return False, self.state.circuit_breaker_reason

        return True, "Trading allowed"

    def get_risk_summary(self) -> Dict:
        """Get current risk state summary."""
        return {
            "trading_allowed": not self.state.circuit_breaker_active,
            "circuit_breaker_reason": self.state.circuit_breaker_reason,
            "current_equity": self.state.current_equity,
            "peak_equity": self.state.peak_equity,
            "current_drawdown_pct": self.state.current_drawdown * 100,
            "consecutive_losses": self.state.consecutive_losses,
            "consecutive_wins": self.state.consecutive_wins,
            "trend_loss_streak": self.state.trend_loss_streak,
            "volatility_regime": self.state.current_vol_regime.value,
            "trend_regime": self.state.current_trend_regime.value,
            "slippage_alert": self.state.slippage_alert_active,
            "avg_slippage": np.mean(self.state.recent_slippage) if self.state.recent_slippage else 0,
            "risk_multiplier": self.state.current_risk_multiplier,
        }

    def reset_state(self, new_capital: Optional[float] = None) -> None:
        """Reset risk state, optionally with new capital."""
        capital = new_capital if new_capital else self.initial_capital
        self.state = RiskState(
            base_risk_per_trade=self.base_risk_per_trade,
            peak_equity=capital,
            current_equity=capital,
        )
        self.execution_history.clear()


class ExecutionQualityMonitor:
    """
    Dedicated execution quality monitoring.

    Tracks slippage patterns and alerts on degradation.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.executions: List[TradeExecution] = []
        self.slippage_by_hour: Dict[int, List[float]] = {h: [] for h in range(24)}
        self.slippage_by_volatility: Dict[str, List[float]] = {
            "LOW_VOL": [],
            "NORMAL_VOL": [],
            "HIGH_VOL": [],
        }

    def record(
        self,
        execution: TradeExecution,
        volatility_regime: str = "NORMAL_VOL",
    ) -> None:
        """Record an execution with context."""
        self.executions.append(execution)
        if len(self.executions) > self.window_size:
            self.executions.pop(0)

        # Track by hour
        hour = execution.timestamp.hour
        self.slippage_by_hour[hour].append(execution.slippage_pips)
        if len(self.slippage_by_hour[hour]) > 50:
            self.slippage_by_hour[hour].pop(0)

        # Track by volatility
        if volatility_regime in self.slippage_by_volatility:
            self.slippage_by_volatility[volatility_regime].append(execution.slippage_pips)
            if len(self.slippage_by_volatility[volatility_regime]) > 50:
                self.slippage_by_volatility[volatility_regime].pop(0)

    def get_quality_report(self) -> Dict:
        """Generate execution quality report."""
        if not self.executions:
            return {"status": "No data"}

        all_slippage = [e.slippage_pips for e in self.executions]

        report = {
            "total_executions": len(self.executions),
            "average_slippage_pips": np.mean(all_slippage),
            "max_slippage_pips": np.max(all_slippage),
            "std_slippage_pips": np.std(all_slippage),
            "adverse_rate": np.mean([s > 0 for s in all_slippage]),
            "worst_hours": [],
            "best_hours": [],
            "slippage_by_regime": {},
        }

        # Find worst/best hours
        hour_avgs = {}
        for hour, slips in self.slippage_by_hour.items():
            if slips:
                hour_avgs[hour] = np.mean(slips)

        if hour_avgs:
            sorted_hours = sorted(hour_avgs.items(), key=lambda x: x[1])
            report["best_hours"] = [h for h, _ in sorted_hours[:3]]
            report["worst_hours"] = [h for h, _ in sorted_hours[-3:]]

        # Regime analysis
        for regime, slips in self.slippage_by_volatility.items():
            if slips:
                report["slippage_by_regime"][regime] = {
                    "avg": np.mean(slips),
                    "max": np.max(slips),
                    "count": len(slips),
                }

        return report


# Factory function for easy instantiation
def create_risk_manager(
    initial_capital: float = 10000.0,
    base_risk_pct: float = 1.0,  # 1% default
    tp_pips: float = 30.0,
    sl_pips: float = 20.0,
) -> RiskControlManager:
    """
    Create a configured risk control manager.

    Args:
        initial_capital: Starting account balance
        base_risk_pct: Base risk per trade as percentage (e.g., 1.0 = 1%)
        tp_pips: Take profit in pips
        sl_pips: Stop loss in pips

    Returns:
        Configured RiskControlManager instance
    """
    return RiskControlManager(
        initial_capital=initial_capital,
        base_risk_per_trade=base_risk_pct / 100.0,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
    )
