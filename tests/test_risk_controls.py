"""
Unit tests for risk_controls.py module.

Tests all risk management features:
1. Volatility regime position sizing
2. Execution quality monitoring
3. Extended adverse trend circuit breaker
4. Multi-level circuit breakers
"""
import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk_controls import (
    RiskControlManager,
    ExecutionQualityMonitor,
    VolatilityRegime,
    TrendRegime,
    TradeExecution,
    create_risk_manager,
)


class TestVolatilityPositionSizing:
    """Test volatility-based position sizing adjustments."""

    def test_high_vol_reduces_position(self):
        """HIGH_VOL regime should reduce position size by 50%."""
        manager = create_risk_manager(initial_capital=10000)

        # Set HIGH_VOL regime
        manager.update_regime(
            atr_current=3.0,  # High ATR
            atr_average=1.5,  # Average ATR
            adx=25,
            price=2000,
            ema_50=1990,
        )

        assert manager.state.current_vol_regime == VolatilityRegime.HIGH_VOL

        lot_size, adjustments = manager.calculate_position_size(
            entry_price=2000,
            stop_loss_price=1980,
        )

        assert adjustments["volatility_adjustment"] == 0.5
        assert "HIGH_VOL" in adjustments["reason"][0]

    def test_low_vol_increases_position(self):
        """LOW_VOL regime should increase position size by 25%."""
        manager = create_risk_manager(initial_capital=10000)

        # Set LOW_VOL regime
        manager.update_regime(
            atr_current=0.3,  # Low ATR
            atr_average=1.0,  # Average ATR
            adx=25,
            price=2000,
            ema_50=1990,
        )

        assert manager.state.current_vol_regime == VolatilityRegime.LOW_VOL

        lot_size, adjustments = manager.calculate_position_size(
            entry_price=2000,
            stop_loss_price=1980,
        )

        assert adjustments["volatility_adjustment"] == 1.25
        assert "LOW_VOL" in adjustments["reason"][0]

    def test_normal_vol_no_adjustment(self):
        """NORMAL_VOL regime should have no position adjustment."""
        manager = create_risk_manager(initial_capital=10000)

        # Set NORMAL_VOL regime
        manager.update_regime(
            atr_current=1.0,
            atr_average=1.0,
            adx=25,
            price=2000,
            ema_50=1990,
        )

        assert manager.state.current_vol_regime == VolatilityRegime.NORMAL_VOL

        lot_size, adjustments = manager.calculate_position_size(
            entry_price=2000,
            stop_loss_price=1980,
        )

        assert adjustments["volatility_adjustment"] == 1.0


class TestTrendRegimeDetection:
    """Test trend regime classification."""

    def test_trend_up_detection(self):
        """ADX > 25 and price > EMA should be TREND_UP."""
        manager = create_risk_manager()

        vol, trend = manager.update_regime(
            atr_current=1.0,
            atr_average=1.0,
            adx=30,
            price=2000,
            ema_50=1950,
        )

        assert trend == TrendRegime.TREND_UP

    def test_trend_down_detection(self):
        """ADX > 25 and price < EMA should be TREND_DOWN."""
        manager = create_risk_manager()

        vol, trend = manager.update_regime(
            atr_current=1.0,
            atr_average=1.0,
            adx=30,
            price=1900,
            ema_50=1950,
        )

        assert trend == TrendRegime.TREND_DOWN

    def test_ranging_detection(self):
        """ADX < 20 should be RANGING."""
        manager = create_risk_manager()

        vol, trend = manager.update_regime(
            atr_current=1.0,
            atr_average=1.0,
            adx=15,
            price=2000,
            ema_50=1990,
        )

        assert trend == TrendRegime.RANGING


class TestConsecutiveLossCircuitBreaker:
    """Test consecutive loss and extended trend circuit breakers."""

    def test_loss_streak_reduces_position(self):
        """10 consecutive losses should reduce position by 50%."""
        manager = create_risk_manager(initial_capital=10000)

        # Simulate 10 losing trades
        for i in range(10):
            manager.update_equity(10000 - (i + 1) * 50, trade_won=False)

        assert manager.state.consecutive_losses == 10

        lot_size, adjustments = manager.calculate_position_size(
            entry_price=2000,
            stop_loss_price=1980,
        )

        assert adjustments["trend_loss_adjustment"] == 0.5
        assert "Loss streak" in str(adjustments["reason"])

    def test_extended_loss_triggers_circuit_breaker(self):
        """20 consecutive losses should trigger circuit breaker."""
        manager = create_risk_manager(initial_capital=10000)

        # Simulate 20 losing trades
        for i in range(20):
            manager.update_equity(10000 - (i + 1) * 30, trade_won=False)

        assert manager.state.circuit_breaker_active
        assert "loss streak" in manager.state.circuit_breaker_reason.lower()

        # Position size should be 0
        lot_size, adjustments = manager.calculate_position_size(
            entry_price=2000,
            stop_loss_price=1980,
        )

        assert lot_size == 0
        assert adjustments["circuit_breaker_blocked"]

    def test_win_resets_loss_streak(self):
        """A winning trade should reset consecutive loss count."""
        manager = create_risk_manager(initial_capital=10000)

        # Simulate 5 losses
        for i in range(5):
            manager.update_equity(10000 - (i + 1) * 50, trade_won=False)

        assert manager.state.consecutive_losses == 5

        # Win resets
        manager.update_equity(9800, trade_won=True)

        assert manager.state.consecutive_losses == 0
        assert manager.state.consecutive_wins == 1


class TestDrawdownCircuitBreaker:
    """Test drawdown-based circuit breaker."""

    def test_25pct_drawdown_triggers_circuit_breaker(self):
        """25% drawdown should trigger circuit breaker."""
        manager = create_risk_manager(initial_capital=10000)

        # Simulate drawdown
        manager.update_equity(7400, trade_won=False)

        assert manager.state.current_drawdown >= 0.25
        assert manager.state.circuit_breaker_active
        assert "Drawdown" in manager.state.circuit_breaker_reason

    def test_trading_blocked_during_circuit_breaker(self):
        """Trading should be blocked when circuit breaker is active."""
        manager = create_risk_manager(initial_capital=10000)

        manager.update_equity(7400, trade_won=False)

        allowed, reason = manager.get_trading_allowed()

        assert not allowed
        assert "Drawdown" in reason


class TestExecutionQualityMonitoring:
    """Test execution quality tracking and alerts."""

    def test_slippage_calculation(self):
        """Slippage should be calculated correctly."""
        execution = TradeExecution(
            timestamp=datetime.now(),
            expected_price=2000.00,
            actual_price=2000.05,  # 5 pips adverse for long
            direction=1,
            lot_size=0.1,
        )

        assert execution.slippage_pips == pytest.approx(5.0, abs=0.01)

    def test_slippage_alert_triggered(self):
        """Alert should trigger when avg slippage exceeds threshold."""
        manager = create_risk_manager()

        # Record many high-slippage executions
        for i in range(15):
            manager.record_execution(
                expected_price=2000.00,
                actual_price=2000.05,  # 5 pips slippage
                direction=1,
                lot_size=0.1,
            )

        assert manager.state.slippage_alert_active

    def test_slippage_alert_reduces_position(self):
        """Active slippage alert should reduce position size."""
        manager = create_risk_manager(initial_capital=10000)

        # Trigger slippage alert
        for i in range(15):
            manager.record_execution(
                expected_price=2000.00,
                actual_price=2000.05,
                direction=1,
                lot_size=0.1,
            )

        lot_size, adjustments = manager.calculate_position_size(
            entry_price=2000,
            stop_loss_price=1980,
        )

        assert adjustments["execution_quality_adjustment"] == 0.7
        assert "Poor execution" in str(adjustments["reason"])


class TestExecutionQualityMonitor:
    """Test dedicated execution quality monitor."""

    def test_quality_report_generation(self):
        """Quality report should contain expected metrics."""
        monitor = ExecutionQualityMonitor()

        for i in range(10):
            execution = TradeExecution(
                timestamp=datetime.now(),
                expected_price=2000.00,
                actual_price=2000.00 + (i * 0.01),
                direction=1,
                lot_size=0.1,
            )
            monitor.record(execution, "NORMAL_VOL")

        report = monitor.get_quality_report()

        assert "total_executions" in report
        assert report["total_executions"] == 10
        assert "average_slippage_pips" in report
        assert "slippage_by_regime" in report


class TestMultipleAdjustmentsCombined:
    """Test combined risk adjustments."""

    def test_high_vol_plus_loss_streak(self):
        """HIGH_VOL + loss streak should compound reductions."""
        manager = create_risk_manager(initial_capital=10000)

        # Set HIGH_VOL
        manager.update_regime(
            atr_current=3.0,
            atr_average=1.5,
            adx=25,
            price=2000,
            ema_50=1990,
        )

        # Simulate 10 losses
        for i in range(10):
            manager.update_equity(10000 - (i + 1) * 30, trade_won=False)

        lot_size, adjustments = manager.calculate_position_size(
            entry_price=2000,
            stop_loss_price=1980,
        )

        # Combined: 0.5 * 0.5 = 0.25
        assert adjustments["final_multiplier"] == 0.25

    def test_min_multiplier_enforced(self):
        """Minimum multiplier should be enforced."""
        manager = create_risk_manager(initial_capital=10000)

        # Set HIGH_VOL
        manager.update_regime(
            atr_current=3.0,
            atr_average=1.5,
            adx=25,
            price=2000,
            ema_50=1990,
        )

        # Loss streak
        for i in range(10):
            manager.update_equity(10000 - (i + 1) * 30, trade_won=False)

        # Trigger slippage alert
        for i in range(15):
            manager.record_execution(2000, 2000.05, 1, 0.1)

        lot_size, adjustments = manager.calculate_position_size(
            entry_price=2000,
            stop_loss_price=1980,
        )

        # Should hit minimum (0.5 * 0.5 * 0.7 = 0.175, but min is 0.25)
        assert adjustments["final_multiplier"] >= 0.25


class TestFactoryFunction:
    """Test factory function."""

    def test_create_risk_manager(self):
        """Factory should create properly configured manager."""
        manager = create_risk_manager(
            initial_capital=50000,
            base_risk_pct=2.0,
            tp_pips=40,
            sl_pips=25,
        )

        assert manager.initial_capital == 50000
        assert manager.base_risk_per_trade == 0.02
        assert manager.tp_pips == 40
        assert manager.sl_pips == 25


class TestRiskSummary:
    """Test risk summary generation."""

    def test_summary_contains_all_fields(self):
        """Summary should contain all expected fields."""
        manager = create_risk_manager()

        summary = manager.get_risk_summary()

        expected_fields = [
            "trading_allowed",
            "current_equity",
            "peak_equity",
            "current_drawdown_pct",
            "consecutive_losses",
            "volatility_regime",
            "trend_regime",
            "slippage_alert",
        ]

        for field in expected_fields:
            assert field in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
