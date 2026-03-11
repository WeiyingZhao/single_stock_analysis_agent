"""Unit tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
class TestTechnicalIndicators:
    """Test suite for technical indicator calculations."""

    @pytest.fixture
    def sample_price_series(self):
        """Create a sample price series for testing."""
        # Create a simple uptrend
        prices = pd.Series([100 + i for i in range(50)])
        prices.index = pd.date_range(start="2024-01-01", periods=50, freq="D")
        return prices

    @pytest.fixture
    def sample_ohlcv_df(self):
        """Create a sample OHLCV DataFrame for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "Open": [100 + i * 0.5 for i in range(100)],
            "High": [105 + i * 0.5 for i in range(100)],
            "Low": [95 + i * 0.5 for i in range(100)],
            "Close": [102 + i * 0.5 for i in range(100)],
            "Volume": [1000000 + i * 10000 for i in range(100)],
        }, index=dates)
        return df

    def test_rsi_calculation(self, sample_price_series):
        """Test RSI calculation returns valid values."""
        from tools.technical_indicators import calculate_rsi

        rsi = calculate_rsi(sample_price_series, period=14)

        # RSI should be between 0 and 100
        assert rsi.notna().any(), "RSI should have non-NaN values"
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_macd_calculation(self, sample_price_series):
        """Test MACD calculation returns three components."""
        from tools.technical_indicators import calculate_macd

        macd_line, signal_line, histogram = calculate_macd(sample_price_series)

        # All three components should be Series
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        # Histogram should equal MACD - Signal
        valid_indices = macd_line.notna() & signal_line.notna()
        if valid_indices.any():
            np.testing.assert_array_almost_equal(
                histogram[valid_indices],
                (macd_line - signal_line)[valid_indices],
                decimal=5
            )

    def test_bollinger_bands_calculation(self, sample_price_series):
        """Test Bollinger Bands calculation."""
        from tools.technical_indicators import calculate_bollinger_bands

        upper, middle, lower = calculate_bollinger_bands(sample_price_series)

        # Upper should be above middle, middle above lower
        valid_indices = upper.notna() & middle.notna() & lower.notna()
        assert (upper[valid_indices] >= middle[valid_indices]).all()
        assert (middle[valid_indices] >= lower[valid_indices]).all()

    def test_moving_averages_calculation(self, sample_price_series):
        """Test moving averages calculation."""
        from tools.technical_indicators import calculate_moving_averages

        mas = calculate_moving_averages(sample_price_series, periods=[20, 50])

        assert "SMA_20" in mas
        assert "SMA_50" in mas
        assert isinstance(mas["SMA_20"], pd.Series)
        assert isinstance(mas["SMA_50"], pd.Series)

    def test_technical_analysis_integration(self, sample_ohlcv_df):
        """Test comprehensive technical analysis."""
        from tools.technical_indicators import get_technical_analysis

        analysis = get_technical_analysis(sample_ohlcv_df)

        # Should return a dictionary
        assert isinstance(analysis, dict)

        # Should have major components
        expected_keys = ["rsi", "macd", "bollinger_bands", "moving_averages"]
        for key in expected_keys:
            assert key in analysis, f"Missing {key} in technical analysis"

    def test_technical_analysis_with_insufficient_data(self):
        """Test technical analysis with insufficient data."""
        from tools.technical_indicators import get_technical_analysis

        # Create DataFrame with too few rows
        small_df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [102, 103, 104],
            "Low": [99, 100, 101],
            "Close": [101, 102, 103],
            "Volume": [1000000, 1000000, 1000000],
        })

        result = get_technical_analysis(small_df)

        # Should return error or handle gracefully
        assert "error" in result or "data_points" in result

    def test_interpret_technical_signals(self, sample_ohlcv_df):
        """Test technical signal interpretation."""
        from tools.technical_indicators import (
            get_technical_analysis,
            interpret_technical_signals,
        )

        analysis = get_technical_analysis(sample_ohlcv_df)
        interpretation = interpret_technical_signals(analysis)

        # Should have overall signal
        assert "overall" in interpretation
        assert interpretation["overall"] in ["Bullish", "Bearish", "Neutral"]

        # Should have signals list
        assert "signals" in interpretation
        assert isinstance(interpretation["signals"], list)
