"""Unit tests for configuration validation."""

import os

import pytest


@pytest.mark.unit
class TestConfig:
    """Test suite for Config class."""

    def test_config_imports_successfully(self):
        """Test that config module can be imported."""
        from config import Config

        assert Config is not None

    def test_config_has_required_attributes(self):
        """Test that Config has all required attributes."""
        from config import Config

        required_attrs = [
            "STOCK_SYMBOL",
            "HISTORICAL_YEARS",
            "SIGNIFICANT_CHANGE_THRESHOLD",
            "SIMILARITY_THRESHOLD",
            "CONTEXT_WINDOW_HOURS",
            "CHROMA_PERSIST_DIR",
            "EMBEDDING_MODEL",
        ]

        for attr in required_attrs:
            assert hasattr(Config, attr), f"Config missing required attribute: {attr}"

    def test_config_validation_with_valid_values(self, sample_config):
        """Test that config validation passes with valid values."""
        # This tests that the current configuration is valid
        from config import Config

        # Should not raise any exceptions
        assert Config.validate() is True

    def test_similarity_threshold_bounds(self):
        """Test that similarity threshold is between 0 and 1."""
        from config import Config

        assert 0 <= Config.SIMILARITY_THRESHOLD <= 1

    def test_historical_years_positive(self):
        """Test that historical years is positive."""
        from config import Config

        assert Config.HISTORICAL_YEARS > 0

    def test_significant_change_threshold_positive(self):
        """Test that significant change threshold is positive."""
        from config import Config

        assert Config.SIGNIFICANT_CHANGE_THRESHOLD > 0

    def test_context_window_hours_positive(self):
        """Test that context window hours is positive."""
        from config import Config

        assert Config.CONTEXT_WINDOW_HOURS > 0

    def test_enable_technical_indicators_is_bool(self):
        """Test that ENABLE_TECHNICAL_INDICATORS is boolean."""
        from config import Config

        assert isinstance(Config.ENABLE_TECHNICAL_INDICATORS, bool)

    def test_enable_related_stocks_is_bool(self):
        """Test that ENABLE_RELATED_STOCKS is boolean."""
        from config import Config

        assert isinstance(Config.ENABLE_RELATED_STOCKS, bool)
