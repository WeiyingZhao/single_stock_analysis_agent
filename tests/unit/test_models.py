"""Unit tests for data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError


@pytest.mark.unit
class TestEventProfile:
    """Test suite for EventProfile model."""

    def test_event_profile_creation(self):
        """Test basic EventProfile creation."""
        from models.event_profile import EventProfile

        profile = EventProfile(
            event_id="TSLA_2024-01-15",
            date=datetime(2024, 1, 15),
            symbol="TSLA",
            open_price=200.0,
            close_price=210.0,
            high_price=212.0,
            low_price=199.0,
            volume=150000000,
            price_change_pct=5.0,
            news_summary="Tesla announces record deliveries",
            sentiment_score=0.75,
            sentiment_label="Positive"
        )

        assert profile.event_id == "TSLA_2024-01-15"
        assert profile.symbol == "TSLA"
        assert profile.price_change_pct == 5.0

    def test_event_profile_with_optional_fields(self):
        """Test EventProfile with optional technical and sector fields."""
        from models.event_profile import EventProfile

        profile = EventProfile(
            event_id="TSLA_2024-01-15",
            date=datetime(2024, 1, 15),
            symbol="TSLA",
            open_price=200.0,
            close_price=210.0,
            high_price=212.0,
            low_price=199.0,
            volume=150000000,
            price_change_pct=5.0,
            # Optional fields
            rsi=65.5,
            macd_signal="Bullish",
            technical_signal="Bullish",
            sector="Electric Vehicles",
            sector_momentum="Positive"
        )

        assert profile.rsi == 65.5
        assert profile.macd_signal == "Bullish"
        assert profile.sector == "Electric Vehicles"

    def test_event_profile_to_dict(self):
        """Test EventProfile.to_dict() method."""
        from models.event_profile import EventProfile

        profile = EventProfile(
            event_id="TSLA_2024-01-15",
            date=datetime(2024, 1, 15),
            symbol="TSLA",
            open_price=200.0,
            close_price=210.0,
            high_price=212.0,
            low_price=199.0,
            volume=150000000,
            price_change_pct=5.0
        )

        profile_dict = profile.to_dict()

        assert isinstance(profile_dict, dict)
        assert profile_dict["event_id"] == "TSLA_2024-01-15"
        assert profile_dict["symbol"] == "TSLA"

    def test_event_profile_to_text_description(self):
        """Test EventProfile.to_text_description() method."""
        from models.event_profile import EventProfile

        profile = EventProfile(
            event_id="TSLA_2024-01-15",
            date=datetime(2024, 1, 15),
            symbol="TSLA",
            open_price=200.0,
            close_price=210.0,
            high_price=212.0,
            low_price=199.0,
            volume=150000000,
            price_change_pct=5.0,
            news_summary="Tesla announces record deliveries"
        )

        description = profile.to_text_description()

        assert isinstance(description, str)
        assert "TSLA" in description
        assert "2024-01-15" in description
        assert "+5.00%" in description


@pytest.mark.unit
class TestCurrentDayProfile:
    """Test suite for CurrentDayProfile model."""

    def test_current_day_profile_creation(self):
        """Test basic CurrentDayProfile creation."""
        from models.event_profile import CurrentDayProfile

        profile = CurrentDayProfile(
            date=datetime(2024, 1, 20),
            symbol="TSLA",
            news_summary="Market analysis for today",
            sentiment_score=0.6,
            sentiment_label="Positive"
        )

        assert profile.symbol == "TSLA"
        assert profile.sentiment_label == "Positive"

    def test_current_day_profile_with_price_data(self):
        """Test CurrentDayProfile with optional price data."""
        from models.event_profile import CurrentDayProfile

        profile = CurrentDayProfile(
            date=datetime(2024, 1, 20),
            symbol="TSLA",
            current_price=205.0,
            open_price=200.0,
            high_price=206.0,
            low_price=199.0,
            volume=100000000,
            news_summary="Market analysis",
            sentiment_score=0.6,
            sentiment_label="Positive"
        )

        assert profile.current_price == 205.0
        assert profile.open_price == 200.0

    def test_current_day_profile_to_text_description(self):
        """Test CurrentDayProfile.to_text_description() method."""
        from models.event_profile import CurrentDayProfile

        profile = CurrentDayProfile(
            date=datetime(2024, 1, 20),
            symbol="TSLA",
            current_price=205.0,
            news_summary="Tesla planning expansion",
            sentiment_score=0.6,
            sentiment_label="Positive"
        )

        description = profile.to_text_description()

        assert isinstance(description, str)
        assert "TSLA" in description
        assert "Positive" in description
