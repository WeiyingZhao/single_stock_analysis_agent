"""Pytest configuration and shared fixtures for all tests."""

import os
from pathlib import Path
from typing import Generator

import pytest


# ============================================================================
# Test Configuration
# ============================================================================

# Set test environment variables
os.environ["TESTING"] = "1"
os.environ["GOOGLE_API_KEY"] = "test-api-key-for-testing"
os.environ["STOCK_SYMBOL"] = "TSLA"
os.environ["CHROMA_PERSIST_DIR"] = "./test_chroma_db"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Set up test environment before all tests."""
    # Setup
    print("\n🧪 Setting up test environment...")

    # Ensure test database directory doesn't exist
    test_db_dir = Path("./test_chroma_db")
    if test_db_dir.exists():
        import shutil
        shutil.rmtree(test_db_dir)

    yield

    # Cleanup
    print("\n🧹 Cleaning up test environment...")
    if test_db_dir.exists():
        import shutil
        shutil.rmtree(test_db_dir)


# ============================================================================
# Common Fixtures
# ============================================================================

@pytest.fixture
def mock_stock_data():
    """Provide mock stock data for testing."""
    import pandas as pd
    from datetime import datetime, timedelta

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        periods=100,
        freq="D"
    )

    return pd.DataFrame({
        "Open": [100 + i * 0.5 for i in range(100)],
        "High": [105 + i * 0.5 for i in range(100)],
        "Low": [95 + i * 0.5 for i in range(100)],
        "Close": [102 + i * 0.5 for i in range(100)],
        "Volume": [1000000 + i * 10000 for i in range(100)],
    }, index=dates)


@pytest.fixture
def mock_news_articles():
    """Provide mock news articles for testing."""
    return [
        {
            "title": "Tesla announces record deliveries",
            "summary": "Tesla delivered record number of vehicles in Q4",
            "url": "https://example.com/news1",
            "date": "2024-01-15"
        },
        {
            "title": "Stock market rally continues",
            "summary": "Major indices hit new highs",
            "url": "https://example.com/news2",
            "date": "2024-01-15"
        }
    ]


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "STOCK_SYMBOL": "TSLA",
        "HISTORICAL_YEARS": 5,
        "SIGNIFICANT_CHANGE_THRESHOLD": 5.0,
        "SIMILARITY_THRESHOLD": 0.85,
        "CONTEXT_WINDOW_HOURS": 72,
        "ENABLE_TECHNICAL_INDICATORS": True,
        "ENABLE_RELATED_STOCKS": True,
    }


# ============================================================================
# Markers for Test Organization
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers",
        "llm: marks tests that call actual LLM APIs (expensive)"
    )
