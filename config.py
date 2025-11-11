"""Configuration settings for Tesla Stock Analysis System."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the stock analysis system."""

    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Stock Configuration
    STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "TSLA")
    HISTORICAL_YEARS = int(os.getenv("HISTORICAL_YEARS", "5"))

    # Analysis Thresholds
    SIGNIFICANT_CHANGE_THRESHOLD = float(os.getenv("SIGNIFICANT_CHANGE_THRESHOLD", "5.0"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    CONTEXT_WINDOW_HOURS = int(os.getenv("CONTEXT_WINDOW_HOURS", "72"))

    # Vector Database
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tesla_events")

    # Embedding Model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # LLM Configuration
    LLM_MODEL = "gemini-2.0-flash-exp"  # Updated to Gemini 2.0 (Gemini 1.x is deprecated)
    LLM_TEMPERATURE = 0.1

    # Data Sources
    NEWS_SOURCES = [
        "https://finance.yahoo.com/quote/TSLA/news",
        "https://www.cnbc.com/quotes/TSLA",
    ]

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in .env file. "
                "Get your key from: https://makersuite.google.com/app/apikey"
            )

        # Validate value ranges
        if cls.SIMILARITY_THRESHOLD < 0 or cls.SIMILARITY_THRESHOLD > 1:
            raise ValueError(
                f"SIMILARITY_THRESHOLD must be between 0 and 1, got {cls.SIMILARITY_THRESHOLD}"
            )

        if cls.HISTORICAL_YEARS < 1:
            raise ValueError(
                f"HISTORICAL_YEARS must be at least 1, got {cls.HISTORICAL_YEARS}"
            )

        if cls.SIGNIFICANT_CHANGE_THRESHOLD <= 0:
            raise ValueError(
                f"SIGNIFICANT_CHANGE_THRESHOLD must be positive, got {cls.SIGNIFICANT_CHANGE_THRESHOLD}"
            )

        if cls.CONTEXT_WINDOW_HOURS <= 0:
            raise ValueError(
                f"CONTEXT_WINDOW_HOURS must be positive, got {cls.CONTEXT_WINDOW_HOURS}"
            )

        return True

# Validate configuration on import
Config.validate()
