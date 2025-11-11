"""Data models for historical event profiles."""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class EventProfile(BaseModel):
    """Model for a historical stock event profile."""

    event_id: str = Field(..., description="Unique identifier for the event")
    date: datetime = Field(..., description="Date of the event")
    symbol: str = Field(default="TSLA", description="Stock symbol")

    # Price data
    open_price: float = Field(..., description="Opening price")
    close_price: float = Field(..., description="Closing price")
    high_price: float = Field(..., description="Highest price")
    low_price: float = Field(..., description="Lowest price")
    volume: int = Field(..., description="Trading volume")
    price_change_pct: float = Field(..., description="Percentage price change")

    # Context data
    news_summary: str = Field(default="", description="Summary of relevant news")
    sentiment_score: float = Field(default=0.0, description="Overall sentiment score")
    sentiment_label: str = Field(default="Neutral", description="Sentiment label")

    # Additional context
    economic_context: Optional[str] = Field(None, description="Economic context")
    company_events: Optional[str] = Field(None, description="Company-specific events")

    # Embedding
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

    # Classification
    event_type: Optional[str] = Field(None, description="Type of event (e.g., 'Earnings', 'Product Launch')")
    significance: str = Field(default="High", description="Significance level")

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "TSLA_2024-01-15",
                "date": "2024-01-15T00:00:00",
                "symbol": "TSLA",
                "open_price": 200.0,
                "close_price": 210.5,
                "high_price": 212.0,
                "low_price": 199.5,
                "volume": 150000000,
                "price_change_pct": 5.25,
                "news_summary": "Tesla announces record deliveries",
                "sentiment_score": 0.75,
                "sentiment_label": "Positive",
                "event_type": "Product Delivery",
                "significance": "High"
            }
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "volume": self.volume,
            "price_change_pct": self.price_change_pct,
            "news_summary": self.news_summary,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "economic_context": self.economic_context,
            "company_events": self.company_events,
            "event_type": self.event_type,
            "significance": self.significance
        }

    def to_text_description(self) -> str:
        """
        Convert event profile to text description for embedding.

        Returns:
            Text description of the event
        """
        description = f"""
        Date: {self.date.strftime('%Y-%m-%d')}
        Stock: {self.symbol}
        Price Change: {self.price_change_pct:+.2f}%
        Price: ${self.close_price:.2f} (Open: ${self.open_price:.2f}, High: ${self.high_price:.2f}, Low: ${self.low_price:.2f})
        Volume: {self.volume:,}
        Sentiment: {self.sentiment_label} (Score: {self.sentiment_score:.2f})

        News Summary:
        {self.news_summary}

        Event Type: {self.event_type or 'Unknown'}
        """

        if self.economic_context:
            description += f"\nEconomic Context: {self.economic_context}"

        if self.company_events:
            description += f"\nCompany Events: {self.company_events}"

        return description.strip()


class CurrentDayProfile(BaseModel):
    """Model for current day's profile."""

    date: datetime = Field(..., description="Current date")
    symbol: str = Field(default="TSLA", description="Stock symbol")

    # Price data (if market has opened)
    open_price: Optional[float] = Field(None, description="Opening price")
    current_price: Optional[float] = Field(None, description="Current price")
    high_price: Optional[float] = Field(None, description="Highest price")
    low_price: Optional[float] = Field(None, description="Lowest price")
    volume: Optional[int] = Field(None, description="Trading volume")

    # Context data
    news_summary: str = Field(..., description="Summary of recent news")
    sentiment_score: float = Field(..., description="Overall sentiment score")
    sentiment_label: str = Field(..., description="Sentiment label")

    # Additional context
    market_context: Optional[str] = Field(None, description="Market context")
    recent_events: Optional[str] = Field(None, description="Recent company events")

    # Embedding
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

    def to_text_description(self) -> str:
        """
        Convert current day profile to text description for embedding.

        Returns:
            Text description of the current situation
        """
        description = f"""
        Date: {self.date.strftime('%Y-%m-%d')}
        Stock: {self.symbol}
        """

        if self.current_price:
            description += f"\nCurrent Price: ${self.current_price:.2f}"
            if self.open_price:
                pct_change = ((self.current_price - self.open_price) / self.open_price) * 100
                description += f" ({pct_change:+.2f}% from open)"

        if self.volume:
            description += f"\nVolume: {self.volume:,}"

        description += f"""
        Sentiment: {self.sentiment_label} (Score: {self.sentiment_score:.2f})

        News Summary:
        {self.news_summary}
        """

        if self.market_context:
            description += f"\nMarket Context: {self.market_context}"

        if self.recent_events:
            description += f"\nRecent Events: {self.recent_events}"

        return description.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-01-20T00:00:00",
                "symbol": "TSLA",
                "current_price": 205.0,
                "news_summary": "Tesla planning new factory expansion",
                "sentiment_score": 0.6,
                "sentiment_label": "Positive"
            }
        }
