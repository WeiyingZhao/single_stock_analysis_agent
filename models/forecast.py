"""Data models for stock forecasts."""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    """Predicted stock direction."""
    UP = "Up"
    DOWN = "Down"
    NEUTRAL = "Neutral"


class MatchedEvent(BaseModel):
    """Model for a matched historical event."""

    event_id: str = Field(..., description="Historical event ID")
    date: datetime = Field(..., description="Date of historical event")
    price_change_pct: float = Field(..., description="Price change percentage")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    news_summary: str = Field(..., description="Summary of the historical event")
    event_type: Optional[str] = Field(None, description="Type of event")

    # Subsequent performance
    next_day_change: Optional[float] = Field(None, description="Next day price change %")
    three_day_change: Optional[float] = Field(None, description="Three day price change %")

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "TSLA_2023-10-15",
                "date": "2023-10-15T00:00:00",
                "price_change_pct": 5.5,
                "similarity_score": 0.92,
                "news_summary": "Tesla announces record deliveries",
                "event_type": "Product Delivery",
                "next_day_change": 2.1,
                "three_day_change": 3.8
            }
        }


class Forecast(BaseModel):
    """Model for stock forecast."""

    symbol: str = Field(default="TSLA", description="Stock symbol")
    forecast_date: datetime = Field(..., description="Date of forecast")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp of forecast generation")

    # Prediction
    predicted_direction: Direction = Field(..., description="Predicted stock direction")
    confidence_score: float = Field(..., description="Confidence score (0-1)")

    # Matched events
    matched_events: List[MatchedEvent] = Field(default_factory=list, description="List of matched historical events")
    best_match_id: Optional[str] = Field(None, description="ID of the best matching event")

    # Explanation
    reasoning: str = Field(..., description="Natural language explanation of the prediction")
    key_factors: List[str] = Field(default_factory=list, description="Key factors influencing the prediction")

    # Current context
    current_sentiment: str = Field(..., description="Current market sentiment")
    current_news_summary: str = Field(..., description="Summary of current news")

    # Risk assessment
    risk_level: str = Field(default="Medium", description="Risk level of the prediction")
    caveats: List[str] = Field(default_factory=list, description="Important caveats or warnings")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "TSLA",
                "forecast_date": "2024-01-20T00:00:00",
                "predicted_direction": "Up",
                "confidence_score": 0.85,
                "reasoning": "Current market conditions closely match historical event from Oct 2023...",
                "current_sentiment": "Positive",
                "risk_level": "Medium"
            }
        }

    def to_report(self) -> str:
        """
        Generate a human-readable forecast report.

        Returns:
            Formatted report string
        """
        report = f"""
{'='*80}
TESLA STOCK FORECAST REPORT
{'='*80}

Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Forecast Date: {self.forecast_date.strftime('%Y-%m-%d')}
Symbol: {self.symbol}

{'='*80}
PREDICTION
{'='*80}

Direction: {self.predicted_direction.value}
Confidence: {self.confidence_score:.1%}
Risk Level: {self.risk_level}

{'='*80}
REASONING
{'='*80}

{self.reasoning}

{'='*80}
KEY FACTORS
{'='*80}
"""
        for i, factor in enumerate(self.key_factors, 1):
            report += f"\n{i}. {factor}"

        if self.matched_events:
            report += f"\n\n{'='*80}\n"
            report += "SIMILAR HISTORICAL EVENTS\n"
            report += f"{'='*80}\n"

            for i, event in enumerate(self.matched_events, 1):
                report += f"\n{i}. {event.date.strftime('%Y-%m-%d')} (Similarity: {event.similarity_score:.1%})\n"
                report += f"   Price Change: {event.price_change_pct:+.2f}%\n"
                if event.next_day_change:
                    report += f"   Next Day: {event.next_day_change:+.2f}%\n"
                if event.three_day_change:
                    report += f"   3-Day Change: {event.three_day_change:+.2f}%\n"
                report += f"   Event: {event.news_summary[:100]}...\n"

        report += f"\n{'='*80}\n"
        report += "CURRENT MARKET CONTEXT\n"
        report += f"{'='*80}\n\n"
        report += f"Sentiment: {self.current_sentiment}\n"
        report += f"News: {self.current_news_summary}\n"

        if self.caveats:
            report += f"\n{'='*80}\n"
            report += "IMPORTANT CAVEATS\n"
            report += f"{'='*80}\n"
            for i, caveat in enumerate(self.caveats, 1):
                report += f"\n{i}. {caveat}"

        report += f"\n\n{'='*80}\n"
        report += "DISCLAIMER: This forecast is for informational purposes only.\n"
        report += "Past performance does not guarantee future results.\n"
        report += f"{'='*80}\n"

        return report

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "forecast_date": self.forecast_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "predicted_direction": self.predicted_direction.value,
            "confidence_score": self.confidence_score,
            "matched_events": [
                {
                    "event_id": e.event_id,
                    "date": e.date.isoformat(),
                    "similarity_score": e.similarity_score,
                    "price_change_pct": e.price_change_pct
                }
                for e in self.matched_events
            ],
            "reasoning": self.reasoning,
            "current_sentiment": self.current_sentiment,
            "risk_level": self.risk_level
        }


class NoMatchResult(BaseModel):
    """Model for when no significant historical match is found."""

    symbol: str = Field(default="TSLA", description="Stock symbol")
    forecast_date: datetime = Field(..., description="Date of attempted forecast")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp")

    message: str = Field(
        default="No significant historical parallel found.",
        description="Result message"
    )
    highest_similarity: float = Field(default=0.0, description="Highest similarity score found")
    threshold: float = Field(..., description="Required similarity threshold")

    current_sentiment: str = Field(..., description="Current market sentiment")
    current_news_summary: str = Field(..., description="Summary of current news")

    def to_report(self) -> str:
        """Generate a human-readable no-match report."""
        return f"""
{'='*80}
TESLA STOCK FORECAST REPORT
{'='*80}

Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Forecast Date: {self.forecast_date.strftime('%Y-%m-%d')}
Symbol: {self.symbol}

{'='*80}
RESULT
{'='*80}

{self.message}

The current market conditions do not sufficiently match any historical
significant price movement events in our database.

Highest Similarity Found: {self.highest_similarity:.1%}
Required Threshold: {self.threshold:.1%}

{'='*80}
CURRENT MARKET CONTEXT
{'='*80}

Sentiment: {self.current_sentiment}
News: {self.current_news_summary}

{'='*80}
RECOMMENDATION
{'='*80}

Without a strong historical parallel, we cannot make a confident prediction.
Continue monitoring current market conditions and news developments.

{'='*80}
"""
