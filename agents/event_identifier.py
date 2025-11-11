"""Historical Event Identification Agent for detecting significant price movements."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from datetime import datetime, timedelta
from typing import List
import pandas as pd

from config import Config
from models.event_profile import EventProfile
from tools.stock_data import get_historical_data_df
from tools.news_scraper import scrape_yahoo_news
from tools.sentiment import get_market_sentiment_summary
from utils.analysis import (
    identify_significant_changes,
    get_subsequent_performance,
    categorize_event_type
)
from utils.embeddings import generate_text_embedding
from tools.vector_store import VectorStoreManager


def create_event_identifier_agent():
    """
    Create the Historical Event Identification Agent.

    This agent analyzes historical data to identify and characterize
    significant price movement events.

    Returns:
        LangGraph agent for event identification
    """

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        google_api_key=Config.GOOGLE_API_KEY
    )

    # System prompt
    system_prompt = f"""You are a Historical Event Identification Agent specialized in analyzing stock price movements.

Your responsibilities:
1. Identify days with significant price changes (threshold: {Config.SIGNIFICANT_CHANGE_THRESHOLD}%)
2. Analyze the context surrounding each significant event
3. Summarize the likely causal factors for each price movement
4. Create structured event profiles for storage

When analyzing events:
- Look for price changes exceeding +/- {Config.SIGNIFICANT_CHANGE_THRESHOLD}%
- Consider the {Config.CONTEXT_WINDOW_HOURS}-hour window around each event
- Examine news, sentiment, and market conditions
- Classify the type of event (Earnings, Product Launch, etc.)
- Provide clear, concise summaries of why the price moved

Your summaries should be:
- Factual and based on available data
- Focused on the most likely causal factors
- 2-3 sentences long
- Actionable for future predictions

Example summary: "Tesla stock surged 6.2% following announcement of record Q3 deliveries
exceeding analyst expectations. Positive sentiment dominated as production targets were
met ahead of schedule, boosting investor confidence."
"""

    # Create agent without tools (it will use functions directly)
    agent = create_agent(
        llm,
        [],  # No tools needed, uses direct functions
        system_prompt=system_prompt
    )

    return agent


def identify_and_store_historical_events(
    symbol: str = None,
    years: int = None
) -> dict:
    """
    Identify significant historical events and store them in the vector database.

    Args:
        symbol: Stock symbol (default from config)
        years: Number of years to analyze (default from config)

    Returns:
        Dictionary with processing results
    """
    symbol = symbol or Config.STOCK_SYMBOL
    years = years or Config.HISTORICAL_YEARS

    print(f"\n{'='*80}")
    print(f"IDENTIFYING HISTORICAL EVENTS FOR {symbol}")
    print(f"{'='*80}\n")

    # Step 1: Fetch historical stock data
    print("Step 1: Fetching historical stock data...")
    df = get_historical_data_df(symbol, years)
    print(f"  ✓ Loaded {len(df)} days of data")

    # Step 2: Identify significant changes
    print(f"\nStep 2: Identifying significant price changes (threshold: {Config.SIGNIFICANT_CHANGE_THRESHOLD}%)...")
    significant_df = identify_significant_changes(df, Config.SIGNIFICANT_CHANGE_THRESHOLD)
    print(f"  ✓ Found {len(significant_df)} significant events")

    if len(significant_df) == 0:
        return {
            "success": True,
            "events_found": 0,
            "events_stored": 0,
            "message": "No significant events found"
        }

    # Step 3: Process each significant event
    print(f"\nStep 3: Processing events and creating profiles...")

    agent = create_event_identifier_agent()
    vector_store = VectorStoreManager()

    event_profiles = []
    stored_count = 0

    for idx, (date, row) in enumerate(significant_df.iterrows(), 1):
        try:
            print(f"\n  Processing event {idx}/{len(significant_df)}: {date.strftime('%Y-%m-%d')} ({row['Pct_Change']:+.2f}%)")

            # Create event profile
            event_profile = create_event_profile(
                date=date,
                row=row,
                df=df,
                symbol=symbol,
                agent=agent
            )

            if event_profile:
                event_profiles.append(event_profile)

                # Store in vector database
                success = vector_store.add_event_profile(
                    event_id=event_profile.event_id,
                    embedding=event_profile.embedding,
                    metadata=event_profile.to_dict(),
                    document=event_profile.to_text_description()
                )

                if success:
                    stored_count += 1
                    print(f"    ✓ Stored event profile")
                else:
                    print(f"    ✗ Failed to store event profile")

        except Exception as e:
            print(f"    ✗ Error processing event: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Events found: {len(significant_df)}")
    print(f"Events processed: {len(event_profiles)}")
    print(f"Events stored: {stored_count}")
    print(f"{'='*80}\n")

    return {
        "success": True,
        "events_found": len(significant_df),
        "events_processed": len(event_profiles),
        "events_stored": stored_count,
        "event_profiles": event_profiles
    }


def create_event_profile(
    date: datetime,
    row: pd.Series,
    df: pd.DataFrame,
    symbol: str,
    agent
) -> EventProfile:
    """
    Create an event profile for a significant price movement.

    Args:
        date: Date of the event
        row: DataFrame row with price data
        df: Full historical DataFrame
        symbol: Stock symbol
        agent: LLM agent for summarization

    Returns:
        EventProfile object
    """

    # Generate event ID
    event_id = f"{symbol}_{date.strftime('%Y-%m-%d')}"

    # Get subsequent performance
    performance = get_subsequent_performance(df, date, days=[1, 3])

    # Get news (best effort - will be limited for historical dates)
    # In production, you would use a historical news API
    news_summary = f"Significant price movement: {row['Pct_Change']:+.2f}%"

    # Analyze sentiment from news summary
    sentiment_label = "Positive" if row['Pct_Change'] > 0 else "Negative"
    sentiment_score = min(abs(row['Pct_Change']) / 10, 1.0)  # Rough estimate
    if row['Pct_Change'] < 0:
        sentiment_score = -sentiment_score

    # Use LLM to generate event summary
    try:
        summary_prompt = f"""Analyze this Tesla stock event and provide a 2-3 sentence summary of likely causes:

Date: {date.strftime('%Y-%m-%d')}
Price Change: {row['Pct_Change']:+.2f}%
Open: ${row['Open']:.2f}
Close: ${row['Close']:.2f}
Volume: {row['Volume']:,}
Direction: {"Increase" if row['Pct_Change'] > 0 else "Decrease"}

Provide a factual summary of what likely caused this price movement. Focus on the most probable reasons based on typical market drivers for Tesla stock (e.g., deliveries, earnings, product news, macro factors, Elon Musk related news, regulatory issues).
"""

        result = agent.invoke({"messages": [("user", summary_prompt)]})
        generated_summary = result['messages'][-1].content if result['messages'] else news_summary

    except Exception as e:
        print(f"      Warning: Could not generate summary with LLM: {e}")
        generated_summary = news_summary

    # Categorize event type
    event_type = categorize_event_type(row['Pct_Change'], generated_summary)

    # Create the event profile
    event_profile = EventProfile(
        event_id=event_id,
        date=date,
        symbol=symbol,
        open_price=float(row['Open']),
        close_price=float(row['Close']),
        high_price=float(row['High']),
        low_price=float(row['Low']),
        volume=int(row['Volume']),
        price_change_pct=float(row['Pct_Change']),
        news_summary=generated_summary,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        event_type=event_type,
        significance="High"
    )

    # Generate embedding
    text_description = event_profile.to_text_description()
    embedding = generate_text_embedding(text_description)
    event_profile.embedding = embedding

    return event_profile


# Example usage
if __name__ == "__main__":
    print("Testing Historical Event Identifier Agent...")

    # Test identifying events (use smaller dataset for testing)
    result = identify_and_store_historical_events(years=1)
    print(f"\nResult: {result}")
