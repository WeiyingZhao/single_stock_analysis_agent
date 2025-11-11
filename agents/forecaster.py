"""Similarity and Forecasting Agent for stock predictions."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from datetime import datetime
from typing import Optional, Union

from config import Config
from models.event_profile import CurrentDayProfile
from models.forecast import Forecast, MatchedEvent, Direction, NoMatchResult
from tools.vector_store import VectorStoreManager
from tools.stock_data import get_historical_data_df
from utils.analysis import get_subsequent_performance


def create_forecasting_agent():
    """
    Create the Similarity and Forecasting Agent.

    This agent compares current conditions to historical events and
    generates predictions based on similarity.

    Returns:
        LangGraph agent for forecasting
    """

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        google_api_key=Config.GOOGLE_API_KEY
    )

    # System prompt
    system_prompt = f"""You are a Similarity and Forecasting Agent specialized in stock prediction through historical pattern matching.

Your responsibilities:
1. Compare current market conditions to historical events using similarity matching
2. Identify the most relevant historical parallels
3. Analyze how the stock performed after similar historical events
4. Generate predictions with confidence scores and explanations

When making predictions:
- Similarity threshold: {Config.SIMILARITY_THRESHOLD} (0-1 scale)
- Only make predictions when similarity is above threshold
- Provide clear reasoning for predictions
- Include confidence scores based on match quality and historical outcomes
- Identify key factors influencing the prediction

Your predictions should include:
- Direction (Up/Down/Neutral)
- Confidence score (0-1)
- Clear explanation linking current conditions to historical events
- List of key factors
- Risk assessment

Be conservative and honest about uncertainty. If no strong match exists, say so."""

    # Create agent
    agent = create_agent(
        llm,
        [],
        system_prompt=system_prompt
    )

    return agent


def generate_forecast(
    current_profile: CurrentDayProfile,
    symbol: str = None
) -> Union[Forecast, NoMatchResult]:
    """
    Generate a forecast by comparing current profile to historical events.

    Args:
        current_profile: Current day profile with embedding
        symbol: Stock symbol (default from config)

    Returns:
        Forecast object or NoMatchResult
    """
    symbol = symbol or Config.STOCK_SYMBOL

    print(f"\n{'='*80}")
    print(f"GENERATING FORECAST FOR {symbol}")
    print(f"{'='*80}\n")

    # Step 1: Query vector database for similar events
    print("Step 1: Searching for similar historical events...")
    vector_store = VectorStoreManager()

    query_result = vector_store.query_similar_events(
        query_embedding=current_profile.embedding,
        n_results=10,
        similarity_threshold=Config.SIMILARITY_THRESHOLD
    )

    if not query_result['success'] or query_result['n_results'] == 0:
        print(f"  ⚠ No similar events found above threshold ({Config.SIMILARITY_THRESHOLD})")

        # Return no-match result
        return NoMatchResult(
            symbol=symbol,
            forecast_date=current_profile.date,
            threshold=Config.SIMILARITY_THRESHOLD,
            current_sentiment=current_profile.sentiment_label,
            current_news_summary=current_profile.news_summary
        )

    results = query_result['results']
    print(f"  ✓ Found {query_result['n_results']} similar events")

    # Step 2: Create matched event objects
    print("\nStep 2: Analyzing matched events...")
    matched_events = []

    for i in range(len(results['ids'])):
        event_id = results['ids'][i]
        distance = results['distances'][i]
        metadata = results['metadatas'][i]

        similarity_score = 1 - distance  # Convert distance to similarity

        matched_event = MatchedEvent(
            event_id=event_id,
            date=datetime.fromisoformat(metadata['date']),
            price_change_pct=metadata['price_change_pct'],
            similarity_score=similarity_score,
            news_summary=metadata.get('news_summary', ''),
            event_type=metadata.get('event_type'),
            next_day_change=metadata.get('1_day_change'),
            three_day_change=metadata.get('3_day_change')
        )

        matched_events.append(matched_event)
        print(f"    Match {i+1}: {event_id} (Similarity: {similarity_score:.1%})")

    # Step 3: Analyze subsequent performance of matched events
    print("\nStep 3: Computing historical outcomes...")
    best_match = matched_events[0]

    # Calculate average subsequent performance
    next_day_changes = [m.next_day_change for m in matched_events if m.next_day_change is not None]
    avg_next_day = sum(next_day_changes) / len(next_day_changes) if next_day_changes else 0.0

    print(f"  ✓ Average next-day change from similar events: {avg_next_day:+.2f}%")

    # Step 4: Determine prediction direction
    if avg_next_day > 1.0:
        predicted_direction = Direction.UP
    elif avg_next_day < -1.0:
        predicted_direction = Direction.DOWN
    else:
        predicted_direction = Direction.NEUTRAL

    # Calculate confidence based on similarity and consistency
    avg_similarity = sum(m.similarity_score for m in matched_events) / len(matched_events)
    confidence_score = avg_similarity * 0.8  # Base confidence on similarity

    # Adjust confidence based on consistency of historical outcomes
    if len(next_day_changes) >= 3:
        same_direction = sum(1 for x in next_day_changes if (x > 0) == (avg_next_day > 0))
        consistency = same_direction / len(next_day_changes)
        confidence_score = (confidence_score + consistency) / 2

    confidence_score = min(confidence_score, 0.95)  # Cap at 95%

    print(f"  ✓ Prediction: {predicted_direction.value} (Confidence: {confidence_score:.1%})")

    # Step 5: Generate reasoning with LLM
    print("\nStep 4: Generating reasoning...")
    agent = create_forecasting_agent()

    try:
        # Create context for LLM
        matches_summary = "\n".join([
            f"- {m.date.strftime('%Y-%m-%d')}: {m.price_change_pct:+.2f}% → Next day: {m.next_day_change:+.2f}% "
            f"(Similarity: {m.similarity_score:.1%})"
            for m in matched_events[:5]
            if m.next_day_change is not None
        ])

        prompt = f"""Generate a forecast explanation based on this analysis:

Current Conditions:
- Date: {current_profile.date.strftime('%Y-%m-%d')}
- Sentiment: {current_profile.sentiment_label}
- News: {current_profile.news_summary[:200]}

Similar Historical Events (Top 5):
{matches_summary}

Predicted Direction: {predicted_direction.value}
Average Historical Outcome: {avg_next_day:+.2f}%
Confidence: {confidence_score:.1%}

Provide a clear, 3-4 sentence explanation for this prediction. Explain:
1. Why current conditions match these historical events
2. What happened after similar events in the past
3. Key factors supporting this prediction
"""

        result = agent.invoke({"messages": [("user", prompt)]})
        reasoning = result['messages'][-1].content if result['messages'] else "Prediction based on historical similarity"
        print(f"  ✓ Reasoning generated")

    except Exception as e:
        print(f"  ⚠ Could not generate reasoning with LLM: {e}")
        reasoning = f"Current market conditions show {avg_similarity:.1%} similarity to {len(matched_events)} " \
                   f"historical events. After similar events, the stock moved an average of {avg_next_day:+.2f}% " \
                   f"the next day, suggesting {predicted_direction.value} movement."

    # Step 6: Identify key factors
    key_factors = [
        f"Current sentiment is {current_profile.sentiment_label}",
        f"Best historical match: {best_match.date.strftime('%Y-%m-%d')} with {best_match.similarity_score:.1%} similarity",
        f"Historical average outcome: {avg_next_day:+.2f}%",
        f"Based on {len(matched_events)} similar events"
    ]

    # Determine risk level
    if confidence_score >= 0.75:
        risk_level = "Low"
    elif confidence_score >= 0.5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Add caveats
    caveats = [
        "Past performance does not guarantee future results",
        "Market conditions can change rapidly",
        f"Prediction based on {len(matched_events)} historical events"
    ]

    if confidence_score < 0.7:
        caveats.append("Moderate confidence - exercise caution")

    # Step 7: Create Forecast object
    print("\nStep 5: Creating forecast report...")
    forecast = Forecast(
        symbol=symbol,
        forecast_date=current_profile.date,
        predicted_direction=predicted_direction,
        confidence_score=confidence_score,
        matched_events=matched_events,
        best_match_id=best_match.event_id,
        reasoning=reasoning,
        key_factors=key_factors,
        current_sentiment=current_profile.sentiment_label,
        current_news_summary=current_profile.news_summary,
        risk_level=risk_level,
        caveats=caveats
    )

    print(f"\n{'='*80}")
    print(f"FORECAST GENERATION COMPLETE")
    print(f"{'='*80}\n")

    return forecast


# Example usage
if __name__ == "__main__":
    print("Testing Forecasting Agent...")

    # This would normally be called with a CurrentDayProfile
    # from the real-time analyzer agent
    print("Note: Run this after initializing the database with historical events")
