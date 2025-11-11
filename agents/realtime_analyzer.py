"""Real-Time Analysis Agent for current market conditions."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from datetime import datetime
from typing import Dict

from config import Config
from models.event_profile import CurrentDayProfile
from tools.stock_data import get_current_data_df
from tools.news_scraper import scrape_yahoo_news
from tools.sentiment import get_market_sentiment_summary
from utils.embeddings import generate_text_embedding


def create_realtime_analyzer_agent():
    """
    Create the Real-Time Analysis Agent.

    This agent focuses on gathering and analyzing current day information
    including news, sentiment, and market conditions.

    Returns:
        LangGraph agent for real-time analysis
    """

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        google_api_key=Config.GOOGLE_API_KEY
    )

    # System prompt
    system_prompt = f"""You are a Real-Time Market Analysis Agent specialized in analyzing current market conditions for {Config.STOCK_SYMBOL}.

Your responsibilities:
1. Gather the latest news and market information
2. Analyze current sentiment and market narrative
3. Summarize the key themes and factors affecting the stock today
4. Create a structured profile of current conditions

When analyzing current conditions:
- Focus on the most recent and relevant news
- Identify key themes (e.g., product news, earnings, macro factors, leadership)
- Assess overall market sentiment (Positive/Negative/Neutral)
- Provide clear, concise summaries

Your summaries should:
- Be factual and current
- Highlight the most important factors
- Be 3-5 sentences long
- Focus on market-moving information

Example summary: "Tesla news today centers on Q4 delivery numbers beating expectations,
with analysts raising price targets. Market sentiment is positive as production ramp
continues. Broader EV sector showing strength amid favorable policy discussions."
"""

    # Create agent without tools (uses direct functions)
    agent = create_agent(
        llm,
        [],
        system_prompt=system_prompt
    )

    return agent


def analyze_current_day(symbol: str = None) -> CurrentDayProfile:
    """
    Analyze current day conditions and create a profile.

    Args:
        symbol: Stock symbol (default from config)

    Returns:
        CurrentDayProfile object
    """
    symbol = symbol or Config.STOCK_SYMBOL

    print(f"\n{'='*80}")
    print(f"ANALYZING CURRENT DAY FOR {symbol}")
    print(f"{'='*80}\n")

    # Step 1: Get current stock data
    print("Step 1: Fetching current stock data...")
    try:
        df = get_current_data_df(symbol)
        if not df.empty:
            latest = df.iloc[-1]
            open_price = float(latest['Open'])
            current_price = float(latest['Close'])
            high_price = float(latest['High'])
            low_price = float(latest['Low'])
            volume = int(latest['Volume'])
            print(f"  ✓ Current price: ${current_price:.2f}")
        else:
            print("  ⚠ No current price data available (market may be closed)")
            open_price = current_price = high_price = low_price = None
            volume = None
    except Exception as e:
        print(f"  ✗ Error fetching stock data: {e}")
        open_price = current_price = high_price = low_price = None
        volume = None

    # Step 2: Scrape current news
    print("\nStep 2: Fetching current news...")
    try:
        articles = scrape_yahoo_news(symbol, max_articles=15)
        print(f"  ✓ Found {len(articles)} news articles")
    except Exception as e:
        print(f"  ✗ Error fetching news: {e}")
        articles = []

    # Step 3: Analyze sentiment
    print("\nStep 3: Analyzing sentiment...")
    sentiment_data = get_market_sentiment_summary(articles)
    sentiment_label = sentiment_data.get('overall_sentiment', 'Neutral')
    sentiment_score = sentiment_data.get('sentiment_score', 0.0)
    print(f"  ✓ Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")

    # Step 4: Generate narrative summary using LLM
    print("\nStep 4: Generating market narrative...")
    agent = create_realtime_analyzer_agent()

    try:
        # Create context for LLM
        news_titles = "\n".join([f"- {art['title']}" for art in articles[:10]])

        prompt = f"""Analyze current market conditions for Tesla ({symbol}) and provide a 3-5 sentence summary:

Current Price: ${current_price:.2f if current_price else 'N/A'}
Overall Sentiment: {sentiment_label}

Recent News Headlines:
{news_titles if news_titles else "No recent news available"}

Provide a concise narrative summary of the current market situation, key themes, and factors affecting the stock today.
"""

        result = agent.invoke({"messages": [("user", prompt)]})
        news_summary = result['messages'][-1].content if result['messages'] else "Current market analysis"
        print(f"  ✓ Generated narrative")

    except Exception as e:
        print(f"  ⚠ Could not generate narrative with LLM: {e}")
        # Fallback summary
        if articles:
            news_summary = f"Recent news includes: {', '.join([art['title'][:50] for art in articles[:3]])}..."
        else:
            news_summary = f"Limited news available for {symbol} today."

    # Step 5: Create Current Day Profile
    print("\nStep 5: Creating current day profile...")
    current_profile = CurrentDayProfile(
        date=datetime.now(),
        symbol=symbol,
        open_price=open_price,
        current_price=current_price,
        high_price=high_price,
        low_price=low_price,
        volume=volume,
        news_summary=news_summary,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        market_context=f"Analysis based on {len(articles)} news sources"
    )

    # Step 6: Generate embedding
    print("Step 6: Generating embedding...")
    text_description = current_profile.to_text_description()
    embedding = generate_text_embedding(text_description)
    current_profile.embedding = embedding
    print(f"  ✓ Embedding generated ({len(embedding)} dimensions)")

    print(f"\n{'='*80}")
    print(f"CURRENT DAY ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

    return current_profile


def get_current_market_summary(symbol: str = None) -> Dict:
    """
    Get a quick summary of current market conditions.

    Args:
        symbol: Stock symbol (default from config)

    Returns:
        Dictionary with summary information
    """
    symbol = symbol or Config.STOCK_SYMBOL

    try:
        # Get current data
        df = get_current_data_df(symbol)
        articles = scrape_yahoo_news(symbol, max_articles=10)
        sentiment = get_market_sentiment_summary(articles)

        if not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

            summary = {
                "symbol": symbol,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "price": float(latest['Close']),
                "change_pct": float(pct_change),
                "volume": int(latest['Volume']),
                "sentiment": sentiment.get('overall_sentiment', 'Neutral'),
                "sentiment_score": sentiment.get('sentiment_score', 0.0),
                "news_count": len(articles),
                "top_headline": articles[0]['title'] if articles else "No news available"
            }

            return summary

    except Exception as e:
        print(f"Error getting market summary: {e}")

    return {
        "symbol": symbol,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "error": "Could not fetch current data"
    }


# Example usage
if __name__ == "__main__":
    print("Testing Real-Time Analyzer Agent...")

    # Test current day analysis
    profile = analyze_current_day()

    print("\n=== Current Day Profile ===")
    print(profile.to_text_description())
