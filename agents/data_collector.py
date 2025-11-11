"""Data Collection Agent for gathering stock and news data."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tools.stock_data import (
    fetch_historical_stock_data,
    fetch_current_stock_data,
    get_stock_info
)
from tools.news_scraper import (
    fetch_yahoo_finance_news,
    search_tesla_news_headlines
)
from tools.sentiment import analyze_sentiment, analyze_news_sentiment
from config import Config


def create_data_collection_agent():
    """
    Create the Data Collection Agent.

    This agent is responsible for:
    - Fetching historical stock data
    - Retrieving current stock prices
    - Gathering news articles
    - Collecting sentiment data

    Returns:
        LangGraph agent for data collection
    """

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        google_api_key=Config.GOOGLE_API_KEY
    )

    # Define tools available to this agent
    tools = [
        fetch_historical_stock_data,
        fetch_current_stock_data,
        get_stock_info,
        fetch_yahoo_finance_news,
        search_tesla_news_headlines,
        analyze_sentiment,
        analyze_news_sentiment
    ]

    # System prompt
    system_prompt = f"""You are a Data Collection Agent specialized in gathering stock market data and news.

Your responsibilities:
1. Fetch historical and current stock price data for {Config.STOCK_SYMBOL}
2. Retrieve news articles from financial sources
3. Analyze sentiment of news articles
4. Organize and structure all collected data

When collecting historical data:
- Use the fetch_historical_stock_data tool with the stock symbol
- Collect data for the specified time period ({Config.HISTORICAL_YEARS} years)

When collecting current data:
- Use fetch_current_stock_data for today's price information
- Use search_tesla_news_headlines to get recent news
- Use analyze_news_sentiment to assess market sentiment

Always provide comprehensive, well-structured data that includes:
- OHLCV price data (Open, High, Low, Close, Volume)
- Percentage price changes
- News headlines and summaries
- Sentiment scores and labels

Be thorough and accurate in your data collection."""

    # Create the agent
    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt
    )

    return agent


def collect_historical_data(symbol: str = None, years: int = None) -> dict:
    """
    High-level function to collect historical data using the agent.

    Args:
        symbol: Stock symbol (default from config)
        years: Number of years (default from config)

    Returns:
        Dictionary with collected data
    """
    symbol = symbol or Config.STOCK_SYMBOL
    years = years or Config.HISTORICAL_YEARS

    agent = create_data_collection_agent()

    # Create query for the agent
    query = f"""
    Collect comprehensive historical data for {symbol} stock:
    1. Fetch {years} years of historical price data
    2. Include OHLCV data and percentage changes
    3. Organize the data for analysis

    Return the data in a structured format.
    """

    # Invoke the agent
    result = agent.invoke({
        "messages": [("user", query)]
    })

    return result


def collect_current_data(symbol: str = None) -> dict:
    """
    High-level function to collect current day data using the agent.

    Args:
        symbol: Stock symbol (default from config)

    Returns:
        Dictionary with current data and news
    """
    symbol = symbol or Config.STOCK_SYMBOL

    agent = create_data_collection_agent()

    # Create query for the agent
    query = f"""
    Collect current market data for {symbol} stock:
    1. Fetch today's stock price data (OHLCV)
    2. Get recent news headlines (last 10-15 articles)
    3. Analyze the sentiment of the news
    4. Provide a summary of current market conditions

    Return all data in a structured format including:
    - Current price and changes
    - List of news articles with titles
    - Overall sentiment analysis
    - Market narrative summary
    """

    # Invoke the agent
    result = agent.invoke({
        "messages": [("user", query)]
    })

    return result


def collect_news_for_date(symbol: str, date: str) -> dict:
    """
    Collect news data for a specific date (best effort).

    Args:
        symbol: Stock symbol
        date: Date in YYYY-MM-DD format

    Returns:
        Dictionary with news data
    """
    agent = create_data_collection_agent()

    query = f"""
    Try to collect news information for {symbol} around the date {date}:
    1. Search for available news
    2. Analyze sentiment if news is found
    3. Provide context about market conditions

    Note: Historical news may be limited without a paid API.
    Return whatever information is available.
    """

    result = agent.invoke({
        "messages": [("user", query)]
    })

    return result


# Example usage and testing
if __name__ == "__main__":
    print("Testing Data Collection Agent...")

    # Test collecting current data
    print("\n=== Collecting Current Data ===")
    current_data = collect_current_data()
    print(f"Collected: {len(current_data.get('messages', []))} messages")

    # Test collecting historical data (just check if it works)
    print("\n=== Testing Historical Data Collection ===")
    print("Note: Full historical collection may take time...")
