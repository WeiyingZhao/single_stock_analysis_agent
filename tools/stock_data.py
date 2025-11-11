"""Stock data fetching tools using Yahoo Finance."""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from langchain_core.tools import tool

@tool
def fetch_historical_stock_data(
    symbol: str,
    years: int = 5
) -> str:
    """
    Fetch historical stock price data (OHLCV) for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'TSLA')
        years: Number of years of historical data to fetch

    Returns:
        JSON string containing historical stock data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        # Fetch data using yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            return f"No data found for symbol {symbol}"

        # Calculate daily percentage change
        df['Pct_Change'] = df['Close'].pct_change() * 100

        # Convert to JSON-friendly format
        data_dict = {
            "symbol": symbol,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "total_records": len(df),
            "data": df.reset_index().to_dict('records')
        }

        return str(data_dict)

    except Exception as e:
        return f"Error fetching stock data: {str(e)}"


@tool
def fetch_current_stock_data(symbol: str) -> str:
    """
    Fetch current day's stock price data and recent performance.

    Args:
        symbol: Stock ticker symbol (e.g., 'TSLA')

    Returns:
        JSON string containing current stock data
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get recent data (last 5 days)
        df = ticker.history(period="5d")

        if df.empty:
            return f"No current data found for symbol {symbol}"

        # Get the most recent data
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Calculate change
        pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

        # Get info
        info = ticker.info

        result = {
            "symbol": symbol,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "close": float(latest['Close']),
            "volume": int(latest['Volume']),
            "pct_change": float(pct_change),
            "previous_close": float(prev['Close']),
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
        }

        return str(result)

    except Exception as e:
        return f"Error fetching current stock data: {str(e)}"


@tool
def get_stock_info(symbol: str) -> str:
    """
    Get detailed information about a stock including company info and key metrics.

    Args:
        symbol: Stock ticker symbol (e.g., 'TSLA')

    Returns:
        JSON string containing stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        result = {
            "symbol": symbol,
            "company_name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "forward_pe": info.get('forwardPE', 'N/A'),
            "dividend_yield": info.get('dividendYield', 'N/A'),
            "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
            "average_volume": info.get('averageVolume', 'N/A'),
            "beta": info.get('beta', 'N/A'),
        }

        return str(result)

    except Exception as e:
        return f"Error fetching stock info: {str(e)}"


def get_historical_data_df(symbol: str, years: int = 5) -> pd.DataFrame:
    """
    Get historical stock data as a pandas DataFrame (non-tool function for internal use).

    Args:
        symbol: Stock ticker symbol
        years: Number of years of historical data

    Returns:
        DataFrame with historical stock data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    # Calculate percentage change
    df['Pct_Change'] = df['Close'].pct_change() * 100

    return df


def get_current_data_df(symbol: str) -> pd.DataFrame:
    """
    Get current day's stock data as DataFrame (non-tool function for internal use).

    Args:
        symbol: Stock ticker symbol

    Returns:
        DataFrame with current stock data
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5d")

    # Calculate percentage change
    df['Pct_Change'] = df['Close'].pct_change() * 100

    return df
