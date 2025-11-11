"""Statistical analysis utilities for stock data."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from config import Config


def identify_significant_changes(
    df: pd.DataFrame,
    threshold_pct: float = None
) -> pd.DataFrame:
    """
    Identify days with significant price changes.

    Args:
        df: DataFrame with stock data (must have 'Pct_Change' column)
        threshold_pct: Percentage threshold for significance

    Returns:
        DataFrame with only significant change days
    """
    threshold = threshold_pct or Config.SIGNIFICANT_CHANGE_THRESHOLD

    # Filter for significant changes (absolute value)
    significant = df[df['Pct_Change'].abs() >= threshold].copy()

    return significant


def get_surrounding_context(
    df: pd.DataFrame,
    target_date: datetime,
    hours: int = 72
) -> pd.DataFrame:
    """
    Get data surrounding a target date within a time window.

    Args:
        df: DataFrame with stock data
        target_date: Target date
        hours: Number of hours before and after

    Returns:
        DataFrame with surrounding data
    """
    days = hours / 24
    start_date = target_date - timedelta(days=days)
    end_date = target_date + timedelta(days=days)

    # Filter data
    mask = (df.index >= start_date) & (df.index <= end_date)
    context_df = df.loc[mask].copy()

    return context_df


def calculate_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).

    Args:
        df: DataFrame with stock data
        window: Rolling window size in days

    Returns:
        Series with volatility values
    """
    returns = df['Close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

    return volatility


def calculate_moving_averages(
    df: pd.DataFrame,
    windows: List[int] = [20, 50, 200]
) -> pd.DataFrame:
    """
    Calculate moving averages for given windows.

    Args:
        df: DataFrame with stock data
        windows: List of window sizes

    Returns:
        DataFrame with original data and moving averages
    """
    result = df.copy()

    for window in windows:
        col_name = f'MA_{window}'
        result[col_name] = result['Close'].rolling(window=window).mean()

    return result


def get_subsequent_performance(
    df: pd.DataFrame,
    target_date: datetime,
    days: List[int] = [1, 3]
) -> Dict[str, float]:
    """
    Get stock performance in days following a target date.

    Args:
        df: DataFrame with stock data
        target_date: Target date
        days: List of day offsets to check

    Returns:
        Dictionary with performance metrics
    """
    performance = {}

    try:
        # Get the target price
        target_price = df.loc[target_date, 'Close']

        for day_offset in days:
            future_date = target_date + timedelta(days=day_offset)

            # Find the next available trading day
            while future_date not in df.index and future_date <= df.index[-1]:
                future_date += timedelta(days=1)

            if future_date in df.index:
                future_price = df.loc[future_date, 'Close']
                pct_change = ((future_price - target_price) / target_price) * 100
                performance[f'{day_offset}_day_change'] = round(pct_change, 2)
            else:
                performance[f'{day_offset}_day_change'] = None

    except Exception as e:
        print(f"Error calculating subsequent performance: {e}")

    return performance


def calculate_relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).

    Args:
        df: DataFrame with stock data
        window: RSI window size

    Returns:
        Series with RSI values
    """
    delta = df['Close'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def detect_trend(df: pd.DataFrame, window: int = 20) -> str:
    """
    Detect the current trend of the stock.

    Args:
        df: DataFrame with stock data
        window: Window size for trend detection

    Returns:
        Trend description ('Uptrend', 'Downtrend', 'Sideways')
    """
    if len(df) < window:
        return "Insufficient Data"

    # Calculate short-term MA
    ma_short = df['Close'].rolling(window=window).mean()

    # Compare current price to MA
    current_price = df['Close'].iloc[-1]
    current_ma = ma_short.iloc[-1]

    # Check trend direction
    ma_slope = (ma_short.iloc[-1] - ma_short.iloc[-window]) / window

    if current_price > current_ma and ma_slope > 0:
        return "Uptrend"
    elif current_price < current_ma and ma_slope < 0:
        return "Downtrend"
    else:
        return "Sideways"


def get_price_statistics(df: pd.DataFrame, window: int = 252) -> Dict[str, float]:
    """
    Calculate various price statistics.

    Args:
        df: DataFrame with stock data
        window: Window size (default: 1 year of trading days)

    Returns:
        Dictionary with statistics
    """
    recent_df = df.tail(window) if len(df) > window else df

    stats = {
        "current_price": float(df['Close'].iloc[-1]),
        "52_week_high": float(recent_df['High'].max()),
        "52_week_low": float(recent_df['Low'].min()),
        "average_volume": int(recent_df['Volume'].mean()),
        "average_daily_return": float(recent_df['Pct_Change'].mean()),
        "return_std_dev": float(recent_df['Pct_Change'].std()),
        "total_return_pct": float(((df['Close'].iloc[-1] - recent_df['Close'].iloc[0])
                                   / recent_df['Close'].iloc[0]) * 100),
    }

    # Distance from 52-week high/low
    stats["distance_from_high_pct"] = float(
        ((df['Close'].iloc[-1] - stats["52_week_high"]) / stats["52_week_high"]) * 100
    )
    stats["distance_from_low_pct"] = float(
        ((df['Close'].iloc[-1] - stats["52_week_low"]) / stats["52_week_low"]) * 100
    )

    return stats


def categorize_event_type(price_change_pct: float, news_summary: str) -> str:
    """
    Categorize the type of event based on price change and news.

    Args:
        price_change_pct: Percentage price change
        news_summary: Summary of news for the period

    Returns:
        Event type category
    """
    news_lower = news_summary.lower()

    # Check for specific event types in news
    if any(word in news_lower for word in ['earnings', 'revenue', 'profit', 'quarterly']):
        return "Earnings"
    elif any(word in news_lower for word in ['delivery', 'deliveries', 'production', 'manufactured']):
        return "Production/Delivery"
    elif any(word in news_lower for word in ['product', 'launch', 'announcement', 'unveil']):
        return "Product Launch"
    elif any(word in news_lower for word in ['recall', 'investigation', 'lawsuit', 'regulatory']):
        return "Regulatory/Legal"
    elif any(word in news_lower for word in ['ceo', 'executive', 'leadership', 'musk']):
        return "Leadership/Management"
    elif any(word in news_lower for word in ['market', 'economy', 'fed', 'interest']):
        return "Macro-Economic"
    else:
        # Categorize by magnitude
        if abs(price_change_pct) >= 10:
            return "Major Movement"
        elif abs(price_change_pct) >= 5:
            return "Significant Movement"
        else:
            return "Other"


def analyze_volume_pattern(df: pd.DataFrame, target_date: datetime) -> Dict[str, any]:
    """
    Analyze volume patterns around a target date.

    Args:
        df: DataFrame with stock data
        target_date: Target date

    Returns:
        Dictionary with volume analysis
    """
    try:
        # Get 30-day average volume
        avg_volume = df['Volume'].rolling(window=30).mean()
        target_volume = df.loc[target_date, 'Volume']
        avg_at_target = avg_volume.loc[target_date]

        volume_ratio = target_volume / avg_at_target if avg_at_target > 0 else 1.0

        analysis = {
            "target_volume": int(target_volume),
            "average_volume": int(avg_at_target),
            "volume_ratio": round(volume_ratio, 2),
            "volume_category": "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.5 else "Low"
        }

        return analysis

    except Exception as e:
        print(f"Error analyzing volume pattern: {e}")
        return {}
