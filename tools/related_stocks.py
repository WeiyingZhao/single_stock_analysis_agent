"""Related Stocks Analysis - Correlation and Sector Analysis."""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from langchain_core.tools import tool


# Common stock sector mappings and related tickers
SECTOR_TICKERS = {
    "TSLA": {
        "sector": "Electric Vehicles / Automotive",
        "peers": ["GM", "F", "RIVN", "LCID", "NIO"],  # Auto & EV competitors
        "related": ["AAPL", "NVDA", "GOOGL"],  # Tech companies
        "indices": ["SPY", "QQQ", "XLY"],  # S&P 500, NASDAQ, Consumer Discretionary
        "suppliers": ["PANW", "ASML"],  # Chip/tech suppliers
    },
    "AAPL": {
        "sector": "Technology / Consumer Electronics",
        "peers": ["MSFT", "GOOGL", "META", "AMZN"],
        "related": ["NVDA", "TSM", "QCOM"],
        "indices": ["SPY", "QQQ", "XLK"],
    },
    "NVDA": {
        "sector": "Technology / Semiconductors",
        "peers": ["AMD", "INTC", "TSM", "ASML"],
        "related": ["MSFT", "GOOGL", "AMZN"],
        "indices": ["SPY", "QQQ", "SMH"],
    },
    "MSFT": {
        "sector": "Technology / Software",
        "peers": ["GOOGL", "AAPL", "AMZN", "META"],
        "related": ["NVDA", "CRM", "ORCL"],
        "indices": ["SPY", "QQQ", "XLK"],
    },
}


def get_related_tickers(symbol: str) -> Dict[str, List[str]]:
    """
    Get related stock tickers for a given symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with categories of related stocks
    """
    if symbol in SECTOR_TICKERS:
        return SECTOR_TICKERS[symbol]

    # Default related stocks if not in predefined list
    return {
        "sector": "Unknown",
        "peers": [],
        "related": [],
        "indices": ["SPY", "QQQ"],  # At minimum, track major indices
        "suppliers": []
    }


def fetch_multiple_stocks_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks efficiently.

    Args:
        symbols: List of stock symbols
        start_date: Start date for historical data
        end_date: End date (default: today)

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    if end_date is None:
        end_date = datetime.now()

    data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if not df.empty:
                df['Pct_Change'] = df['Close'].pct_change() * 100
                data[symbol] = df
        except Exception as e:
            print(f"  ⚠ Warning: Could not fetch data for {symbol}: {e}")

    return data


def calculate_correlation(
    target_df: pd.DataFrame,
    related_dfs: Dict[str, pd.DataFrame],
    window_days: int = 30
) -> Dict[str, float]:
    """
    Calculate price correlation between target stock and related stocks.

    Args:
        target_df: DataFrame for target stock
        related_dfs: Dictionary of DataFrames for related stocks
        window_days: Rolling window for correlation (default 30 days)

    Returns:
        Dictionary mapping stock symbol to correlation coefficient
    """
    correlations = {}

    target_returns = target_df['Pct_Change'].dropna()

    for symbol, df in related_dfs.items():
        try:
            related_returns = df['Pct_Change'].dropna()

            # Align the series by date
            aligned_target, aligned_related = target_returns.align(
                related_returns,
                join='inner'
            )

            if len(aligned_target) >= window_days:
                # Calculate rolling correlation (use last N days)
                corr = aligned_target.tail(window_days).corr(
                    aligned_related.tail(window_days)
                )
                correlations[symbol] = float(corr) if not np.isnan(corr) else 0.0
            else:
                correlations[symbol] = 0.0

        except Exception as e:
            print(f"  ⚠ Warning: Could not calculate correlation for {symbol}: {e}")
            correlations[symbol] = 0.0

    return correlations


def analyze_related_stocks_on_date(
    symbol: str,
    target_date: datetime,
    lookback_days: int = 5
) -> Dict:
    """
    Analyze how related stocks performed around a specific date.

    Args:
        symbol: Target stock symbol
        target_date: Date to analyze
        lookback_days: Days before and after to analyze

    Returns:
        Dictionary with related stock analysis
    """
    related_info = get_related_tickers(symbol)

    # Collect all related tickers
    all_related = (
        related_info.get("peers", []) +
        related_info.get("related", []) +
        related_info.get("indices", [])
    )

    if not all_related:
        return {
            "sector": related_info.get("sector", "Unknown"),
            "correlations": {},
            "concurrent_movements": {},
            "sector_momentum": "Unknown"
        }

    # Fetch data for related stocks
    start_date = target_date - timedelta(days=lookback_days * 2)
    end_date = target_date + timedelta(days=lookback_days)

    related_data = fetch_multiple_stocks_data(all_related, start_date, end_date)

    # Get target stock data
    target_data = fetch_multiple_stocks_data([symbol], start_date, end_date)
    target_df = target_data.get(symbol)

    if target_df is None or target_df.empty:
        return {
            "sector": related_info.get("sector", "Unknown"),
            "correlations": {},
            "concurrent_movements": {},
            "sector_momentum": "Unknown"
        }

    # Calculate correlations
    correlations = calculate_correlation(target_df, related_data, window_days=lookback_days)

    # Analyze concurrent movements (same day performance)
    concurrent_movements = {}
    target_date_str = target_date.strftime('%Y-%m-%d')

    for sym, df in related_data.items():
        try:
            # Find the closest date to target_date
            df_dates = df.index.strftime('%Y-%m-%d')
            if target_date_str in df_dates.values:
                idx = df.index[df_dates == target_date_str][0]
                pct_change = df.loc[idx, 'Pct_Change']
                concurrent_movements[sym] = float(pct_change) if not np.isnan(pct_change) else 0.0
        except Exception:
            concurrent_movements[sym] = 0.0

    # Determine sector momentum
    if concurrent_movements:
        avg_movement = np.mean(list(concurrent_movements.values()))
        if avg_movement > 1.0:
            sector_momentum = "Positive"
        elif avg_movement < -1.0:
            sector_momentum = "Negative"
        else:
            sector_momentum = "Neutral"
    else:
        sector_momentum = "Unknown"

    return {
        "sector": related_info.get("sector", "Unknown"),
        "correlations": correlations,
        "concurrent_movements": concurrent_movements,
        "sector_momentum": sector_momentum,
        "peers": related_info.get("peers", []),
        "indices": related_info.get("indices", [])
    }


def get_current_related_stocks_status(symbol: str) -> Dict:
    """
    Get current status of related stocks (for real-time analysis).

    Args:
        symbol: Target stock symbol

    Returns:
        Dictionary with current related stocks status
    """
    related_info = get_related_tickers(symbol)

    # Collect all related tickers
    all_related = (
        related_info.get("peers", []) +
        related_info.get("indices", [])
    )

    if not all_related:
        return {
            "sector": related_info.get("sector", "Unknown"),
            "current_movements": {},
            "sector_momentum": "Unknown",
            "highly_correlated": []
        }

    # Fetch recent data (last 5 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)

    related_data = fetch_multiple_stocks_data(all_related, start_date, end_date)
    target_data = fetch_multiple_stocks_data([symbol], start_date, end_date)
    target_df = target_data.get(symbol)

    if target_df is None or target_df.empty:
        return {
            "sector": related_info.get("sector", "Unknown"),
            "current_movements": {},
            "sector_momentum": "Unknown",
            "highly_correlated": []
        }

    # Calculate correlations
    correlations = calculate_correlation(target_df, related_data, window_days=5)

    # Get current day movements
    current_movements = {}
    for sym, df in related_data.items():
        if not df.empty:
            latest_change = df['Pct_Change'].iloc[-1]
            current_movements[sym] = float(latest_change) if not np.isnan(latest_change) else 0.0

    # Identify highly correlated stocks (>0.7)
    highly_correlated = [
        sym for sym, corr in correlations.items()
        if abs(corr) > 0.7
    ]

    # Determine sector momentum
    if current_movements:
        avg_movement = np.mean(list(current_movements.values()))
        if avg_movement > 0.5:
            sector_momentum = "Positive"
        elif avg_movement < -0.5:
            sector_momentum = "Negative"
        else:
            sector_momentum = "Neutral"
    else:
        sector_momentum = "Unknown"

    return {
        "sector": related_info.get("sector", "Unknown"),
        "correlations": correlations,
        "current_movements": current_movements,
        "sector_momentum": sector_momentum,
        "highly_correlated": highly_correlated,
        "peers": related_info.get("peers", []),
        "indices": related_info.get("indices", [])
    }


@tool
def analyze_sector_correlation(symbol: str, days: int = 30) -> str:
    """
    Analyze correlation between stock and its sector peers.

    Args:
        symbol: Stock symbol to analyze
        days: Number of days for correlation analysis

    Returns:
        String describing sector correlations
    """
    result = get_current_related_stocks_status(symbol)

    correlations = result.get("correlations", {})
    if not correlations:
        return f"No correlation data available for {symbol}"

    # Sort by absolute correlation
    sorted_corr = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    report = f"Sector Correlation Analysis for {symbol}:\n"
    report += f"Sector: {result.get('sector', 'Unknown')}\n"
    report += f"Sector Momentum: {result.get('sector_momentum', 'Unknown')}\n\n"
    report += "Top Correlations:\n"

    for sym, corr in sorted_corr[:5]:
        report += f"  {sym}: {corr:+.3f}\n"

    return report


def add_related_stocks_context(symbol: str, date: datetime) -> str:
    """
    Generate textual context about related stocks for embedding.

    Args:
        symbol: Target stock symbol
        date: Date to analyze

    Returns:
        Text description of related stocks context
    """
    analysis = analyze_related_stocks_on_date(symbol, date, lookback_days=3)

    context = f"\nSector: {analysis['sector']}\n"
    context += f"Sector Momentum: {analysis['sector_momentum']}\n"

    # Add correlation info
    if analysis['correlations']:
        top_corr = sorted(
            analysis['correlations'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        context += "Highly Correlated Stocks:\n"
        for sym, corr in top_corr:
            if abs(corr) > 0.5:
                movement = analysis['concurrent_movements'].get(sym, 0.0)
                context += f"  - {sym} (correlation: {corr:.2f}, movement: {movement:+.2f}%)\n"

    return context
