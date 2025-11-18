"""Technical Indicators for Stock Analysis."""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from langchain_core.tools import tool


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices
        period: RSI period (default 14)

    Returns:
        Series of RSI values
    """
    delta = prices.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of closing prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of closing prices
        period: Moving average period
        std_dev: Number of standard deviations

    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)

    return upper_band, middle_band, lower_band


def calculate_moving_averages(
    prices: pd.Series,
    periods: list = [20, 50, 200]
) -> Dict[str, pd.Series]:
    """
    Calculate multiple simple moving averages.

    Args:
        prices: Series of closing prices
        periods: List of periods for MAs

    Returns:
        Dictionary of moving averages
    """
    mas = {}
    for period in periods:
        mas[f"SMA_{period}"] = prices.rolling(window=period).mean()

    return mas


def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate volume-based indicators.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary of volume indicators
    """
    indicators = {}

    # Average volume
    indicators['avg_volume_20d'] = df['Volume'].tail(20).mean()
    indicators['avg_volume_50d'] = df['Volume'].tail(50).mean()

    # Volume ratio (current vs average)
    if len(df) > 0:
        current_volume = df['Volume'].iloc[-1]
        indicators['volume_ratio'] = current_volume / indicators['avg_volume_20d'] if indicators['avg_volume_20d'] > 0 else 1.0

    # On-Balance Volume (OBV)
    obv = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
    indicators['obv'] = obv.iloc[-1] if len(obv) > 0 else 0

    return indicators


def get_technical_analysis(df: pd.DataFrame) -> Dict:
    """
    Comprehensive technical analysis of a stock.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary with all technical indicators and signals
    """
    if df.empty or len(df) < 50:
        return {
            "error": "Insufficient data for technical analysis",
            "data_points": len(df)
        }

    analysis = {}

    # Get closing prices
    closes = df['Close']

    # RSI
    rsi = calculate_rsi(closes)
    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
    analysis['rsi'] = {
        'value': float(current_rsi),
        'signal': 'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'
    }

    # MACD
    macd_line, signal_line, histogram = calculate_macd(closes)
    if not macd_line.empty:
        analysis['macd'] = {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1]),
            'trend': 'Bullish' if histogram.iloc[-1] > 0 else 'Bearish'
        }

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(closes)
    current_price = closes.iloc[-1]
    if not upper.empty:
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        middle_val = middle.iloc[-1]

        # Calculate position within bands
        band_width = upper_val - lower_val
        position = (current_price - lower_val) / band_width if band_width > 0 else 0.5

        analysis['bollinger_bands'] = {
            'upper': float(upper_val),
            'middle': float(middle_val),
            'lower': float(lower_val),
            'position': float(position),
            'signal': 'Near Upper' if position > 0.8 else 'Near Lower' if position < 0.2 else 'Middle'
        }

    # Moving Averages
    mas = calculate_moving_averages(closes)
    ma_analysis = {}

    for name, ma in mas.items():
        if not ma.empty:
            ma_val = ma.iloc[-1]
            ma_analysis[name] = {
                'value': float(ma_val),
                'position': 'Above' if current_price > ma_val else 'Below'
            }

    analysis['moving_averages'] = ma_analysis

    # Trend Analysis
    if 'SMA_50' in ma_analysis and 'SMA_200' in ma_analysis:
        sma_50 = ma_analysis['SMA_50']['value']
        sma_200 = ma_analysis['SMA_200']['value']

        if sma_50 > sma_200:
            trend = 'Golden Cross' if len(df) > 200 else 'Bullish'
        else:
            trend = 'Death Cross' if len(df) > 200 else 'Bearish'

        analysis['trend'] = {
            'signal': trend,
            'strength': abs((sma_50 - sma_200) / sma_200) * 100
        }

    # Volume Analysis
    volume_indicators = calculate_volume_indicators(df)
    analysis['volume'] = volume_indicators

    # Price Momentum
    if len(df) >= 5:
        momentum_5d = ((closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5]) * 100
        analysis['momentum'] = {
            '5_day': float(momentum_5d)
        }

        if len(df) >= 20:
            momentum_20d = ((closes.iloc[-1] - closes.iloc[-20]) / closes.iloc[-20]) * 100
            analysis['momentum']['20_day'] = float(momentum_20d)

    return analysis


def interpret_technical_signals(analysis: Dict) -> Dict[str, str]:
    """
    Interpret technical indicators into trading signals.

    Args:
        analysis: Output from get_technical_analysis

    Returns:
        Dictionary with interpretations and overall signal
    """
    if 'error' in analysis:
        return {'overall': 'Insufficient Data', 'signals': []}

    signals = []
    bullish_count = 0
    bearish_count = 0

    # RSI Signal
    if 'rsi' in analysis:
        rsi_signal = analysis['rsi']['signal']
        if rsi_signal == 'Oversold':
            signals.append("RSI indicates oversold condition (potential buy)")
            bullish_count += 1
        elif rsi_signal == 'Overbought':
            signals.append("RSI indicates overbought condition (potential sell)")
            bearish_count += 1

    # MACD Signal
    if 'macd' in analysis:
        macd_trend = analysis['macd']['trend']
        if macd_trend == 'Bullish':
            signals.append("MACD shows bullish momentum")
            bullish_count += 1
        else:
            signals.append("MACD shows bearish momentum")
            bearish_count += 1

    # Bollinger Bands
    if 'bollinger_bands' in analysis:
        bb_signal = analysis['bollinger_bands']['signal']
        if bb_signal == 'Near Lower':
            signals.append("Price near lower Bollinger Band (potential support)")
            bullish_count += 1
        elif bb_signal == 'Near Upper':
            signals.append("Price near upper Bollinger Band (potential resistance)")
            bearish_count += 1

    # Moving Averages
    if 'moving_averages' in analysis:
        ma_20 = analysis['moving_averages'].get('SMA_20', {})
        if ma_20.get('position') == 'Above':
            signals.append("Price above 20-day MA (short-term strength)")
            bullish_count += 1
        else:
            signals.append("Price below 20-day MA (short-term weakness)")
            bearish_count += 1

    # Trend
    if 'trend' in analysis:
        trend_signal = analysis['trend']['signal']
        if 'Bullish' in trend_signal or 'Golden' in trend_signal:
            signals.append(f"Long-term trend: {trend_signal}")
            bullish_count += 1
        else:
            signals.append(f"Long-term trend: {trend_signal}")
            bearish_count += 1

    # Overall signal
    if bullish_count > bearish_count:
        overall = "Bullish"
    elif bearish_count > bullish_count:
        overall = "Bearish"
    else:
        overall = "Neutral"

    return {
        'overall': overall,
        'signals': signals,
        'bullish_indicators': bullish_count,
        'bearish_indicators': bearish_count
    }


def get_technical_context_text(df: pd.DataFrame) -> str:
    """
    Generate textual description of technical analysis for embedding.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Text description of technical indicators
    """
    analysis = get_technical_analysis(df)

    if 'error' in analysis:
        return "Technical analysis unavailable due to insufficient data."

    interpretation = interpret_technical_signals(analysis)

    context = f"\nTechnical Analysis:\n"
    context += f"Overall Technical Signal: {interpretation['overall']}\n"

    # RSI
    if 'rsi' in analysis:
        context += f"RSI: {analysis['rsi']['value']:.1f} ({analysis['rsi']['signal']})\n"

    # MACD
    if 'macd' in analysis:
        context += f"MACD: {analysis['macd']['trend']} (Histogram: {analysis['macd']['histogram']:.2f})\n"

    # Bollinger Bands
    if 'bollinger_bands' in analysis:
        context += f"Bollinger Position: {analysis['bollinger_bands']['signal']}\n"

    # Trend
    if 'trend' in analysis:
        context += f"Trend: {analysis['trend']['signal']}\n"

    # Volume
    if 'volume' in analysis:
        vol_ratio = analysis['volume'].get('volume_ratio', 1.0)
        context += f"Volume: {vol_ratio:.2f}x average\n"

    # Key signals
    if interpretation['signals']:
        context += "\nKey Technical Signals:\n"
        for signal in interpretation['signals'][:3]:  # Top 3 signals
            context += f"  - {signal}\n"

    return context


@tool
def analyze_technical_indicators(symbol: str) -> str:
    """
    Analyze technical indicators for a stock.

    Args:
        symbol: Stock symbol

    Returns:
        String description of technical analysis
    """
    from tools.stock_data import get_historical_data_df

    try:
        df = get_historical_data_df(symbol, years=1)
        return get_technical_context_text(df)
    except Exception as e:
        return f"Error analyzing technical indicators: {e}"
