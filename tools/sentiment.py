"""Sentiment analysis tools for news and text."""
from typing import Dict, List
from langchain_core.tools import tool
import re


@tool
def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of a given text using keyword-based approach.

    Args:
        text: Text to analyze

    Returns:
        JSON string containing sentiment analysis results
    """
    try:
        # Simple keyword-based sentiment analysis
        # In production, use a proper NLP model or API

        positive_keywords = [
            'profit', 'gain', 'surge', 'rally', 'bullish', 'positive', 'growth',
            'increase', 'breakthrough', 'success', 'milestone', 'record', 'beat',
            'outperform', 'upgrade', 'innovation', 'expand', 'strong', 'boost'
        ]

        negative_keywords = [
            'loss', 'decline', 'drop', 'fall', 'bearish', 'negative', 'decrease',
            'crash', 'failure', 'concern', 'risk', 'miss', 'underperform', 'downgrade',
            'layoff', 'recall', 'investigation', 'lawsuit', 'delay', 'weak', 'disappointing'
        ]

        text_lower = text.lower()

        # Count keyword occurrences
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)

        # Calculate sentiment score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.0
            sentiment_label = "Neutral"
        else:
            sentiment_score = (positive_count - negative_count) / total
            if sentiment_score > 0.2:
                sentiment_label = "Positive"
            elif sentiment_score < -0.2:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

        result = {
            "text_length": len(text),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "confidence": round(abs(sentiment_score), 3)
        }

        return str(result)

    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"


@tool
def analyze_news_sentiment(articles: str) -> str:
    """
    Analyze sentiment of multiple news articles.

    Args:
        articles: JSON string or list of article titles/content

    Returns:
        JSON string containing aggregated sentiment analysis
    """
    try:
        # Parse articles if it's a string representation
        import ast
        if isinstance(articles, str):
            try:
                articles = ast.literal_eval(articles)
            except:
                # If parsing fails, treat as single text
                return analyze_sentiment(articles)

        # Analyze each article
        sentiments = []
        for article in articles:
            if isinstance(article, dict):
                text = article.get('title', '') + ' ' + article.get('content', '')
            else:
                text = str(article)

            sentiment_result = analyze_sentiment_internal(text)
            sentiments.append(sentiment_result)

        # Aggregate results
        avg_score = sum(s['sentiment_score'] for s in sentiments) / len(sentiments)
        positive_articles = sum(1 for s in sentiments if s['sentiment_label'] == 'Positive')
        negative_articles = sum(1 for s in sentiments if s['sentiment_label'] == 'Negative')
        neutral_articles = sum(1 for s in sentiments if s['sentiment_label'] == 'Neutral')

        # Overall sentiment
        if avg_score > 0.2:
            overall_sentiment = "Positive"
        elif avg_score < -0.2:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        result = {
            "total_articles": len(sentiments),
            "average_sentiment_score": round(avg_score, 3),
            "overall_sentiment": overall_sentiment,
            "positive_articles": positive_articles,
            "negative_articles": negative_articles,
            "neutral_articles": neutral_articles,
            "sentiment_breakdown": sentiments
        }

        return str(result)

    except Exception as e:
        return f"Error analyzing news sentiment: {str(e)}"


def analyze_sentiment_internal(text: str) -> Dict:
    """
    Internal sentiment analysis function (non-tool).

    Args:
        text: Text to analyze

    Returns:
        Dictionary with sentiment analysis results
    """
    positive_keywords = [
        'profit', 'gain', 'surge', 'rally', 'bullish', 'positive', 'growth',
        'increase', 'breakthrough', 'success', 'milestone', 'record', 'beat',
        'outperform', 'upgrade', 'innovation', 'expand', 'strong', 'boost',
        'soar', 'excellent', 'outstanding', 'impressive', 'optimistic'
    ]

    negative_keywords = [
        'loss', 'decline', 'drop', 'fall', 'bearish', 'negative', 'decrease',
        'crash', 'failure', 'concern', 'risk', 'miss', 'underperform', 'downgrade',
        'layoff', 'recall', 'investigation', 'lawsuit', 'delay', 'weak', 'disappointing',
        'plunge', 'slump', 'warning', 'threat', 'crisis', 'trouble'
    ]

    text_lower = text.lower()

    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)

    total = positive_count + negative_count
    if total == 0:
        sentiment_score = 0.0
        sentiment_label = "Neutral"
    else:
        sentiment_score = (positive_count - negative_count) / total
        if sentiment_score > 0.2:
            sentiment_label = "Positive"
        elif sentiment_score < -0.2:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sentiment_score": round(sentiment_score, 3),
        "sentiment_label": sentiment_label,
        "confidence": round(abs(sentiment_score), 3)
    }


def get_market_sentiment_summary(articles: List[Dict]) -> Dict:
    """
    Get overall market sentiment from a list of articles.

    Args:
        articles: List of article dictionaries with 'title' and optional 'content'

    Returns:
        Dictionary with market sentiment summary
    """
    if not articles:
        return {
            "overall_sentiment": "Neutral",
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "article_count": 0
        }

    sentiments = []
    for article in articles:
        text = article.get('title', '')
        if article.get('content'):
            text += ' ' + article['content']

        sentiment = analyze_sentiment_internal(text)
        sentiments.append(sentiment)

    avg_score = sum(s['sentiment_score'] for s in sentiments) / len(sentiments)

    if avg_score > 0.2:
        overall_sentiment = "Positive"
    elif avg_score < -0.2:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return {
        "overall_sentiment": overall_sentiment,
        "sentiment_score": round(avg_score, 3),
        "confidence": round(abs(avg_score), 3),
        "article_count": len(articles),
        "positive_count": sum(1 for s in sentiments if s['sentiment_label'] == 'Positive'),
        "negative_count": sum(1 for s in sentiments if s['sentiment_label'] == 'Negative'),
        "neutral_count": sum(1 for s in sentiments if s['sentiment_label'] == 'Neutral')
    }
