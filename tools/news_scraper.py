"""News scraping tools for gathering Tesla-related news and articles."""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict
from langchain_core.tools import tool
import time


@tool
def fetch_yahoo_finance_news(symbol: str, max_articles: int = 10) -> str:
    """
    Fetch recent news articles from Yahoo Finance for a given stock symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'TSLA')
        max_articles: Maximum number of articles to fetch

    Returns:
        JSON string containing news articles
    """
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        articles = []
        news_items = soup.find_all('h3', class_='Mb(5px)', limit=max_articles)

        for item in news_items:
            try:
                title = item.get_text(strip=True)
                link_tag = item.find('a')
                link = link_tag['href'] if link_tag else None

                if link and not link.startswith('http'):
                    link = f"https://finance.yahoo.com{link}"

                articles.append({
                    'title': title,
                    'link': link,
                    'source': 'Yahoo Finance',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                continue

        result = {
            "symbol": symbol,
            "total_articles": len(articles),
            "articles": articles,
            "fetch_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return str(result)

    except Exception as e:
        return f"Error fetching Yahoo Finance news: {str(e)}"


@tool
def fetch_news_for_date_range(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Fetch news articles for a specific date range.

    Args:
        symbol: Stock ticker symbol (e.g., 'TSLA')
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'

    Returns:
        JSON string containing news articles for the date range
    """
    try:
        # In a production system, you would use a news API with date filtering
        # For now, we'll fetch current news and note the limitation
        articles = []

        # Fetch from Yahoo Finance
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        news_items = soup.find_all('h3', class_='Mb(5px)', limit=20)

        for item in news_items:
            try:
                title = item.get_text(strip=True)
                link_tag = item.find('a')
                link = link_tag['href'] if link_tag else None

                if link and not link.startswith('http'):
                    link = f"https://finance.yahoo.com{link}"

                articles.append({
                    'title': title,
                    'link': link,
                    'source': 'Yahoo Finance',
                    'date': datetime.now().strftime("%Y-%m-%d")
                })
            except Exception:
                continue

        result = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_articles": len(articles),
            "articles": articles,
            "note": "Historical news requires NewsAPI or similar service with API key"
        }

        return str(result)

    except Exception as e:
        return f"Error fetching news for date range: {str(e)}"


def scrape_yahoo_news(symbol: str, max_articles: int = 10) -> List[Dict]:
    """
    Scrape Yahoo Finance news (non-tool function for internal use).

    Args:
        symbol: Stock ticker symbol
        max_articles: Maximum number of articles to fetch

    Returns:
        List of article dictionaries
    """
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        articles = []
        news_items = soup.find_all('h3', class_='Mb(5px)', limit=max_articles)

        for item in news_items:
            try:
                title = item.get_text(strip=True)
                link_tag = item.find('a')
                link = link_tag['href'] if link_tag else None

                if link and not link.startswith('http'):
                    link = f"https://finance.yahoo.com{link}"

                articles.append({
                    'title': title,
                    'link': link,
                    'source': 'Yahoo Finance',
                    'timestamp': datetime.now()
                })
            except Exception:
                continue

        return articles

    except Exception as e:
        print(f"Error scraping Yahoo news: {e}")
        return []


def get_article_content(url: str) -> str:
    """
    Fetch the full content of a news article.

    Args:
        url: URL of the article

    Returns:
        Article content as string
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to find article content (varies by site)
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text(strip=True) for p in paragraphs])

        return content

    except Exception as e:
        return f"Error fetching article content: {str(e)}"


@tool
def search_tesla_news_headlines(query: str = "Tesla") -> str:
    """
    Search for Tesla-related news headlines from multiple sources.

    Args:
        query: Search query (default: "Tesla")

    Returns:
        JSON string containing aggregated news headlines
    """
    try:
        all_articles = []

        # Yahoo Finance
        yahoo_articles = scrape_yahoo_news("TSLA", max_articles=10)
        all_articles.extend(yahoo_articles)

        # Could add more sources here (Google News, etc.)
        # For demonstration, we'll use Yahoo Finance as primary source

        result = {
            "query": query,
            "total_articles": len(all_articles),
            "articles": [
                {
                    'title': art['title'],
                    'source': art['source'],
                    'link': art.get('link', 'N/A'),
                    'timestamp': art['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(art['timestamp'], datetime)
                    else str(art['timestamp'])
                }
                for art in all_articles
            ],
            "fetch_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return str(result)

    except Exception as e:
        return f"Error searching news headlines: {str(e)}"
