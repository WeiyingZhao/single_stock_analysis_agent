# Tesla Stock Analysis Multi-Agent System

A sophisticated AI-powered stock analysis and forecasting system that learns from historical price patterns using multi-agent architecture, vector similarity matching, and LLM reasoning.

## Overview

This system uses **5 specialized AI agents** built with LangChain/LangGraph to analyze Tesla (TSLA) stock movements and predict future direction based on pattern matching against historical events. Unlike traditional technical analysis, it combines:

- **Vector embeddings** to find semantically similar market conditions
- **LLM reasoning** (Google Gemini) to understand causal factors
- **Historical pattern matching** across 5 years of significant events
- **Multi-agent collaboration** for comprehensive analysis

## Key Features

- **🤖 Multi-Agent Architecture**: 5 specialized agents work together autonomously
- **🧠 AI-Powered Analysis**: Uses Google Gemini for event summarization and prediction reasoning
- **📊 Vector Similarity Matching**: Finds similar historical patterns using semantic embeddings
- **📈 Pattern Learning**: Learns from 5 years of significant price movements (±5%)
- **🎯 Confidence Scoring**: Provides predictions with confidence levels and risk assessment
- **📰 Real-Time Data**: Integrates current stock prices and news sentiment
- **🔍 Explainable Predictions**: Generates human-readable reasoning for forecasts
- **💾 Local Vector Database**: Uses Chroma for persistent event storage
- **🆓 Free Data Sources**: Yahoo Finance (no API key needed for stock data)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get free key](https://makersuite.google.com/app/apikey))
- Internet connection

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/single_stock_analysis_agent.git
cd single_stock_analysis_agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Usage

**Step 1: Initialize the system (one-time, ~5-15 minutes)**

```bash
python initialize_db.py
```

This will:
- Download 5 years of TSLA historical data
- Identify significant price movements (±5%)
- Generate AI summaries for each event
- Create vector embeddings and store in database
- Typically finds ~45+ significant events

**Step 2: Run daily forecast**

```bash
python main.py
```

You'll get a detailed forecast with:
- Predicted direction (Up/Down/Neutral)
- Confidence score (0-100%)
- Similar historical events with outcomes
- AI-generated reasoning and key factors
- Risk assessment

### Command-Line Options

```bash
# Basic forecast
python main.py

# Different stock symbol (requires reinitialization)
python initialize_db.py --symbol AAPL
python main.py --symbol AAPL

# Custom historical period
python initialize_db.py --years 3

# Save forecast to file
python main.py --save forecast_2025-01-15.txt

# Check system status
python main.py --status

# Minimal output
python main.py --quiet

# Force reinitialize (clears existing data)
python initialize_db.py --force
```

## Architecture

### Multi-Agent System

The system coordinates 5 specialized agents through an orchestrator:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR AGENT                          │
│         (Coordinates workflow & manages system state)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌───────────────────┐                   ┌─────────────────────┐
│ INITIALIZATION    │                   │  DAILY FORECAST     │
│   WORKFLOW        │                   │    WORKFLOW         │
└────────┬──────────┘                   └──────────┬──────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────────┐               ┌──────────────────────┐
│ Data Collection     │               │ Real-Time Analyzer   │
│ Agent               │               │ Agent                │
│ • Fetch stock data  │               │ • Current prices     │
│ • Scrape news       │               │ • Latest news        │
│ • Analyze sentiment │               │ • Market sentiment   │
└────────┬────────────┘               └──────────┬───────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────────┐               ┌──────────────────────┐
│ Event Identifier    │               │ Similarity &         │
│ Agent               │               │ Forecasting Agent    │
│ • Find ±5% moves    │               │ • Vector search      │
│ • LLM summaries     │               │ • Match patterns     │
│ • Create embeddings │               │ • Analyze outcomes   │
│ • Store in Chroma   │               │ • Generate forecast  │
└─────────────────────┘               └──────────────────────┘
```

### How It Works

1. **Initialization Phase** (one-time):
   - Data Collection Agent fetches 5 years of historical TSLA data
   - Event Identifier Agent finds significant price movements (±5% threshold)
   - For each event, analyzes 72-hour context window (news, sentiment, price action)
   - Uses Gemini LLM to generate event summaries and identify causal factors
   - Creates vector embeddings using sentence transformers
   - Stores events with embeddings in Chroma vector database

2. **Daily Forecast Phase**:
   - Real-Time Analyzer Agent fetches current price and scrapes latest news
   - Analyzes current sentiment and generates market narrative using LLM
   - Creates current day profile with vector embedding
   - Similarity & Forecasting Agent searches vector database for similar past events
   - Matches current conditions to historical events (85% similarity threshold)
   - Analyzes subsequent performance of matched events (1-day, 3-day returns)
   - Uses LLM to generate prediction with confidence score and reasoning

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | LangChain + LangGraph | Multi-agent orchestration |
| **LLM** | Google Gemini 2.0 Flash | Event analysis and reasoning |
| **Vector DB** | ChromaDB | Event storage and similarity search |
| **Embeddings** | Sentence Transformers | Text-to-vector conversion (all-MiniLM-L6-v2) |
| **Stock Data** | Yahoo Finance (yfinance) | Historical and real-time prices |
| **News** | Web scraping | Real-time headlines and sentiment |
| **Data Models** | Pydantic | Type-safe data structures |

## Configuration

All settings can be customized via `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *required* | Google Gemini API key |
| `STOCK_SYMBOL` | `TSLA` | Target stock symbol |
| `HISTORICAL_YEARS` | `5` | Years of data to analyze |
| `SIGNIFICANT_CHANGE_THRESHOLD` | `5.0` | Min % change to identify events |
| `SIMILARITY_THRESHOLD` | `0.85` | Min similarity for matching (0-1) |
| `CONTEXT_WINDOW_HOURS` | `72` | Event context window |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector database location |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |

See `.env.example` for complete configuration options.

## Example Output

```
================================================================================
TESLA STOCK FORECAST REPORT
================================================================================

Forecast Date: 2025-01-15
Symbol: TSLA
Current Price: $387.45

PREDICTION
----------
Direction: UP ↑
Confidence: 87.5%
Risk Level: Medium

REASONING
---------
Current market conditions show 92% similarity to October 2023 when Tesla
announced record Q3 deliveries. Today's positive sentiment around production
expansion and strong delivery numbers mirrors that historical event. The
matched events showed an average 2.8% increase over the next trading day.

Key factors:
• Positive news sentiment (score: 0.82)
• Production/delivery announcement
• Strong volume confirmation
• Sector momentum alignment

SIMILAR HISTORICAL EVENTS
-------------------------
1. 2023-10-15 (Similarity: 92.3%)
   Event: Record Q3 deliveries announced
   Next day: +3.2% | 3-day: +4.1%

2. 2023-07-22 (Similarity: 89.1%)
   Event: Production milestone reached
   Next day: +2.5% | 3-day: +3.8%

3. 2024-04-03 (Similarity: 88.7%)
   Event: Factory expansion news
   Next day: +1.9% | 3-day: +2.9%

RISK ASSESSMENT
---------------
• Market volatility: Moderate
• News sentiment strength: High
• Historical pattern consistency: Strong
• Sample size: 3 similar events

DISCLAIMER
----------
This forecast is for educational and research purposes only. Not financial advice.
Past performance does not guarantee future results.

================================================================================
```

## Project Structure

```
single_stock_analysis_agent/
├── agents/                         # Multi-agent system
│   ├── orchestrator.py            # Master coordinator
│   ├── data_collector.py          # Stock & news data gathering
│   ├── event_identifier.py        # Historical event detection
│   ├── realtime_analyzer.py       # Current day analysis
│   └── forecaster.py              # Similarity matching & prediction
│
├── tools/                         # Data source integrations
│   ├── stock_data.py              # Yahoo Finance wrapper
│   ├── news_scraper.py            # Web scraping utilities
│   ├── sentiment.py               # Sentiment analysis
│   └── vector_store.py            # ChromaDB manager
│
├── models/                        # Data models
│   ├── event_profile.py           # EventProfile, CurrentDayProfile
│   └── forecast.py                # Forecast, NoMatchResult
│
├── utils/                         # Utilities
│   ├── embeddings.py              # Sentence transformer wrapper
│   └── analysis.py                # Statistical analysis
│
├── config.py                      # Configuration management
├── initialize_db.py               # One-time database setup
├── main.py                        # Daily forecast execution
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
└── README.md                      # This file
```

## Troubleshooting

### "GOOGLE_API_KEY not found"
- Ensure `.env` file exists in project root
- Verify API key is correctly set (no quotes, no spaces)
- Get a free key from https://makersuite.google.com/app/apikey

### "Vector database is empty"
- Run `python initialize_db.py` first to populate the database
- Check that initialization completed successfully
- Verify `chroma_db/` directory exists

### "No similar events found"
- This is a valid result - the system is being conservative
- Current conditions don't match historical patterns above 85% threshold
- Try lowering `SIMILARITY_THRESHOLD` in `.env` (e.g., `0.75`)

### Import errors or missing modules
```bash
pip install --upgrade -r requirements.txt
```

### Web scraping failures
- Yahoo Finance occasionally blocks requests
- The system will retry and fall back gracefully
- Check internet connection and firewall settings

### Slow initialization
- Normal for 5 years of data (~5-15 minutes)
- Requires multiple LLM calls for event analysis
- Use `--years 2` for faster initialization during testing

## Limitations

This is an **educational and research project**, not a production trading system:

- **Free data sources**: Yahoo Finance has rate limits; historical news is incomplete
- **Single stock focus**: Optimized for TSLA; other stocks require reinitialization
- **Market hours dependency**: Best results during trading hours with fresh data
- **No backtesting**: Historical accuracy not systematically validated
- **Sentiment analysis**: Keyword-based (not transformer-based)
- **No risk management**: Does not include position sizing or stop losses
- **API rate limits**: Google Gemini free tier has usage limits

## Extending the System

### Add a new stock

```bash
python initialize_db.py --symbol AAPL --force
python main.py --symbol AAPL
```

### Modify similarity threshold

Edit `.env`:
```
SIMILARITY_THRESHOLD=0.90  # More strict matching
```

### Add custom data sources

1. Create tool function in `tools/` directory
2. Add to agent's tool list in respective agent file
3. Update agent's system prompt

### Run programmatically

```python
from agents.orchestrator import OrchestratorAgent

# Initialize
orchestrator = OrchestratorAgent(symbol="TSLA")
result = orchestrator.initialize_system(years=5)

# Get forecast
forecast = orchestrator.run_daily_forecast()
print(forecast.to_report())
```

## Testing

Test individual agents:

```bash
python -m agents.data_collector       # Test data collection
python -m agents.event_identifier     # Test event identification
python -m agents.realtime_analyzer    # Test current analysis
python -m agents.forecaster          # Test forecasting
python -m agents.orchestrator        # Test orchestration
```

## Performance

- **Initialization**: 5-15 minutes (one-time)
- **Daily forecast**: 30-60 seconds
- **Vector search**: <1 second
- **Storage**: ~50MB for 5 years of events

## Contributing

Contributions are welcome! Areas for improvement:

- Add backtesting framework
- Implement advanced NLP sentiment models
- Support multiple stocks in single database
- Add technical indicators (RSI, MACD)
- Create web UI dashboard
- Add unit and integration tests
- Improve error handling and retry logic

Please open an issue first to discuss proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Not intended for actual trading decisions
- Past performance does not guarantee future results
- The authors assume no liability for financial losses
- Always consult a qualified financial advisor before trading

## Acknowledgments

- **LangChain/LangGraph**: Multi-agent framework
- **Google Gemini**: LLM for reasoning and analysis
- **ChromaDB**: Vector database
- **Yahoo Finance**: Free stock data
- **Sentence Transformers**: Embedding models

---

**Ready to forecast?** Run `python initialize_db.py` then `python main.py`!
