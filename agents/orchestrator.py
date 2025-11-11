"""Orchestrator Agent for coordinating all agents in the multi-agent system."""
from datetime import datetime
from typing import Dict, Union

from config import Config
from models.forecast import Forecast, NoMatchResult
from agents.event_identifier import identify_and_store_historical_events
from agents.realtime_analyzer import analyze_current_day
from agents.forecaster import generate_forecast
from tools.vector_store import VectorStoreManager


class OrchestratorAgent:
    """
    Master orchestrator that coordinates all agents in the system.

    Workflow:
    1. Initialization: Build historical event database (one-time)
    2. Daily Execution: Analyze current day → Compare to history → Generate forecast
    """

    def __init__(self, symbol: str = None):
        """
        Initialize the orchestrator.

        Args:
            symbol: Stock symbol to analyze (default from config)
        """
        self.symbol = symbol or Config.STOCK_SYMBOL
        self.vector_store = VectorStoreManager()

    def initialize_system(self, years: int = None) -> Dict:
        """
        Initialize the system by building the historical event database.

        This should be run once to populate the vector database with
        historical significant events.

        Args:
            years: Number of years of historical data to analyze

        Returns:
            Dictionary with initialization results
        """
        print(f"\n{'='*80}")
        print(f"INITIALIZING TESLA STOCK ANALYSIS SYSTEM")
        print(f"{'='*80}\n")
        print(f"Symbol: {self.symbol}")
        print(f"Historical Period: {years or Config.HISTORICAL_YEARS} years")
        print(f"Significance Threshold: {Config.SIGNIFICANT_CHANGE_THRESHOLD}%")
        print(f"Similarity Threshold: {Config.SIMILARITY_THRESHOLD}")
        print(f"\nThis may take several minutes...\n")

        # Run the event identifier agent
        result = identify_and_store_historical_events(
            symbol=self.symbol,
            years=years
        )

        print(f"\n{'='*80}")
        print(f"SYSTEM INITIALIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Status: {'Success' if result['success'] else 'Failed'}")
        print(f"Events stored: {result['events_stored']}")
        print(f"Vector database: {self.vector_store.persist_directory}")
        print(f"{'='*80}\n")

        return result

    def run_daily_forecast(self) -> Union[Forecast, NoMatchResult]:
        """
        Execute the daily forecast workflow.

        Steps:
        1. Analyze current day conditions (Real-Time Analyzer)
        2. Compare to historical events (Similarity & Forecasting)
        3. Generate and return forecast

        Returns:
            Forecast object or NoMatchResult
        """
        print(f"\n{'='*80}")
        print(f"DAILY FORECAST WORKFLOW")
        print(f"{'='*80}\n")
        print(f"Symbol: {self.symbol}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        # Check if database has events
        event_count = self.vector_store.count_events()
        if event_count == 0:
            print("\n⚠️  WARNING: Vector database is empty!")
            print("Please run initialize_system() first to populate historical events.\n")
            return NoMatchResult(
                symbol=self.symbol,
                forecast_date=datetime.now(),
                message="System not initialized. Run initialization first.",
                threshold=Config.SIMILARITY_THRESHOLD,
                current_sentiment="Unknown",
                current_news_summary="System not initialized"
            )

        print(f"✓ Vector database contains {event_count} historical events\n")

        # Step 1: Analyze current day
        print("=" * 80)
        print("STEP 1: REAL-TIME ANALYSIS")
        print("=" * 80)
        current_profile = analyze_current_day(self.symbol)

        # Step 2: Generate forecast
        print("\n" + "=" * 80)
        print("STEP 2: SIMILARITY MATCHING & FORECASTING")
        print("=" * 80)
        forecast = generate_forecast(current_profile, self.symbol)

        # Step 3: Display results
        print("\n" + "=" * 80)
        print("STEP 3: FORECAST RESULTS")
        print("=" * 80)

        if isinstance(forecast, Forecast):
            print("\n✓ FORECAST GENERATED\n")
            print(forecast.to_report())
        else:
            print("\n⚠️  NO MATCH FOUND\n")
            print(forecast.to_report())

        return forecast

    def get_system_status(self) -> Dict:
        """
        Get the current status of the system.

        Returns:
            Dictionary with system status information
        """
        event_count = self.vector_store.count_events()

        status = {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "vector_database": {
                "path": self.vector_store.persist_directory,
                "collection": self.vector_store.collection_name,
                "event_count": event_count,
                "initialized": event_count > 0
            },
            "configuration": {
                "historical_years": Config.HISTORICAL_YEARS,
                "significance_threshold": Config.SIGNIFICANT_CHANGE_THRESHOLD,
                "similarity_threshold": Config.SIMILARITY_THRESHOLD,
                "llm_model": Config.LLM_MODEL
            }
        }

        return status

    def print_system_status(self):
        """Print a formatted system status report."""
        status = self.get_system_status()

        print(f"\n{'='*80}")
        print("SYSTEM STATUS")
        print(f"{'='*80}\n")
        print(f"Symbol: {status['symbol']}")
        print(f"Timestamp: {status['timestamp']}")
        print(f"\nVector Database:")
        print(f"  Location: {status['vector_database']['path']}")
        print(f"  Collection: {status['vector_database']['collection']}")
        print(f"  Events Stored: {status['vector_database']['event_count']}")
        print(f"  Initialized: {'Yes' if status['vector_database']['initialized'] else 'No'}")
        print(f"\nConfiguration:")
        print(f"  Historical Period: {status['configuration']['historical_years']} years")
        print(f"  Significance Threshold: {status['configuration']['significance_threshold']}%")
        print(f"  Similarity Threshold: {status['configuration']['similarity_threshold']}")
        print(f"  LLM Model: {status['configuration']['llm_model']}")
        print(f"{'='*80}\n")


def run_initialization(symbol: str = None, years: int = None):
    """
    Convenience function to run system initialization.

    Args:
        symbol: Stock symbol (default from config)
        years: Number of years (default from config)

    Returns:
        Initialization results
    """
    orchestrator = OrchestratorAgent(symbol=symbol)
    return orchestrator.initialize_system(years=years)


def run_daily_forecast(symbol: str = None):
    """
    Convenience function to run daily forecast.

    Args:
        symbol: Stock symbol (default from config)

    Returns:
        Forecast or NoMatchResult
    """
    orchestrator = OrchestratorAgent(symbol=symbol)
    return orchestrator.run_daily_forecast()


def print_system_status(symbol: str = None):
    """
    Convenience function to print system status.

    Args:
        symbol: Stock symbol (default from config)
    """
    orchestrator = OrchestratorAgent(symbol=symbol)
    orchestrator.print_system_status()


# Example usage
if __name__ == "__main__":
    print("Testing Orchestrator Agent...")

    # Create orchestrator
    orchestrator = OrchestratorAgent()

    # Check system status
    orchestrator.print_system_status()

    print("\nTo initialize the system, run:")
    print("  orchestrator.initialize_system()")
    print("\nTo generate a forecast, run:")
    print("  orchestrator.run_daily_forecast()")
