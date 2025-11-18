#!/usr/bin/env python3
"""
Stock Analysis and Forecasting System - Main Entry Point

Multi-agent AI system for stock forecasting using historical pattern matching,
technical indicators, sector correlation analysis, and related stock movements.

Supports any stock symbol - just initialize the database for the target stock.

Usage:
    python main.py [--symbol SYMBOL] [--status] [--save]

Example:
    python main.py                  # Run forecast for default symbol
    python main.py --symbol TSLA    # Run forecast for TSLA
    python main.py --symbol AAPL    # Run forecast for AAPL
    python main.py --symbol NVDA    # Run forecast for NVDA
    python main.py --status         # Show system status only
    python main.py --save output.txt  # Save forecast to file
"""

import argparse
import sys
from datetime import datetime

from config import Config
from agents.orchestrator import OrchestratorAgent
from models.forecast import Forecast, NoMatchResult


def main():
    """Main function for running daily forecasts."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Stock Analysis and Forecasting System - AI-Powered Multi-Agent Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily forecast for default symbol
  python main.py

  # Analyze specific stocks
  python main.py --symbol AAPL
  python main.py --symbol NVDA
  python main.py --symbol MSFT

  # Check system status
  python main.py --status

  # Run forecast and save to file
  python main.py --save forecast_2024-01-20.txt

Notes:
  - The system must be initialized first (run initialize_db.py --symbol SYMBOL)
  - Requires GOOGLE_API_KEY in .env file
  - Uses free Yahoo Finance data
  - Supports any stock symbol with sufficient historical data
        """
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=Config.STOCK_SYMBOL,
        help=f'Stock symbol to analyze (default: {Config.STOCK_SYMBOL})'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )

    parser.add_argument(
        '--save',
        type=str,
        metavar='FILENAME',
        help='Save forecast report to file'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimize output (only show final forecast)'
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = OrchestratorAgent(symbol=args.symbol)

    # If status flag, show status and exit
    if args.status:
        orchestrator.print_system_status()
        return

    # Display header
    if not args.quiet:
        print(f"\n{'='*80}")
        print("STOCK ANALYSIS AND FORECASTING SYSTEM")
        print("AI-Powered Multi-Agent Platform")
        print(f"{'='*80}\n")
        print(f"Symbol: {args.symbol}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"LLM Model: {Config.LLM_MODEL}")
        print(f"Features: Technical Indicators, Sector Analysis, Pattern Matching")
        print(f"{'='*80}\n")

    # Check if system is initialized
    event_count = orchestrator.vector_store.count_events()
    if event_count == 0:
        print("✗ ERROR: System not initialized!")
        print("\nThe vector database is empty. Please run initialization first:")
        print("  python initialize_db.py")
        print("\nThis will populate the database with historical events.")
        sys.exit(1)

    if not args.quiet:
        print(f"✓ System initialized with {event_count} historical events\n")

    # Run daily forecast
    try:
        if not args.quiet:
            print("Starting daily forecast workflow...\n")

        forecast = orchestrator.run_daily_forecast()

        # Generate report
        report = forecast.to_report()

        # Display report
        if args.quiet and isinstance(forecast, Forecast):
            # Quiet mode: just show key info
            print(f"\nForecast for {args.symbol} on {forecast.forecast_date.strftime('%Y-%m-%d')}:")
            print(f"  Direction: {forecast.predicted_direction.value}")
            print(f"  Confidence: {forecast.confidence_score:.1%}")
            print(f"  Risk: {forecast.risk_level}")
        else:
            # Full report
            print(report)

        # Save to file if requested
        if args.save:
            try:
                with open(args.save, 'w') as f:
                    f.write(report)
                print(f"\n✓ Forecast saved to: {args.save}")
            except Exception as e:
                print(f"\n✗ Error saving forecast: {e}")

        # Return appropriate exit code
        if isinstance(forecast, NoMatchResult):
            sys.exit(2)  # No match found
        else:
            sys.exit(0)  # Success

    except KeyboardInterrupt:
        print("\n\n⚠️  Forecast interrupted by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n✗ ERROR during forecast: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_forecast_programmatically(symbol: str = None, verbose: bool = True):
    """
    Run forecast programmatically (for use in other scripts).

    Args:
        symbol: Stock symbol to analyze
        verbose: Whether to print progress

    Returns:
        Forecast or NoMatchResult object
    """
    orchestrator = OrchestratorAgent(symbol=symbol)

    # Check initialization
    if orchestrator.vector_store.count_events() == 0:
        raise RuntimeError("System not initialized. Run initialize_db.py first.")

    # Run forecast
    forecast = orchestrator.run_daily_forecast()

    if verbose:
        print(forecast.to_report())

    return forecast


if __name__ == "__main__":
    main()
