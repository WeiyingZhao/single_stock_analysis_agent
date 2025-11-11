#!/usr/bin/env python3
"""
Initialization script for Tesla Stock Analysis System.

This script should be run ONCE to populate the vector database with
historical significant events. It will:
1. Fetch historical stock data
2. Identify significant price movements
3. Create event profiles
4. Store embeddings in Chroma vector database

Usage:
    python initialize_db.py [--years YEARS] [--symbol SYMBOL]

Example:
    python initialize_db.py --years 5 --symbol TSLA
"""

import argparse
import sys
from datetime import datetime

from config import Config
from agents.orchestrator import OrchestratorAgent


def main():
    """Main function for database initialization."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Initialize Tesla Stock Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with default settings (5 years of TSLA data)
  python initialize_db.py

  # Initialize with custom years
  python initialize_db.py --years 3

  # Initialize with different symbol
  python initialize_db.py --symbol AAPL

  # Reinitialize (clears existing data)
  python initialize_db.py --force
        """
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default=Config.STOCK_SYMBOL,
        help=f'Stock symbol to analyze (default: {Config.STOCK_SYMBOL})'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=Config.HISTORICAL_YEARS,
        help=f'Number of years of historical data (default: {Config.HISTORICAL_YEARS})'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reinitialization (clears existing data)'
    )

    args = parser.parse_args()

    # Display configuration
    print(f"\n{'='*80}")
    print("TESLA STOCK ANALYSIS SYSTEM - INITIALIZATION")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Historical Period: {args.years} years")
    print(f"  Significance Threshold: {Config.SIGNIFICANT_CHANGE_THRESHOLD}%")
    print(f"  Similarity Threshold: {Config.SIMILARITY_THRESHOLD}")
    print(f"  LLM Model: {Config.LLM_MODEL}")
    print(f"  Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"\n{'='*80}\n")

    # Create orchestrator
    orchestrator = OrchestratorAgent(symbol=args.symbol)

    # Check if already initialized
    event_count = orchestrator.vector_store.count_events()
    if event_count > 0 and not args.force:
        print(f"⚠️  WARNING: Database already contains {event_count} events!")
        print("\nOptions:")
        print("  1. Run with --force to reinitialize (clears existing data)")
        print("  2. Skip initialization and use existing data")
        print("\nProceed with reinitialization? (yes/no): ", end='')

        response = input().strip().lower()
        if response not in ['yes', 'y']:
            print("\nInitialization cancelled. Using existing data.")
            orchestrator.print_system_status()
            return

        print("\nClearing existing data...")
        orchestrator.vector_store.clear_collection()
        print("✓ Data cleared")

    # Run initialization
    try:
        start_time = datetime.now()
        print(f"\nStarting initialization at {start_time.strftime('%Y-%m-%d %H:%M:%S')}...")
        print("This may take several minutes depending on the amount of data...\n")

        result = orchestrator.initialize_system(years=args.years)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Display results
        print(f"\n{'='*80}")
        print("INITIALIZATION COMPLETE")
        print(f"{'='*80}\n")
        print(f"Status: {'✓ Success' if result['success'] else '✗ Failed'}")
        print(f"Events Found: {result['events_found']}")
        print(f"Events Processed: {result['events_processed']}")
        print(f"Events Stored: {result['events_stored']}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"\n{'='*80}\n")

        if result['events_stored'] > 0:
            print("✓ System is ready for forecasting!")
            print("\nNext steps:")
            print("  1. Run 'python main.py' for daily forecast")
            print("  2. Or import and use: from agents.orchestrator import run_daily_forecast")
        else:
            print("⚠️  No events were stored. Check configuration and data availability.")

        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Initialization interrupted by user.")
        print("Database may be partially populated.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n✗ ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
