#!/usr/bin/env python3
"""
csv_to_sqlite.py

Universal CSV to SQLite upserter for OHLCV data.
Part of the data_manager.data_sources pipeline.

Supports multiple CSV formats with auto-detection:
  1. TradingView exports:
     - Format A: Unix, Date, Symbol, Open, High, Low, Close, Volume BTC, Volume USDT, tradecount
     - Format B: unix, date, symbol, open, high, low, close, volume, volume_from, tradecount
  2. influx_to_sql.py exports: symbol, timeframe, ts, o, h, l, c, v, source

Features:
- Auto-detects CSV format from column names
- Standardizes symbols to UPPERCASE WITHOUT SLASH (BTCUSDT format, matches influx_to_sql.py)
- Converts timestamps (ms → seconds, matches schema.sql)
- Uses base volume (Volume BTC or volume column) consistently
- Batch upsert via DataManager.upsert_ohlcv()
- Progress reporting and validation

Usage:
    # From data_manager/data_sources directory
    python csv_to_sqlite.py --csv ../../exports/data.csv --db ../../exports/marketdata.db --timeframe 1h

    # As module
    python -m data_manager.data_sources.csv_to_sqlite --csv data.csv --db marketdata.db --timeframe 1h

    # Multiple files
    python csv_to_sqlite.py --csv file1.csv file2.csv file3.csv --db marketdata.db --timeframe 1m

Python API:
    from data_manager.data_sources.csv_to_sqlite import upsert_csv_to_sqlite

    await upsert_csv_to_sqlite(
        csv_path='data.csv',
        db_path='marketdata.db',
        timeframe='1h',  # required for TradingView format
        source='csv_import',
        normalize_symbols=True,  # default, standardizes to BTCUSDT format
        batch_size=500
    )
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)

# Import DataManager using relative import
try:
    from ..data_manager.manager import DataManager
except ImportError:
    # Fallback for direct script execution
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data_manager.manager import DataManager
    except ImportError:
        print("Error: Cannot import DataManager. Run from data_manager directory or install as package.")
        sys.exit(1)


# ========== CSV Format Detection ==========

def detect_csv_format(df: pd.DataFrame) -> str:
    """
    Detect CSV format based on column names.

    Returns:
        'influx_export': influx_to_sql.py export format
        'tradingview': TradingView export format (uppercase columns)
        'tradingview_lower': TradingView variant (lowercase columns)
        'unknown': unrecognized format
    """
    cols = set(df.columns)

    # Format 2: influx_to_sql export (symbol, timeframe, ts, o, h, l, c, v, source)
    if 'ts' in cols and 'o' in cols and 'h' in cols and 'l' in cols and 'c' in cols and 'v' in cols:
        return 'influx_export'

    # Format 1: TradingView (Unix, Date, Symbol, Open, High, Low, Close, Volume BTC, ...)
    if 'Unix' in cols and 'Open' in cols and 'High' in cols and 'Low' in cols and 'Close' in cols:
        return 'tradingview'

    # Variant: lowercase unix
    if 'unix' in cols and 'open' in cols and 'high' in cols and 'low' in cols and 'close' in cols:
        return 'tradingview_lower'

    return 'unknown'


def normalize_symbol(symbol: str, remove_slash: bool = True) -> str:
    """
    Normalize symbol to standard format: UPPERCASE WITHOUT SLASH.

    This matches the standardization used by influx_to_sql.py (symbol.upper()).

    Examples:
        'btcusdt' → 'BTCUSDT'
        'BTC/USDT' → 'BTCUSDT'
        '  eth/usdt  ' → 'ETHUSDT'

    Args:
        symbol: Input symbol (e.g., 'BTC/USDT', 'BTCUSDT', 'btcusdt')
        remove_slash: If True, remove '/' separator (default: True)

    Returns:
        Standardized symbol string (UPPERCASE, NO SLASH)
    """
    s = symbol.strip().upper()
    if remove_slash:
        s = s.replace('/', '')
    return s


# ========== CSV Parsing ==========

def parse_tradingview_csv(
    df: pd.DataFrame,
    timeframe: str,
    source: str = 'csv_import',
    normalize_symbols: bool = True
) -> List[Dict[str, Any]]:
    """
    Parse TradingView format CSV.

    Expected columns (case-insensitive):
        Format A: Unix, Date, Symbol, Open, High, Low, Close, Volume BTC, Volume USDT, tradecount
        Format B: unix, date, symbol, open, high, low, close, volume, volume_from, tradecount

    Args:
        df: DataFrame with CSV data
        timeframe: Timeframe string (e.g., '1h', '30m', '1d')
        source: Source tag for database (default: 'csv_import')
        normalize_symbols: If True, standardize symbols to BTCUSDT format (default: True)

    Returns:
        List of dicts with keys: symbol, timeframe, ts, o, h, l, c, v, source
    """
    # Normalize column names to lowercase for case-insensitive matching
    df_cols = {col.lower(): col for col in df.columns}

    # Required columns (case-insensitive)
    required = {
        'unix': df_cols.get('unix'),
        'symbol': df_cols.get('symbol'),
        'open': df_cols.get('open'),
        'high': df_cols.get('high'),
        'low': df_cols.get('low'),
        'close': df_cols.get('close'),
    }

    # Volume column: try multiple variants (base volume)
    # Priority: "Volume BTC" > "volume_btc" > "volume"
    volume_col = None
    for vol_name in ['volume btc', 'volume_btc', 'volume']:
        if vol_name in df_cols:
            volume_col = df_cols[vol_name]
            break

    if volume_col is None:
        required['volume'] = None  # Mark as missing

    missing = [k for k, v in required.items() if v is None]
    if missing or volume_col is None:
        raise ValueError(f"Missing required columns in TradingView CSV: {missing + (['volume'] if volume_col is None else [])}")

    rows = []
    for _, row in df.iterrows():
        try:
            # Extract values using actual column names
            ts_ms = int(row[required['unix']])
            ts = ts_ms // 1000  # Convert ms to seconds (schema.sql format)

            symbol_raw = str(row[required['symbol']])
            symbol = normalize_symbol(symbol_raw) if normalize_symbols else symbol_raw

            o = float(row[required['open']])
            h = float(row[required['high']])
            l = float(row[required['low']])
            c = float(row[required['close']])
            v = float(row[volume_col])  # Use base volume (BTC)

            rows.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'ts': ts,
                'o': o,
                'h': h,
                'l': l,
                'c': c,
                'v': v,
                'source': source
            })
        except (ValueError, KeyError) as e:
            # Skip malformed rows
            print(f"Warning: Skipping malformed row: {e}")
            continue

    return rows


def parse_influx_export_csv(
    df: pd.DataFrame,
    normalize_symbols: bool = True,
    override_source: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Parse influx_to_sql.py export format CSV.

    Expected columns:
        symbol, timeframe, ts, o, h, l, c, v, source

    Args:
        df: DataFrame with CSV data
        normalize_symbols: If True, standardize symbols to BTCUSDT format (default: True)
        override_source: Override source tag from CSV (optional)

    Returns:
        List of dicts with keys: symbol, timeframe, ts, o, h, l, c, v, source
    """
    required = ['symbol', 'timeframe', 'ts', 'o', 'h', 'l', 'c', 'v']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in influx_export CSV: {missing}")

    rows = []
    for _, row in df.iterrows():
        try:
            ts_ms = int(row['ts'])
            ts = ts_ms // 1000  # Convert ms to seconds (schema.sql format)

            symbol_raw = str(row['symbol'])
            symbol = normalize_symbol(symbol_raw) if normalize_symbols else symbol_raw

            o = float(row['o'])
            h = float(row['h'])
            l = float(row['l'])
            c = float(row['c'])
            v = float(row['v'])

            # Use source from CSV if present, otherwise use override
            source = override_source if override_source else row.get('source', 'csv_import')

            rows.append({
                'symbol': symbol,
                'timeframe': row['timeframe'],
                'ts': ts,
                'o': o,
                'h': h,
                'l': l,
                'c': c,
                'v': v,
                'source': source
            })
        except (ValueError, KeyError) as e:
            print(f"Warning: Skipping malformed row: {e}")
            continue

    return rows


# ========== Upsert Logic ==========

async def upsert_rows_to_db(
    dm: DataManager,
    rows: List[Dict[str, Any]],
    batch_size: int = 500,
    verbose: bool = True
) -> int:
    """
    Upsert parsed rows to SQLite using DataManager.

    Groups rows by (symbol, timeframe) and batches upserts.

    Args:
        dm: DataManager instance
        rows: List of parsed row dicts
        batch_size: Number of rows per upsert batch (default: 500)
        verbose: Print progress messages (default: True)

    Returns:
        Total number of rows upserted
    """
    # Group by (symbol, timeframe)
    groups: Dict[Tuple[str, str], List[List[Any]]] = {}

    for row in rows:
        key = (row['symbol'], row['timeframe'])
        if key not in groups:
            groups[key] = []

        # Format: [ts, o, h, l, c, v]
        groups[key].append([
            row['ts'],
            row['o'],
            row['h'],
            row['l'],
            row['c'],
            row['v']
        ])

    total_upserted = 0

    for (symbol, timeframe), batch_rows in groups.items():
        if verbose:
            print(f"[INFO] Upserting {len(batch_rows)} rows for {symbol} {timeframe}...")

        # Get source from first row (all rows in group should have same source)
        source = next((r['source'] for r in rows if r['symbol'] == symbol and r['timeframe'] == timeframe), 'csv_import')

        # Process in batches
        for i in range(0, len(batch_rows), batch_size):
            batch = batch_rows[i:i+batch_size]
            count = await dm.upsert_ohlcv(symbol, timeframe, batch, source=source)
            total_upserted += count

            if verbose and len(batch_rows) > batch_size:
                print(f"  Batch {i//batch_size + 1}/{(len(batch_rows)-1)//batch_size + 1} ({count} rows)")

    return total_upserted


async def upsert_csv_to_sqlite(
    csv_path: str,
    db_path: str,
    timeframe: Optional[str] = None,
    source: Optional[str] = None,
    normalize_symbols: bool = True,  # DEFAULT: True (standardization enabled)
    batch_size: int = 500,
    verbose: bool = True
) -> int:
    """
    Main function: Load CSV, detect format, parse, and upsert to SQLite.

    Standardization (normalize_symbols=True, default):
        - Symbols: UPPERCASE WITHOUT SLASH (BTCUSDT, matches influx_to_sql.py)
        - Timestamps: Seconds (matches schema.sql ts INTEGER format)
        - Volume: Base volume (BTC for BTCUSDT)

    Args:
        csv_path: Path to CSV file
        db_path: Path to SQLite database
        timeframe: Timeframe (required for TradingView format, optional for influx_export)
        source: Source tag (optional, defaults based on format)
        normalize_symbols: If True, standardize symbols to BTCUSDT format (default: True)
        batch_size: Number of rows per upsert batch (default: 500)
        verbose: Print progress messages (default: True)

    Returns:
        Number of rows upserted

    Raises:
        FileNotFoundError: CSV file not found
        ValueError: Unknown CSV format or missing required columns
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if verbose:
        print(f"[INFO] Loading CSV: {csv_file.name}")

    # Load CSV with pandas
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    if df.empty:
        if verbose:
            print("[WARN] CSV file is empty. Skipping.")
        return 0

    # Detect format
    fmt = detect_csv_format(df)
    if verbose:
        print(f"[INFO] Detected format: {fmt}")

    if fmt == 'unknown':
        raise ValueError(f"Unknown CSV format. Expected TradingView or influx_export format.\nColumns found: {list(df.columns)}")

    # Parse based on format
    if fmt in ('tradingview', 'tradingview_lower'):
        if not timeframe:
            raise ValueError("TradingView format requires --timeframe argument")

        default_source = source or 'csv_tradingview'
        rows = parse_tradingview_csv(df, timeframe, default_source, normalize_symbols)

    elif fmt == 'influx_export':
        default_source = source or None  # Use source from CSV
        rows = parse_influx_export_csv(df, normalize_symbols, override_source=default_source)

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if not rows:
        if verbose:
            print("[WARN] No valid rows parsed from CSV. Skipping.")
        return 0

    if verbose:
        print(f"[INFO] Parsed {len(rows)} valid rows")

        # Show timestamp range (with error handling for invalid timestamps)
        min_ts = min(r['ts'] for r in rows)
        max_ts = max(r['ts'] for r in rows)
        from datetime import datetime

        try:
            # Try to format timestamps as dates
            min_dt = datetime.fromtimestamp(min_ts)
            max_dt = datetime.fromtimestamp(max_ts)
            print(f"[INFO] Timestamp range: {min_dt} to {max_dt}")
        except (OSError, ValueError, OverflowError) as e:
            # Fallback: show raw timestamps if conversion fails (Windows-safe formatting)
            print(f"[INFO] Timestamp range (raw): {min_ts} to {max_ts} seconds")
            print(f"[WARN] Could not convert timestamps to dates: {e}")

        # Show unique symbols
        symbols = set(r['symbol'] for r in rows)
        print(f"[INFO] Symbols: {', '.join(sorted(symbols))}")

        # Show unique timeframes
        timeframes = set(r['timeframe'] for r in rows)
        print(f"[INFO] Timeframes: {', '.join(sorted(timeframes))}")

    # Initialize DataManager
    dm = DataManager(db_path=db_path)
    await dm.init_db()

    try:
        # Upsert to database
        total = await upsert_rows_to_db(dm, rows, batch_size, verbose)

        if verbose:
            print(f"[SUCCESS] Upserted {total} rows to {db_path}")

        return total

    finally:
        await dm.close()


# ========== CLI ==========

def parse_args():
    parser = argparse.ArgumentParser(
        description="CSV to SQLite upserter for OHLCV data (data_manager pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Influx export (auto-detected, symbols auto-standardized to BTCUSDT)
  python csv_to_sqlite.py --csv influx_export.csv --db ../../exports/marketdata.db

  # TradingView export (requires timeframe, auto-standardizes symbols)
  python csv_to_sqlite.py --csv tradingview.csv --db ../../exports/marketdata.db --timeframe 1h

  # Multiple files (batch import)
  python csv_to_sqlite.py --csv file1.csv file2.csv file3.csv --db ../../exports/marketdata.db --timeframe 1m

  # Preserve original symbol format (disable standardization)
  python csv_to_sqlite.py --csv data.csv --db marketdata.db --no-normalize-symbol

Standardization (enabled by default):
  - Symbols: UPPERCASE WITHOUT SLASH (BTCUSDT, not BTC/USDT)
  - Timestamps: Seconds (ms → s conversion)
  - Volume: Base volume (Volume BTC or volume column)
        """
    )

    parser.add_argument(
        '--csv',
        type=str,
        nargs='+',
        required=True,
        help="CSV file(s) to import"
    )

    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help="SQLite database path (e.g., ../../exports/marketdata.db)"
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default=None,
        help="Timeframe (required for TradingView format, e.g., 1h, 30m, 1d)"
    )

    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help="Source tag (optional, defaults based on format)"
    )

    parser.add_argument(
        '--no-normalize-symbol',
        action='store_true',
        help="Disable symbol standardization (preserve original format like BTC/USDT)"
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help="Batch size for upsert operations (default: 500)"
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress messages"
    )

    return parser.parse_args()


async def main_async():
    args = parse_args()

    total_rows = 0

    for csv_file in args.csv:
        try:
            count = await upsert_csv_to_sqlite(
                csv_path=csv_file,
                db_path=args.db,
                timeframe=args.timeframe,
                source=args.source,
                normalize_symbols=not args.no_normalize_symbol,  # Default: True (standardization enabled)
                batch_size=args.batch_size,
                verbose=not args.quiet
            )
            total_rows += count

        except Exception as e:
            print(f"[ERROR] Failed to process {csv_file}: {e}")
            import traceback
            traceback.print_exc()

    if not args.quiet:
        print(f"\n[SUMMARY] Total rows upserted across all files: {total_rows}")


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
