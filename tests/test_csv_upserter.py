#!/usr/bin/env python3
"""
test_csv_upserter.py

Unit tests for csv_to_sqlite_upserter.py

Tests:
1. CSV format detection (TradingView vs influx_export)
2. Symbol normalization
3. Timestamp conversion (ms → seconds)
4. Volume column mapping (Volume BTC vs v)
5. Parsing both CSV formats
6. End-to-end upsert to database

Usage:
    python test_csv_upserter.py
"""

import sys
import asyncio
import tempfile
import sqlite3
from pathlib import Path

# Import functions to test
from csv_to_sqlite_upserter import (
    detect_csv_format,
    normalize_symbol,
    parse_tradingview_csv,
    parse_influx_export_csv,
    upsert_csv_to_sqlite
)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas required for tests")
    sys.exit(1)


# ========== Test Data ==========

TRADINGVIEW_CSV_CONTENT = """Unix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount
1760396400000,2025-10-13 23:00:00,BTCUSDT,115507.2,115507.2,115005.44,115166.0,554.61822,63929309.6375022,122790
1760392800000,2025-10-13 22:00:00,BTCUSDT,115700.0,115835.04,115442.67,115507.19,332.95022,38505391.8936591,97554
1760389200000,2025-10-13 21:00:00,BTCUSDT,115713.37,115786.04,115553.33,115700.0,221.7783,25649513.2873119,77744
"""

TRADINGVIEW_LOWER_CSV_CONTENT = """unix,date,symbol,open,high,low,close,Volume BTC,Volume USDT,tradecount
1760396400000,2025-10-13 23:00:00,BTCUSDT,115507.2,115507.2,115005.44,115166.0,554.61822,63929309.6375022,122790
1760392800000,2025-10-13 22:00:00,BTCUSDT,115700.0,115835.04,115442.67,115507.19,332.95022,38505391.8936591,97554
"""

INFLUX_EXPORT_CSV_CONTENT = """symbol,timeframe,ts,o,h,l,c,v,source
BTC/USDT,1m,1758638100000,112680.0,112680.0,112616.93,112661.93,40.79515,binance_ws
BTC/USDT,1m,1758638160000,112661.94,112678.54,112616.93,112664.0,23.32069,binance_ws
BTC/USDT,1m,1758638220000,112664.0,112764.76,112664.0,112764.75,20.3482,binance_ws
"""

# ========== Test Functions ==========

def test_detect_csv_format():
    """Test CSV format detection"""
    print("=" * 60)
    print("TEST: CSV Format Detection")
    print("=" * 60)

    # Test TradingView format
    df_tv = pd.read_csv(pd.StringIO(TRADINGVIEW_CSV_CONTENT))
    fmt_tv = detect_csv_format(df_tv)
    assert fmt_tv == 'tradingview', f"Expected 'tradingview', got '{fmt_tv}'"
    print("✅ TradingView format detected correctly")

    # Test TradingView lowercase
    df_tvl = pd.read_csv(pd.StringIO(TRADINGVIEW_LOWER_CSV_CONTENT))
    fmt_tvl = detect_csv_format(df_tvl)
    assert fmt_tvl == 'tradingview_lower', f"Expected 'tradingview_lower', got '{fmt_tvl}'"
    print("✅ TradingView lowercase format detected correctly")

    # Test influx_export format
    df_ie = pd.read_csv(pd.StringIO(INFLUX_EXPORT_CSV_CONTENT))
    fmt_ie = detect_csv_format(df_ie)
    assert fmt_ie == 'influx_export', f"Expected 'influx_export', got '{fmt_ie}'"
    print("✅ influx_export format detected correctly")

    # Test unknown format
    df_unknown = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    fmt_unknown = detect_csv_format(df_unknown)
    assert fmt_unknown == 'unknown', f"Expected 'unknown', got '{fmt_unknown}'"
    print("✅ Unknown format detected correctly")

    print()


def test_normalize_symbol():
    """Test symbol normalization"""
    print("=" * 60)
    print("TEST: Symbol Normalization")
    print("=" * 60)

    tests = [
        ('BTC/USDT', 'BTCUSDT'),
        ('btc/usdt', 'BTCUSDT'),
        ('BTCUSDT', 'BTCUSDT'),
        ('ETH/USDT', 'ETHUSDT'),
        ('  BTC/USDT  ', 'BTCUSDT'),  # with whitespace
    ]

    for input_sym, expected in tests:
        result = normalize_symbol(input_sym, remove_slash=True)
        assert result == expected, f"normalize_symbol('{input_sym}') = '{result}', expected '{expected}'"
        print(f"✅ '{input_sym}' → '{result}'")

    # Test without slash removal
    result_no_remove = normalize_symbol('BTC/USDT', remove_slash=False)
    assert result_no_remove == 'BTC/USDT', f"Expected 'BTC/USDT', got '{result_no_remove}'"
    print(f"✅ 'BTC/USDT' (no removal) → '{result_no_remove}'")

    print()


def test_parse_tradingview_csv():
    """Test TradingView CSV parsing"""
    print("=" * 60)
    print("TEST: TradingView CSV Parsing")
    print("=" * 60)

    df = pd.read_csv(pd.StringIO(TRADINGVIEW_CSV_CONTENT))
    rows = parse_tradingview_csv(df, timeframe='1h', source='test_source', normalize_symbols=True)

    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
    print(f"✅ Parsed {len(rows)} rows")

    # Check first row
    row0 = rows[0]
    assert row0['symbol'] == 'BTCUSDT', f"Expected 'BTCUSDT', got '{row0['symbol']}'"
    assert row0['timeframe'] == '1h', f"Expected '1h', got '{row0['timeframe']}'"
    assert row0['source'] == 'test_source', f"Expected 'test_source', got '{row0['source']}'"

    # Check timestamp conversion (ms → seconds)
    expected_ts = 1760396400000 // 1000
    assert row0['ts'] == expected_ts, f"Expected {expected_ts}, got {row0['ts']}"
    print(f"✅ Timestamp converted: 1760396400000 ms → {row0['ts']} seconds")

    # Check OHLCV values
    assert row0['o'] == 115507.2
    assert row0['h'] == 115507.2
    assert row0['l'] == 115005.44
    assert row0['c'] == 115166.0
    assert row0['v'] == 554.61822  # Volume BTC
    print(f"✅ OHLCV values correct (O={row0['o']}, H={row0['h']}, L={row0['l']}, C={row0['c']}, V={row0['v']})")

    # Check volume is from "Volume BTC" column, not "Volume USDT"
    # Volume BTC = 554.61822, Volume USDT = 63929309.6375022
    assert row0['v'] == 554.61822, "Should use Volume BTC, not Volume USDT"
    print(f"✅ Volume BTC used (not Volume USDT)")

    print()


def test_parse_influx_export_csv():
    """Test influx_export CSV parsing"""
    print("=" * 60)
    print("TEST: influx_export CSV Parsing")
    print("=" * 60)

    df = pd.read_csv(pd.StringIO(INFLUX_EXPORT_CSV_CONTENT))
    rows = parse_influx_export_csv(df, normalize_symbols=True, override_source=None)

    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
    print(f"✅ Parsed {len(rows)} rows")

    # Check first row
    row0 = rows[0]
    assert row0['symbol'] == 'BTCUSDT', f"Expected 'BTCUSDT', got '{row0['symbol']}' (should be normalized)"
    assert row0['timeframe'] == '1m', f"Expected '1m', got '{row0['timeframe']}'"
    assert row0['source'] == 'binance_ws', f"Expected 'binance_ws', got '{row0['source']}'"

    # Check timestamp conversion (ms → seconds)
    expected_ts = 1758638100000 // 1000
    assert row0['ts'] == expected_ts, f"Expected {expected_ts}, got {row0['ts']}"
    print(f"✅ Timestamp converted: 1758638100000 ms → {row0['ts']} seconds")

    # Check OHLCV values
    assert row0['o'] == 112680.0
    assert row0['h'] == 112680.0
    assert row0['l'] == 112616.93
    assert row0['c'] == 112661.93
    assert row0['v'] == 40.79515  # Base volume (BTC)
    print(f"✅ OHLCV values correct (O={row0['o']}, H={row0['h']}, L={row0['l']}, C={row0['c']}, V={row0['v']})")

    # Test without symbol normalization
    rows_no_norm = parse_influx_export_csv(df, normalize_symbols=False)
    assert rows_no_norm[0]['symbol'] == 'BTC/USDT', "Should preserve 'BTC/USDT' when normalization disabled"
    print(f"✅ Symbol normalization optional (preserved 'BTC/USDT' when disabled)")

    print()


async def test_end_to_end_upsert():
    """Test end-to-end upsert to database"""
    print("=" * 60)
    print("TEST: End-to-End Upsert")
    print("=" * 60)

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create temporary CSV file (TradingView format)
        csv_path = Path(tmpdir) / "test.csv"
        csv_path.write_text(TRADINGVIEW_CSV_CONTENT)

        print(f"Created temp CSV: {csv_path}")
        print(f"Created temp DB: {db_path}")

        # Upsert CSV to database
        count = await upsert_csv_to_sqlite(
            csv_path=str(csv_path),
            db_path=str(db_path),
            timeframe='1h',
            source='test',
            normalize_symbols=True,
            batch_size=500,
            verbose=False
        )

        assert count == 3, f"Expected 3 rows upserted, got {count}"
        print(f"✅ Upserted {count} rows")

        # Verify database contents
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv'")
        tables = cursor.fetchall()
        assert len(tables) == 1, "ohlcv table should exist"
        print("✅ ohlcv table created")

        # Check row count
        cursor.execute("SELECT COUNT(*) FROM ohlcv")
        row_count = cursor.fetchone()[0]
        assert row_count == 3, f"Expected 3 rows in DB, got {row_count}"
        print(f"✅ Database contains {row_count} rows")

        # Check first row values
        cursor.execute("SELECT symbol, timeframe, ts, open, high, low, close, volume, source FROM ohlcv ORDER BY ts DESC LIMIT 1")
        row = cursor.fetchone()

        expected_symbol = 'BTCUSDT'
        expected_timeframe = '1h'
        expected_ts = 1760396400000 // 1000
        expected_source = 'test'

        assert row[0] == expected_symbol, f"Expected symbol '{expected_symbol}', got '{row[0]}'"
        assert row[1] == expected_timeframe, f"Expected timeframe '{expected_timeframe}', got '{row[1]}'"
        assert row[2] == expected_ts, f"Expected ts {expected_ts}, got {row[2]}"
        assert row[3] == 115507.2, f"Expected open 115507.2, got {row[3]}"
        assert row[4] == 115507.2, f"Expected high 115507.2, got {row[4]}"
        assert row[5] == 115005.44, f"Expected low 115005.44, got {row[5]}"
        assert row[6] == 115166.0, f"Expected close 115166.0, got {row[6]}"
        assert row[7] == 554.61822, f"Expected volume 554.61822, got {row[7]}"
        assert row[8] == expected_source, f"Expected source '{expected_source}', got '{row[8]}'"

        print(f"✅ Row values correct: {row}")

        # Check unique index exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_ohlcv_unique'")
        indices = cursor.fetchall()
        assert len(indices) == 1, "idx_ohlcv_unique index should exist"
        print("✅ Unique index created")

        # Test upsert behavior (insert duplicate with different values)
        csv_duplicate = """Unix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount
1760396400000,2025-10-13 23:00:00,BTCUSDT,999999.0,999999.0,999999.0,999999.0,999.0,999.0,999
"""
        csv_dup_path = Path(tmpdir) / "duplicate.csv"
        csv_dup_path.write_text(csv_duplicate)

        count_dup = await upsert_csv_to_sqlite(
            csv_path=str(csv_dup_path),
            db_path=str(db_path),
            timeframe='1h',
            source='test',
            normalize_symbols=True,
            batch_size=500,
            verbose=False
        )

        # Should still be 3 rows (upsert, not insert)
        cursor.execute("SELECT COUNT(*) FROM ohlcv")
        row_count_after = cursor.fetchone()[0]
        assert row_count_after == 3, f"Expected 3 rows after upsert (no duplicates), got {row_count_after}"
        print(f"✅ Upsert prevented duplicate (still {row_count_after} rows)")

        # Check that values were updated
        cursor.execute("SELECT close, volume FROM ohlcv WHERE ts = ?", (expected_ts,))
        updated_row = cursor.fetchone()
        assert updated_row[0] == 999999.0, f"Expected close updated to 999999.0, got {updated_row[0]}"
        assert updated_row[1] == 999.0, f"Expected volume updated to 999.0, got {updated_row[1]}"
        print(f"✅ Upsert updated values: close={updated_row[0]}, volume={updated_row[1]}")

        conn.close()

    print()


async def test_influx_export_end_to_end():
    """Test influx_export format end-to-end"""
    print("=" * 60)
    print("TEST: influx_export End-to-End")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_influx.db"
        csv_path = Path(tmpdir) / "influx.csv"
        csv_path.write_text(INFLUX_EXPORT_CSV_CONTENT)

        # Upsert (no timeframe needed for influx_export)
        count = await upsert_csv_to_sqlite(
            csv_path=str(csv_path),
            db_path=str(db_path),
            normalize_symbols=True,
            verbose=False
        )

        assert count == 3, f"Expected 3 rows, got {count}"
        print(f"✅ Upserted {count} rows")

        # Verify
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT symbol, timeframe, source FROM ohlcv LIMIT 1")
        row = cursor.fetchone()

        assert row[0] == 'BTCUSDT', f"Expected 'BTCUSDT', got '{row[0]}'"
        assert row[1] == '1m', f"Expected '1m', got '{row[1]}'"
        assert row[2] == 'binance_ws', f"Expected 'binance_ws', got '{row[2]}'"
        print(f"✅ influx_export format parsed correctly: symbol={row[0]}, timeframe={row[1]}, source={row[2]}")

        conn.close()

    print()


# ========== Run All Tests ==========

async def run_all_tests():
    """Run all test functions"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "CSV UPSERTER TESTS" + " " * 25 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Sync tests
    test_detect_csv_format()
    test_normalize_symbol()
    test_parse_tradingview_csv()
    test_parse_influx_export_csv()

    # Async tests
    await test_end_to_end_upsert()
    await test_influx_export_end_to_end()

    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("The csv_to_sqlite_upserter.py script is working correctly.")
    print()


def main():
    try:
        asyncio.run(run_all_tests())
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
