# CSV to SQLite Upserter Guide

## Overview

`csv_to_sqlite_upserter.py` is a universal tool for importing OHLCV data from CSV files into your SQLite database with automatic format detection and standardization.

## Supported CSV Formats

### Format 1: TradingView Exports

**Columns** (case-insensitive):
```csv
Unix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount
1760396400000,2025-10-13 23:00:00,BTCUSDT,115507.2,115507.2,115005.44,115166.0,554.61822,63929309.6375022,122790
```

**Characteristics**:
- ✅ Timestamp in milliseconds (`Unix` column)
- ✅ Human-readable date (ignored, derived from timestamp)
- ✅ **Uses Volume BTC** (base volume) — consistent with your system
- ❌ No timeframe column → **must provide via `--timeframe` argument**
- ❌ No source column → defaults to `csv_tradingview`

### Format 2: influx_to_sql.py Exports

**Columns**:
```csv
symbol,timeframe,ts,o,h,l,c,v,source
BTC/USDT,1m,1758638100000,112680.0,112680.0,112616.93,112661.93,40.79515,binance_ws
```

**Characteristics**:
- ✅ Timestamp in milliseconds (`ts` column)
- ✅ Timeframe included (`1m`, `30m`, `1h`, etc.)
- ✅ Source tag included (`binance_ws`, `influx`, etc.)
- ✅ **Uses base volume** (`v` column) — consistent with your system
- ✅ Auto-detected, no additional arguments required

---

## Auto-Detection Logic

The script automatically detects the format based on column names:

| Format | Detection Criteria |
|--------|-------------------|
| **influx_export** | Has columns: `ts`, `o`, `h`, `l`, `c`, `v` |
| **tradingview** | Has columns: `Unix`, `Open`, `High`, `Low`, `Close` |
| **tradingview_lower** | Has columns: `unix`, `open`, `high`, `low`, `close` (case variant) |

If the format is unrecognized, the script will exit with an error listing the columns found.

---

## Data Standardization

### 1. Timestamp Conversion
- **Input**: Milliseconds (both formats)
- **Output**: Seconds (SQLite standard)
- **Conversion**: `ts_seconds = ts_milliseconds // 1000`

### 2. Symbol Normalization

**Default behavior** (`--normalize-symbol` flag):
- `BTC/USDT` → `BTCUSDT`
- `ETH/USDT` → `ETHUSDT`
- Removes `/` separator
- Converts to uppercase

**Without flag**:
- Preserves original format

### 3. Volume Mapping

**Critical**: Your system uses **base volume (Volume BTC)**, not quote volume.

| Format | Volume Column | Notes |
|--------|--------------|-------|
| TradingView | `Volume BTC` (column 7) | Base volume ✅ |
| influx_export | `v` | Base volume from Binance WS ✅ |

**Not used**: `Volume USDT` (quote volume) from TradingView exports

### 4. Upsert Behavior

Uses `DataManager.upsert_ohlcv()` with conflict resolution on `(symbol, timeframe, ts)`:
- **Existing row**: Updates OHLC values
- **New row**: Inserts
- **Atomic**: Uses SQLite `ON CONFLICT ... DO UPDATE`

---

## Usage Examples

### Basic Usage

#### Format 2: influx_export (auto-complete)
```bash
python csv_to_sqlite_upserter.py \
    --csv influx_export.csv \
    --db data_manager/exports/marketdata_base.db
```

**Output**:
```
[INFO] Loading CSV: influx_export.csv
[INFO] Detected format: influx_export
[INFO] Parsed 15234 valid rows
[INFO] Timestamp range: 2024-01-15 10:00:00 → 2024-01-20 18:30:00
[INFO] Symbols: BTCUSDT
[INFO] Timeframes: 1m
[INFO] Upserting 15234 rows for BTCUSDT 1m...
[SUCCESS] ✅ Upserted 15234 rows to marketdata_base.db
```

#### Format 1: TradingView (requires --timeframe)
```bash
python csv_to_sqlite_upserter.py \
    --csv Binance_BTCUSDT_1h.csv \
    --db data_manager/exports/marketdata_base.db \
    --timeframe 1h
```

**Error if missing timeframe**:
```
ValueError: TradingView format requires --timeframe argument
```

### Advanced Usage

#### Multiple CSV Files
```bash
python csv_to_sqlite_upserter.py \
    --csv Binance_BTCUSDT_2020_minute.csv \
          Binance_BTCUSDT_2024_minute.csv \
          Binance_BTCUSDT_2025_minute.csv \
    --db data_manager/exports/marketdata_base.db \
    --timeframe 1m
```

**Output**:
```
[INFO] Loading CSV: Binance_BTCUSDT_2020_minute.csv
[INFO] Detected format: tradingview_lower
[INFO] Parsed 525600 valid rows
...
[SUCCESS] ✅ Upserted 525600 rows

[INFO] Loading CSV: Binance_BTCUSDT_2024_minute.csv
...

[SUMMARY] Total rows upserted across all files: 1576800
```

#### With Symbol Normalization
```bash
python csv_to_sqlite_upserter.py \
    --csv influx_export.csv \
    --db marketdata.db \
    --normalize-symbol
```

**Effect**: `BTC/USDT` → `BTCUSDT` in database

#### Custom Source Tag
```bash
python csv_to_sqlite_upserter.py \
    --csv historical_data.csv \
    --db marketdata.db \
    --timeframe 1h \
    --source historical_binance_2020
```

**Result**: All rows tagged with `source='historical_binance_2020'`

#### Batch Size Tuning
```bash
# Smaller batches (better for memory-constrained systems)
python csv_to_sqlite_upserter.py \
    --csv large_file.csv \
    --db marketdata.db \
    --batch-size 100

# Larger batches (faster for high-performance systems)
python csv_to_sqlite_upserter.py \
    --csv large_file.csv \
    --db marketdata.db \
    --batch-size 1000
```

#### Quiet Mode
```bash
python csv_to_sqlite_upserter.py \
    --csv data.csv \
    --db marketdata.db \
    --timeframe 1h \
    --quiet
```

**Output**: Only errors printed, no progress messages

---

## Python API

### Function Signature

```python
async def upsert_csv_to_sqlite(
    csv_path: str,
    db_path: str,
    timeframe: Optional[str] = None,
    source: Optional[str] = None,
    normalize_symbols: bool = True,
    batch_size: int = 500,
    verbose: bool = True
) -> int:
    """
    Load CSV, detect format, parse, and upsert to SQLite.

    Args:
        csv_path: Path to CSV file
        db_path: Path to SQLite database
        timeframe: Timeframe (required for TradingView, optional for influx_export)
        source: Source tag (optional, defaults based on format)
        normalize_symbols: If True, normalize symbols (BTC/USDT → BTCUSDT)
        batch_size: Number of rows per upsert batch
        verbose: Print progress messages

    Returns:
        Number of rows upserted
    """
```

### Example Usage

```python
import asyncio
from csv_to_sqlite_upserter import upsert_csv_to_sqlite

async def import_historical_data():
    # Import multiple files
    files = [
        'Binance_BTCUSDT_2020_minute.csv',
        'Binance_BTCUSDT_2021_minute.csv',
        'Binance_BTCUSDT_2022_minute.csv'
    ]

    total = 0
    for csv_file in files:
        count = await upsert_csv_to_sqlite(
            csv_path=csv_file,
            db_path='data_manager/exports/marketdata_base.db',
            timeframe='1m',
            source='historical_import',
            normalize_symbols=True,
            batch_size=500
        )
        total += count
        print(f"Imported {count} rows from {csv_file}")

    print(f"Total: {total} rows")

# Run
asyncio.run(import_historical_data())
```

---

## Error Handling

### Unknown Format
```
ValueError: Unknown CSV format. Expected TradingView or influx_export format.
Columns found: ['Date', 'Price', 'Volume']
```

**Solution**: Ensure CSV has required columns for one of the supported formats.

### Missing Timeframe
```
ValueError: TradingView format requires --timeframe argument
```

**Solution**: Add `--timeframe` argument:
```bash
python csv_to_sqlite_upserter.py --csv data.csv --db marketdata.db --timeframe 1h
```

### Missing Columns
```
ValueError: Missing required columns in TradingView CSV: ['volume btc']
```

**Solution**: Check CSV has all required columns. For TradingView: `Unix, Symbol, Open, High, Low, Close, Volume BTC`

### Malformed Rows
```
Warning: Skipping malformed row: could not convert string to float: 'N/A'
```

**Effect**: Malformed rows are skipped, script continues. Check CSV for data quality issues.

### File Not Found
```
FileNotFoundError: CSV file not found: data.csv
```

**Solution**: Verify file path is correct.

---

## Performance

### Benchmarks (Approximate)

| CSV Size | Rows | Time | Memory |
|----------|------|------|--------|
| 10 MB | 50,000 | ~2s | ~50 MB |
| 100 MB | 500,000 | ~20s | ~200 MB |
| 1 GB | 5,000,000 | ~3m | ~1 GB |

**Notes**:
- Uses pandas for CSV parsing (efficient for large files)
- Batch upserts minimize database locks
- Async I/O prevents blocking

### Optimization Tips

1. **Increase batch size** for faster imports (if memory allows):
   ```bash
   --batch-size 1000
   ```

2. **Disable normalization** if symbols already standardized:
   ```bash
   # No --normalize-symbol flag
   ```

3. **Use quiet mode** for automated scripts:
   ```bash
   --quiet
   ```

4. **Process multiple files in parallel** (Python API):
   ```python
   tasks = [upsert_csv_to_sqlite(f, db) for f in files]
   await asyncio.gather(*tasks)
   ```

---

## Integration with Existing Workflow

### 1. Historical Data Import (One-time)

```bash
# Import all historical TradingView exports
python csv_to_sqlite_upserter.py \
    --csv data_manager/exports/binance\ btcusdt/*.csv \
    --db data_manager/exports/marketdata_base.db \
    --timeframe 1m \
    --normalize-symbol \
    --source historical_tradingview
```

### 2. Influx Export Sync (Periodic)

```bash
# Export from InfluxDB to CSV (using influx_to_sql.py export feature)
# Then import to SQLite
python csv_to_sqlite_upserter.py \
    --csv influx_export_btcusdt_1m.csv \
    --db data_manager/exports/marketdata_base.db \
    --normalize-symbol
```

### 3. Automated Pipeline (Cron/Scheduled Task)

```bash
#!/bin/bash
# daily_import.sh

# Download latest CSV from source
wget https://example.com/daily_btcusdt.csv -O /tmp/daily.csv

# Import to SQLite
python csv_to_sqlite_upserter.py \
    --csv /tmp/daily.csv \
    --db data_manager/exports/marketdata_base.db \
    --timeframe 1m \
    --quiet

# Clean up
rm /tmp/daily.csv
```

---

## Database Schema Compatibility

The script writes to the `ohlcv` table with this schema:

```sql
CREATE TABLE ohlcv (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts INTEGER NOT NULL,  -- Unix timestamp (seconds)
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,  -- Base volume (BTC for BTCUSDT)
    source TEXT,
    UNIQUE(symbol, timeframe, ts)  -- Enforced by idx_ohlcv_unique
);
```

**Unique index** (created by DataManager):
```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique
ON ohlcv(symbol, timeframe, ts);
```

This enables upsert behavior (`ON CONFLICT ... DO UPDATE`).

---

## Troubleshooting

### Problem: Duplicate entries after import
**Cause**: Upsert should prevent this. Check if unique index exists:
```sql
SELECT * FROM sqlite_master WHERE type='index' AND name='idx_ohlcv_unique';
```

**Solution**: Recreate index:
```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique ON ohlcv(symbol, timeframe, ts);
```

### Problem: Import is slow
**Cause**: Small batch size or database locks

**Solutions**:
1. Increase batch size: `--batch-size 1000`
2. Ensure no other processes accessing database
3. Use SSD for database file

### Problem: Timestamps are off by 1000x
**Cause**: CSV already in seconds, not milliseconds

**Solution**: Check CSV format. If timestamps are already in seconds, modify script:
```python
# In parse_tradingview_csv() or parse_influx_export_csv()
ts = ts_ms  # Don't divide by 1000
```

### Problem: Symbol not matching in database
**Cause**: Normalization mismatch

**Solutions**:
1. Use `--normalize-symbol` consistently
2. Check existing symbols in DB:
   ```sql
   SELECT DISTINCT symbol FROM ohlcv;
   ```
3. Standardize all imports with same flag

---

## File Location

**Script**: `C:\Users\soyko\Documents\git2\Market\cow_lvl-2oct\csv_to_sqlite_upserter.py`

**Dependencies**:
- pandas
- DataManager (from `data_manager/data_manager/manager.py` or `data_manager/manager.py`)
- aiosqlite (via DataManager)

**Install**:
```bash
pip install pandas aiosqlite
```

---

## Summary

**Key Features**:
✅ Auto-detects CSV format (TradingView vs influx_export)
✅ Uses **base volume (Volume BTC)** consistently
✅ Converts timestamps ms → seconds
✅ Normalizes symbols (optional)
✅ Batch upserts for performance
✅ Handles duplicates gracefully (upsert)
✅ CLI + Python API
✅ Progress reporting and validation

**Quick Start**:
```bash
# TradingView export
python csv_to_sqlite_upserter.py --csv data.csv --db marketdata.db --timeframe 1h

# influx_to_sql export
python csv_to_sqlite_upserter.py --csv data.csv --db marketdata.db
```

**Documentation**:
- This guide: `CSV_UPSERTER_GUIDE.md`
- Script help: `python csv_to_sqlite_upserter.py --help`
- DataManager docs: `data_manager/data_manager/manager.py`
