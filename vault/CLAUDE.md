# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency trading system with ML-based price prediction using LSTM models. The system consists of:
- **Data ingestion pipeline**: Kafka-based streaming (WebSocket → Kafka → SQLite/InfluxDB)
- **Feature engineering**: Technical indicators including Fibonacci levels, RSI, ATR, Bollinger Bands
- **ML training**: PyTorch LSTM models for price forecasting
- **Trading execution**: Paper and live trading via CCXT
- **GUI**: Tkinter-based interface for model training, monitoring, and trading

## Core Architecture

### Data Flow
```
Binance WebSocket → Kafka (topic: prices) → Consumer → SQLite (primary + replica)
                                           ↳ Optional: InfluxDB
```

The data manager uses:
- `websocket_to_kafka.py`: Async producer streaming Binance kline data to Kafka
- `kafka_consumer_sqlite.py`: Async consumer with batching, CSV export, and graceful shutdown
- `DataManager` class: Async SQLite operations with upsert support and optional replication

### Model Pipeline
1. **Data preparation** (`prepare_dataset.py`): Load from SQLite, create features, build sequences
2. **Feature engineering** (`fiboevo.py`): `add_technical_features()` generates 50+ indicators
3. **Training** (`fiboevo.py`): LSTM2Head model (dual-output: return prediction + volatility)
4. **Inference** (`trading_daemon.py`): Background daemon for continuous prediction and trading

### Key Components

**fiboevo.py**
- Main ML module with LSTM model definition (`LSTM2Head`)
- `add_technical_features()`: Creates deterministic feature set with Fibonacci retracements/extensions
- `create_sequences_from_df()`: Builds (seq_len, features) sequences for LSTM input
- `prepare_input_for_model()`: Pipeline ensuring correct scaling/preprocessing
- `ensure_scaler_feature_names()`: Fallback to assign feature_names_in_ from meta

**trading_daemon.py**
- Background daemon for automated trading
- Loads data from SQLite, computes features, scales, predicts, executes trades
- Thread-safe with locks for config and artifacts
- Supports paper trading (simulation) and live trading (via CCXT)
- Writes to ledger CSV for trade history

**TradeApp.py**
- Tkinter GUI ("Champiguru") with tabs: Config, Prepare, Train, Status, Forecast
- Integrates with `trading_daemon.py` for live trading
- Supports WebSocket streaming, order book visualization, and trade monitoring
- Uses threading for non-blocking operations

**data_manager/**
- `manager.py`: DataManager class with async SQLite operations
- `insert_ohlcv()`: Conservative INSERT OR IGNORE (no updates)
- `upsert_ohlcv()`: INSERT ON CONFLICT DO UPDATE for real-time streams
- Supports replica DB and optional InfluxDB push

**config_manager.py**
- Centralized JSON config with atomic writes and backups
- Supports environment variable placeholders: `${ENV:VAR_NAME}`
- `merge_config()`: Deep merge for CLI/GUI overrides

## Common Development Commands

### Environment Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Data Pipeline (Kafka-based)
```bash
# Start Kafka (Docker Compose in data_manager/)
cd data_manager
docker-compose up -d

# Run producer (WebSocket → Kafka)
python -m data_manager.data_sources.websocket_to_kafka --symbol btcusdt --timeframe 1m --topic prices

# Run consumer (Kafka → SQLite)
python -m data_manager.data_sources.kafka_consumer_sqlite --topic prices --group sqlite_saver --db data_manager/exports/marketdata.db
```

### Model Training
```bash
# Prepare dataset from SQLite
python prepare_dataset.py

# Train model (via fiboevo directly or through GUI)
python fiboevo.py --mode train --sqlite data_manager/exports/marketdata_base.db --symbol BTCUSDT --timeframe 30m --epochs 20 --seq-len 32 --horizon 10

# Artifacts saved to artifacts/: model_best.pt, scaler.pkl, meta.json
```

### Trading Daemon
```bash
# Paper trading (simulation)
python trading_daemon.py --sqlite data_manager/exports/marketdata_base.db --table ohlcv --symbol BTCUSDT --timeframe 30m --paper --seq_len 32

# Live trading requires exchange credentials (set in GUI or config)
```

### GUI
```bash
# Launch main GUI
python TradeApp.py
```

## Important Patterns

### Feature Column Consistency
**Critical**: Feature columns must match between training and inference. The system uses `meta.json` to store `feature_cols` during training. The daemon prefers `model_meta['feature_cols']` if available.

- Training: `fiboevo.add_technical_features()` creates features in deterministic order
- Inference: `trading_daemon.iteration_once()` uses same feature creation, then filters to `meta['feature_cols']`
- If >50% of expected features are missing, the daemon aborts the iteration

### Model Artifacts Structure
```
artifacts/
├── model_best.pt          # PyTorch checkpoint (state_dict or full model)
├── scaler.pkl             # scikit-learn StandardScaler
├── meta.json              # Model metadata (feature_cols, input_size, hidden, num_layers, horizon)
└── ledger.csv             # Trade execution log
```

**Loading models**: `trading_daemon.load_model_and_scaler()` handles various checkpoint formats:
- Extracts state_dict from nested dicts (`state`, `model_state_dict`, `state_dict`)
- Removes `module.` prefix from DataParallel models
- Infers architecture from weight shapes if meta incomplete
- Combines checkpoint meta with meta.json (file takes precedence)

### Scaler Handling
The system uses StandardScaler with feature names. Key utilities:
- `fiboevo.ensure_scaler_feature_names(scaler, meta)`: Assigns `feature_names_in_` if missing
- `fiboevo.prepare_input_for_model(X, scaler, feature_cols)`: Ensures correct scaling without double-scaling

### NaN Strategy
DataManager supports `nan_strategy`:
- `'drop'`: Skip rows with NaN values (default)
- Alternative: allow None values in DB

Feature engineering uses `.dropna()` after indicator calculation to ensure clean sequences.

### Threading and Concurrency
- **GUI**: Non-blocking via `threading.Thread` for long operations (prepare, train, daemon)
- **Data manager**: Async/await (aiosqlite, aiokafka) for I/O operations
- **Daemon**: Thread-safe with `threading.Lock` for config/artifact updates

## Configuration Files

- `config/gui_config.json`: GUI settings (persisted on close)
- `artifacts/daemon_cfg.json`: Trading daemon config (optional, constructor args take precedence)
- `config/influx.json`: InfluxDB connection settings

## Database Schema

SQLite table `ohlcv`:
```sql
CREATE TABLE ohlcv (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    source TEXT,
    UNIQUE(symbol, timeframe, ts)  -- via idx_ohlcv_unique
);
```

## Timeframe Conventions
- Format: `{number}{unit}` where unit is `s` (seconds), `m` (minutes), `h` (hours), `d` (days)
- Examples: `1m`, `30m`, `1h`, `4h`, `1d`
- Conversion: `timeframe_to_seconds()` in TradeApp.py

## Dependencies

**Core**:
- pandas, numpy
- ccxt (exchange API)
- joblib (model serialization)

**Optional**:
- torch (LSTM models)
- scikit-learn (StandardScaler)
- deap (genetic algorithm optimization)
- influxdb-client
- websocket-client (GUI WebSocket streaming)
- aiokafka, aiosqlite (data pipeline)

## Testing and Validation

**Smoke test trading daemon**:
```bash
python trading_daemon.py --sqlite data_manager/exports/marketdata_base.db --table ohlcv --symbol BTCUSDT --timeframe 30m --paper --seq_len 32 --poll 2.0
```

**Check environment**:
```bash
python check_env.py
```

## Critical: Recent fiboevo.py Updates (Breaking Changes)

### API Changes Requiring Updates

**1. load_model() now returns tuple:**
```python
# OLD:
model = fiboevo.load_model(path)

# NEW:
model, meta = fiboevo.load_model(path)
```

**Impact**: All callers in `trading_daemon.py`, `TradeApp.py`, and any scripts must update.

**2. New prepare_input_for_model() function:**
```python
x_tensor = fiboevo.prepare_input_for_model(
    df,               # DataFrame with features
    feature_cols,     # List of feature column names
    seq_len,          # Sequence length
    scaler=scaler,    # StandardScaler (optional)
    method="per_row"  # scaling method
)
```

**Purpose**: Prevents double-scaling, enforces feature column order, validates alignment with scaler.feature_names_in_

**Required in**: `trading_daemon.iteration_once()`, `TradeApp._get_latest_prediction_thread()`, `TradeApp._run_forecast()`

**3. Temporal split to prevent data leakage:**
```python
from fiboevo import temporal_split_indexes

n_train_seq, n_val_seq, n_test_seq, train_rows_end = temporal_split_indexes(
    n_total=len(df),
    seq_len=seq_len,
    horizon=horizon,
    val_frac=0.2,
    test_frac=0.1
)

# Fit scaler ONLY on train rows (critical!)
scaler.fit(df[feature_cols].iloc[:train_rows_end + 1])
```

**Required in**: `prepare_dataset.py`, any training scripts

### Data Leakage Prevention

**Known Issues Fixed:**
1. ✅ Scaler fitted on full dataset - now must fit only on train split
2. ✅ Feature column order mismatch - now validated by `prepare_input_for_model()`
3. ✅ Silent misalignment - now raises `RuntimeError` with clear message

**Remaining Checks:**
- Run GUI Audit tab after loading data to check for:
  - Suspicious feature names ("shift", "lead", "next", "future")
  - High correlation (>0.95) with future close
  - Exact matches to shifted future values
  - Scaler statistics suggesting full-dataset fit

**Audit Location**: TradeApp.py → Audit tab → "Run Audit" button

### Integration Checklist

See `ANALYSIS.md` for detailed migration guide. Priority order:

1. **trading_daemon.py**:
   - Update `load_model_and_scaler()` to unpack tuple
   - Replace manual sequence scaling with `prepare_input_for_model()` in `iteration_once()`
   - Store `model_meta` in daemon state

2. **TradeApp.py**:
   - Update `_load_model_from_artifacts()`
   - Update `_load_model_file_background()`
   - Update `_get_latest_prediction_thread()`
   - Update `_run_forecast()`

3. **prepare_dataset.py**:
   - Use `temporal_split_indexes()` before fitting scaler
   - Save complete metadata with model
   - Set `scaler.feature_names_in_` when saving

### Sliding Window Implementation

**Current Status**: Fixed-length sliding window is implemented in `create_sequences_from_df()`:
- Takes last `seq_len` rows to create input sequence
- No overlapping considerations (each prediction uses independent window)
- Window slides by full `horizon` steps in backtest (not step-by-step)

**Gap Management**:
- **No explicit gap handling** - assumes continuous data
- Missing timestamps in SQLite are not detected/filled
- Risk: If timeframe data has gaps (e.g., missing candles), sequences may span non-consecutive periods

**Recommendation for gap handling**:
```python
# In prepare_dataset.py or daemon, after loading:
df = df.sort_values('ts').reset_index(drop=True)

# Check for gaps
expected_delta = timeframe_to_seconds(timeframe)
df['ts_diff'] = df['ts'].diff()
gaps = df[df['ts_diff'] > expected_delta * 1.5]

if len(gaps) > 0:
    print(f"Warning: {len(gaps)} gaps detected in data")
    # Option 1: Forward-fill
    # df = df.asfreq(f'{expected_delta}s', method='ffill')
    # Option 2: Drop sequences spanning gaps
    # (requires gap-aware create_sequences)
```

**Current behavior**: Sequences spanning gaps are created without warning, which may degrade model performance.

## Notes

- **Windows-specific**: This repo is developed on Windows (paths use backslashes internally but pathlib normalizes)
- **Git**: Not currently a git repo (as of this snapshot)
- **Kafka**: Docker Compose config in `data_manager/` for local development
- **Secrets**: Never commit API keys. Use environment variables or `config_manager.py` placeholders: `${ENV:BINANCE_KEY}`
- **Log files**: GUI creates logs in `logs/` with rotation
- **Breaking changes**: See ANALYSIS.md for detailed migration guide from old fiboevo.py API
- to memorize
- to memorize