# BitCorn Farmer - Cryptocurrency Trading System

**LSTM-based price prediction with multi-horizon forecasting and live trading dashboard**

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy pandas matplotlib scikit-learn joblib websocket-client

# 2. Start GUI
python TradeApp.py

# 3. Import data (CSV or WebSocket)
python csv_to_sqlite_upserter.py --csv BTCUSDT_1h.csv --symbol BTCUSDT --timeframe 1h

# 4. Train model
# → Training tab → Select "v2" feature system → "Prepare + Train"

# 5. Run live predictions
# → Status tab → "Start Daemon" → Enable "Multi-Horizon Mode"
```

## Features

✅ **Multi-Horizon Predictions** - Forecast at 1h, 3h, 5h, 10h, 15h, 20h, 30h simultaneously
✅ **Live Dashboard** - Real-time prediction fan with confidence intervals
✅ **Clean Features** - v2 system with 14 stationary features (no price leakage)
✅ **WebSocket Streaming** - Live market data from Binance
✅ **Paper/Live Trading** - Automated execution via CCXT
✅ **Feature Selection** - Toggle between v1 (39 features) and v2 (14 clean)

## Documentation

| Document | Purpose |
|----------|---------|
| **[Getting Started Guide](docs/GETTING_STARTED.md)** | Installation, training, predictions |
| **[Developer Guide](docs/DEVELOPER_GUIDE.md)** | Architecture, code structure, extension points |
| **[Multi-Horizon Dashboard](docs/MULTI_HORIZON_DASHBOARD.md)** | Live dashboard usage |
| **[Multi-Horizon Inference](MULTI_HORIZON_INFERENCE.md)** | Mathematical foundations |
| **[Retraining Summary](RETRAINING_SUMMARY.md)** | v2 feature system guide |
| **[CSV Upserter Guide](CSV_UPSERTER_GUIDE.md)** | Data pipeline |

## Project Structure

```
BitCorn_Farmer/
├── core/                     # Core modules (feature registry)
├── tests/                    # Test files
├── examples/                 # Example scripts
├── outputs/                  # Generated plots & predictions (gitignored)
├── docs/                     # Documentation (consolidated)
├── artifacts/                # Model artifacts (v2 - current)
├── artifacts_deprecated/     # Backup (v1 - legacy)
├── vault/                    # Historical/archived docs
├── data_manager/            # Data ingestion pipeline
├── config/                  # Configuration files
├── TradeApp.py              # Main GUI (~3,700 lines)
├── trading_daemon.py        # Background inference daemon
├── fiboevo.py               # Feature engineering + LSTM model
└── multi_horizon_fan_inference.py  # Multi-horizon predictions
```

## System Components

```
WebSocket/CSV → SQLite → Feature Engineering → LSTM Model → Predictions
                    ↓
               Multi-Horizon Dashboard (Live)
```

### Feature Systems

**v2 (Current) - 14 Clean Features:**
- All stationary (no price-level leakage)
- Log returns, ratios, bounded indicators
- 2.5x faster computation than v1
- **Recommended for production**

**v1 (Legacy) - 39 Features:**
- Includes 26 price-level features
- Fibonacci levels and extensions
- Kept for compatibility with old models

**Switch in GUI:** Training tab → Feature System dropdown

### Model Architecture

**LSTM2Head** - Dual-output model:
- **Output 1:** Log return prediction
- **Output 2:** Volatility (uncertainty)
- Trained on clean v2 features
- Artifacts: `model_best.pt`, `scaler.pkl`, `meta.json`

## Common Commands

```bash
# Import data from CSV
python csv_to_sqlite_upserter.py --csv BTCUSDT_1h.csv --symbol BTCUSDT --timeframe 1h

# Train model (CLI)
python retrain_clean_features.py --epochs 100 --hidden 128

# Run predictions (CLI)
cd examples/ && python example_multi_horizon.py

# Run tests
cd tests/ && python test_integration.py

# Start daemon (CLI)
python trading_daemon.py --sqlite data_manager/exports/marketdata_base.db --paper
```

## Database Schema

```sql
CREATE TABLE ohlcv (
    ts INTEGER,
    timestamp TEXT,
    symbol TEXT,
    timeframe TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (ts, symbol, timeframe)
);
```

## Configuration

- `config/gui_config.json` - GUI settings (auto-saved)
- `artifacts/daemon_cfg.json` - Daemon configuration
- `artifacts/meta.json` - Model metadata (features, hyperparams)

## Development

See **[Developer Guide](docs/DEVELOPER_GUIDE.md)** for:
- Code structure and architecture
- Adding custom features
- Extending the system
- Debugging guide
- Contribution workflow

## Notes

- Developed on Windows 10/11 (cross-platform compatible)
- Supports paper trading (simulation) and live trading (CCXT)
- WebSocket streaming from Binance
- Optional Kafka pipeline for production

---

**Version:** 2.0 (Post-v2 Migration)
**Last Updated:** 2025-10-31
**Python:** 3.10+
