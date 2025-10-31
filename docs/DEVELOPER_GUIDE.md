# BitCorn Farmer - Developer Guide

**Date:** 2025-10-31
**Version:** 2.0 (Post-v2 Migration)
**Author:** Claude (Anthropic)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Code Structure](#code-structure)
3. [Feature Engineering System](#feature-engineering-system)
4. [Multi-Horizon Prediction System](#multi-horizon-prediction-system)
5. [Database Schema](#database-schema)
6. [Extension Points](#extension-points)
7. [Development Workflow](#development-workflow)

---

## 1. Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         TradeApp GUI                         │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────┐ │
│  │ Training │ Backtest │  Status  │  Audit   │ Websocket  │ │
│  └──────────┴──────────┴──────────┴──────────┴────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │      TradingDaemon            │
         │  ┌─────────────────────────┐  │
         │  │  Inference Loop (5s)    │←─┼─ WebSocket Data
         │  │  - Load recent data     │  │
         │  │  - Compute features     │  │
         │  │  - Run predictions      │  │
         │  │  - Update queue         │  │
         │  └─────────────────────────┘  │
         └───────────────┬───────────────┘
                         │
                 ┌───────┴────────┐
                 │                │
                 ↓                ↓
         ┌──────────────┐  ┌─────────────┐
         │   SQLite     │  │  Artifacts  │
         │  (OHLCV)     │  │  - Model    │
         │              │  │  - Scaler   │
         │              │  │  - Meta     │
         └──────────────┘  └─────────────┘
```

### Data Flow

1. **WebSocket** → Real-time market data from Binance
2. **SQLite Database** → Stores OHLCV candlestick data
3. **Feature Engineering** → Computes 14-39 technical indicators
4. **LSTM Model** → Dual-head prediction (log_return + volatility)
5. **GUI Display** → Live dashboard with prediction fan
6. **Trading Logic** → Paper/live trading execution

---

## 2. Code Structure

### Main Files (Project Root)

| File | Purpose | Lines |
|------|---------|-------|
| `TradeApp.py` | Main GUI application | ~3,700 |
| `trading_daemon.py` | Background inference daemon | ~1,050 |
| `fiboevo.py` | Feature engineering + LSTM model | ~650 |
| `multi_horizon_fan_inference.py` | Multi-horizon predictions | ~320 |
| `dashboard_visualizations_simple.py` | Live plotting | ~165 |

### Core Modules

| Module | Purpose |
|--------|---------|
| `core/feature_registry.py` | Feature system selection (v1/v2) |
| `config_manager.py` | Configuration handling |
| `fp_utils.py` | Utility functions |
| `csv_to_sqlite_upserter.py` | Data ingestion pipeline |

### New Directory Structure (Post-Reorganization)

```
BitCorn_Farmer/
├── core/                     # Core modules
│   ├── feature_registry.py   # Feature system registry
│   └── __init__.py
├── tests/                    # All test files
│   ├── test_csv_upserter.py
│   ├── test_integration.py
│   └── ...
├── examples/                 # Example scripts
│   ├── example_multi_horizon.py
│   └── ...
├── outputs/                  # Generated outputs (gitignored)
│   ├── plots/               # PNG/PDF charts
│   └── predictions/         # CSV prediction results
├── docs/                     # Documentation
│   ├── DEVELOPER_GUIDE.md   # This file
│   ├── MULTI_HORIZON_DASHBOARD.md
│   └── GETTING_STARTED.md
├── artifacts/                # Model artifacts (v2)
├── artifacts_deprecated/     # Backup (v1)
├── vault/                    # Historical docs
├── data_manager/            # Data pipeline
└── config/                  # Configuration files
```

---

## 3. Feature Engineering System

### Feature Registry Architecture

The `core/feature_registry.py` module provides a **centralized system** for managing different feature engineering approaches:

```python
from core.feature_registry import FEATURE_REGISTRY

# List available systems
systems = FEATURE_REGISTRY.list_systems()
# {'v1': {'description': '39 features...', 'n_features': 39, 'version': '1.0'},
#  'v2': {'description': '14 clean features...', 'n_features': 14, 'version': '2.0'}}

# Compute features using v2 (default)
df_features = FEATURE_REGISTRY.compute_features(
    close, high, low, volume,
    system_name="v2"
)
```

### v1 vs v2 Feature Systems

| Aspect | v1 (Legacy) | v2 (Current) |
|--------|-------------|--------------|
| **Features** | 39 indicators | 14 clean indicators |
| **Price Levels** | 26 (68%) | 0 (0%) |
| **Data Leakage** | High risk | Low risk |
| **Stationarity** | Mixed | Better |
| **Use Case** | Deprecated | Production |

#### v1 Features (39 total, 26 price-level)

**Stationary (13):**
- `log_ret_1`, `log_ret_5`, `ret_1`, `ret_5`
- `rsi_14`, `raw_vol_10`, `raw_vol_30`
- `td_buy_setup`, `td_sell_setup`
- `ma_diff_20`, `bb_width`, `bb_std`, `atr_14`

**Non-Stationary Price Levels (26):**
- `log_close`, `sma_5/20/50`, `ema_5/20/50`
- `bb_m`, `bb_up`, `bb_dn`
- `fib_r_236/382/500/618/786`
- `dist_fib_r_*` (10 features)
- `fibext_1272/1618/2000`
- `dist_fibext_*` (6 features)

#### v2 Features (14 total, 0 price-level) ✓

**All Stationary:**
- `log_ret_1`, `log_ret_5` (log returns)
- `sma_ratio_5/20/50` (price/SMA ratios)
- `ema_ratio_5/20/50` (price/EMA ratios)
- `bb_position` ((price - bb_lower) / (bb_upper - bb_lower))
- `bb_width` (bb_upper - bb_lower)
- `rsi_14` (0-100 bounded indicator)
- `atr_pct` (ATR / price percentage)
- `raw_vol_10`, `raw_vol_30` (rolling volatility)

### Switching Between Systems

**In GUI (TradeApp.py):**
- Training tab → Feature System dropdown → Select "v1" or "v2"

**In Code:**
```python
# TradeApp.py
self.feature_system_var = StringVar(value="v2")  # GUI control

# trading_daemon.py
self.feature_system = "v2"  # Default system

# Use registry
df_features = FEATURE_REGISTRY.compute_features(
    close, high, low, volume,
    system_name=self.feature_system
)
```

### Adding Custom Feature Systems

1. Define your feature engineering function:
```python
def add_technical_features_custom(close, high, low, volume, dropna_after=True):
    # Your custom logic
    df = pd.DataFrame()
    # ... compute features
    return df
```

2. Register it:
```python
FEATURE_REGISTRY.register_system(
    name="custom",
    function=add_technical_features_custom,
    description="My custom features",
    expected_features=["feat1", "feat2", ...],
    version="1.0"
)
```

3. Select it in GUI or code:
```python
df_features = FEATURE_REGISTRY.compute_features(
    close, high, low, volume,
    system_name="custom"
)
```

---

## 4. Multi-Horizon Prediction System

### Architecture

The system generates predictions at multiple future time horizons (1h, 3h, 5h, 10h, 15h, 20h, 30h) using a **scaling approach** based on Brownian motion assumptions:

**Mathematical Foundation:**

```
Native prediction (trained horizon h_native = 10):
  ŷ_10 = log_return_10h
  σ_10 = volatility_10h

Scaled prediction (target horizon h):
  scale_factor = h / h_native
  ŷ_h = ŷ_10 × scale_factor              (linear drift)
  σ_h = σ_10 × √scale_factor              (Brownian volatility)

Price prediction:
  P_{t+h} = P_t × exp(ŷ_h)

Confidence intervals (95%):
  CI_lower = P_t × exp(ŷ_h - 1.96 × σ_h)
  CI_upper = P_t × exp(ŷ_h + 1.96 × σ_h)
```

### Single-Point Prediction

**Optimized for real-time inference:**

```python
from multi_horizon_fan_inference import predict_single_point_multi_horizon

predictions = predict_single_point_multi_horizon(
    df=df_features,          # DataFrame with features
    model=lstm_model,        # Trained LSTM2Head
    meta=model_meta,         # Model metadata
    scaler=scaler,           # Fitted scaler
    device=torch.device("cpu"),
    horizons=[1, 3, 5, 10, 15, 20, 30],
    method="scaling"
)

# Returns:
# {
#   1: {'price': 106520, 'change_pct': 0.07, 'ci_lower_95': 106400, ...},
#   3: {'price': 106680, 'change_pct': 0.22, ...},
#   ...
# }
```

### TradingDaemon Integration

The daemon automatically generates multi-horizon predictions every 5 seconds when enabled:

```python
# In trading_daemon.py
def iteration_once_multi_horizon(self) -> Dict[int, Dict]:
    # 1. Load recent data
    # 2. Compute features
    # 3. Generate multi-horizon predictions
    predictions = predict_single_point_multi_horizon(...)

    # 4. Push to queue for GUI consumption
    self.predictions_queue.put_nowait(predictions)

    return predictions
```

### GUI Display (Status Tab)

The Status tab polls the daemon's prediction queue and updates:

1. **Prediction Fan Canvas** - Color-coded lines showing each horizon
2. **Confidence Bands** - 95% CI shaded areas
3. **Summary Table** - Tabular view of all predictions

**Enabling multi-horizon mode:**
```python
# In GUI
self.daemon.multi_horizon_mode = True
self.daemon.multi_horizon_horizons = [1, 3, 5, 10, 15, 20, 30]
```

---

## 5. Database Schema

### SQLite Tables

**OHLCV Table:**
```sql
CREATE TABLE ohlcv (
    ts INTEGER,              -- Unix timestamp (seconds)
    timestamp TEXT,          -- ISO datetime string
    symbol TEXT,             -- e.g., "BTCUSDT"
    timeframe TEXT,          -- e.g., "1h", "30m"
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (ts, symbol, timeframe)
);
```

**AggTrade Table (WebSocket):**
```sql
CREATE TABLE aggtrade (
    ts INTEGER,              -- Unix timestamp (milliseconds)
    symbol TEXT,
    price REAL,
    quantity REAL,
    side TEXT,               -- "buy" or "sell"
    PRIMARY KEY (ts, symbol)
);
```

### Data Ingestion Pipeline

```
CSV Files (Binance)
    ↓
csv_to_sqlite_upserter.py
    ↓
marketdata_base.db (SQLite)
    ↓
TradingDaemon._load_recent_rows()
    ↓
Feature Engineering
    ↓
LSTM Predictions
```

---

## 6. Extension Points

### Planned Features (See FUTURE_EXTENSIBILITY_GUIDE.md)

1. **GUI Feature Editor**
   - Visual pipeline builder
   - Drag-and-drop feature blocks
   - Live preview of computed features
   - Export to custom feature system

2. **Val/Train Split Configuration**
   - Rolling window cross-validation
   - Walk-forward validation
   - Custom split strategies
   - GUI controls for split parameters

3. **Multiple Prediction Fans**
   - Compare different models side-by-side
   - Different confidence levels (68%, 80%, 95%, 99%)
   - Ensemble predictions
   - Model A/B testing

4. **Rolling Window Configuration**
   - Adjustable sequence length (seq_len)
   - Custom prediction horizon
   - Stride settings for training

### Implementation Guide

For adding new features:

1. **Feature Engineering:**
   - Add function to `fiboevo.py`
   - Register in `core/feature_registry.py`
   - Update GUI dropdown in `TradeApp.py`

2. **Model Architecture:**
   - Modify `LSTM2Head` class in `fiboevo.py`
   - Update training loop in `TradeApp._start_train_model()`
   - Test with `examples/example_multi_horizon.py`

3. **GUI Components:**
   - Add tab in `TradeApp._build_ui()`
   - Create worker thread for background tasks
   - Use queue for thread-safe communication

4. **Data Sources:**
   - Add new table schema to `data_manager/scripts/`
   - Update `csv_to_sqlite_upserter.py`
   - Modify `TradingDaemon._load_recent_rows()`

---

## 7. Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone <repo_url>
cd BitCorn_Farmer

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt  # If exists
# or manually:
pip install torch numpy pandas matplotlib scikit-learn joblib

# Run tests
cd tests/
python test_integration.py
```

### Testing Changes

**Feature Engineering:**
```bash
python tests/test_feature_inspection.py
```

**Multi-Horizon Predictions:**
```bash
python tests/test_multi_horizon_integration.py
```

**Full Integration:**
```bash
python TradeApp.py  # Start GUI and test manually
```

### Code Style Guidelines

- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes
- **Docstrings:** Google-style docstrings for all public functions
- **Type Hints:** Use type hints for function signatures
- **Logging:** Use `logging` module, not `print()`
- **Error Handling:** Always catch specific exceptions, avoid bare `except:`

### Contribution Workflow

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and test thoroughly
3. Update documentation if needed
4. Commit with descriptive messages
5. Push and create pull request
6. Address review comments

---

## Common Development Tasks

### Adding a New Technical Indicator

1. **Define function in fiboevo.py:**
```python
def compute_custom_indicator(close, window=14):
    # Your logic
    return pd.Series(values)
```

2. **Add to feature engineering function:**
```python
def add_technical_features_v2(close, high, low, volume, dropna_after=True):
    df = pd.DataFrame()
    # ... existing features
    df['custom_indicator'] = compute_custom_indicator(close)
    return df
```

3. **Update expected_features in registry:**
```python
# core/feature_registry.py
expected_features = [..., 'custom_indicator']
```

4. **Retrain model with new features:**
- Launch TradeApp
- Go to Training tab
- Click "Prepare + Train"

### Debugging Prediction Issues

**Check feature compatibility:**
```python
from core.feature_registry import validate_feature_compatibility

model_features = meta["feature_cols"]  # From meta.json
system_features = FEATURE_REGISTRY.get_system_info("v2")["expected_features"]

is_compatible, missing = validate_feature_compatibility(model_features, system_features)
if not is_compatible:
    print(f"Missing features: {missing}")
```

**Check data pipeline:**
```python
# In trading_daemon.py, add logging
df = self._load_recent_rows(limit=100)
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
```

**Check model input shape:**
```python
# Before model inference
print(f"Input tensor shape: {input_tensor.shape}")
# Should be: (1, seq_len, n_features)
# e.g., (1, 32, 14) for v2 with seq_len=32
```

---

## Troubleshooting

### Issue: "Missing features" error

**Cause:** Model trained with different feature set than currently being computed

**Solution:**
1. Check meta.json: `"feature_cols": [...]`
2. Verify feature system matches:
```python
from core.feature_registry import detect_system_from_meta
system = detect_system_from_meta(meta)  # Should return "v2" for current model
```
3. If mismatch, retrain model or switch feature system in GUI

### Issue: Predictions not updating in GUI

**Cause:** Daemon not running or multi-horizon mode disabled

**Solution:**
1. Check daemon status in logs
2. Verify `daemon.multi_horizon_mode = True`
3. Check queue for messages: `daemon.predictions_queue.qsize()`
4. Restart daemon if stuck

### Issue: "No data found" in test

**Cause:** Database table doesn't exist or wrong timeframe

**Solution:**
1. Check available tables: `sqlite3 marketdata_base.db ".tables"`
2. Verify timeframe matches model: Check `meta.json` → `"timeframe"`
3. Ensure data ingestion pipeline ran successfully

---

## Performance Optimization

### Inference Speed

- **Single-point prediction:** 100-500ms (v2 features, 32 seq_len)
- **Batch prediction:** ~2-5s for 500 samples

**Optimization tips:**
- Use v2 features (14 vs 39) → 2.5x faster
- Reduce seq_len (32 vs 64) → 2x faster
- Use CPU for small batches (GPU overhead)
- Cache feature computation results

### Memory Usage

- **Model:** ~5-50 MB (depending on hidden size)
- **Scaler:** <1 MB
- **Features DataFrame:** ~1-10 MB (1000 rows)
- **Total daemon:** ~200-500 MB

---

## Additional Resources

- **Multi-Horizon Dashboard:** See `docs/MULTI_HORIZON_DASHBOARD.md`
- **Getting Started:** See `docs/GETTING_STARTED.md`
- **Feature Migration:** See `vault/MIGRACION_COMPLETADA.md`
- **Future Enhancements:** See `FUTURE_EXTENSIBILITY_GUIDE.md`

---

**Document Version:** 2.0
**Last Updated:** 2025-10-31
**Maintained by:** Project contributors
