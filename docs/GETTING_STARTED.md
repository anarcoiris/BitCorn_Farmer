# BitCorn Farmer - Getting Started Guide

**Quick start guide for new users**

---

## Installation

### Requirements

- Python 3.10+
- 8GB RAM minimum
- Windows 10/11, Linux, or macOS

### Setup

1. **Clone repository:**
```bash
git clone <repo_url>
cd BitCorn_Farmer
```

2. **Install dependencies:**
```bash
pip install torch numpy pandas matplotlib scikit-learn joblib websocket-client
```

3. **Verify installation:**
```bash
python -c "import torch, pandas, matplotlib; print('All dependencies installed!')"
```

---

## Data Preparation

### Option 1: Use CSV Data (Recommended for beginners)

1. **Download Binance historical data:**
   - Go to https://data.binance.vision/
   - Download BTCUSDT 1h or 30m data (CSV format)
   - Place in `data_manager/scripts/`

2. **Import to database:**
```bash
python csv_to_sqlite_upserter.py --csv data_manager/scripts/BTCUSDT_1h.csv --symbol BTCUSDT --timeframe 1h
```

### Option 2: Use Live WebSocket

1. **Start TradeApp:**
```bash
python TradeApp.py
```

2. **Connect WebSocket:**
   - Go to Status tab
   - Click "Connect Websocket"
   - Wait for data to accumulate (~1 hour minimum)

---

## Training Your First Model

### Quick Training

1. **Launch GUI:**
```bash
python TradeApp.py
```

2. **Configure training (Training tab):**
   - seq_len: 32
   - horizon: 10
   - hidden: 64
   - epochs: 50-100
   - Feature System: v2 (clean features)

3. **Click "Prepare + Train"**

4. **Wait for completion** (5-30 minutes depending on data size)

5. **Artifacts saved to:**
   - `artifacts/model_best.pt` - Trained model
   - `artifacts/scaler.pkl` - Feature scaler
   - `artifacts/meta.json` - Model metadata

### Manual Training (Advanced)

```bash
# Use retrain_clean_features.py for more control
python retrain_clean_features.py --epochs 100 --hidden 128 --lr 0.0005
```

---

## Running Predictions

### Option 1: GUI Dashboard (Live)

1. **Start TradeApp:**
```bash
python TradeApp.py
```

2. **Go to Status tab**

3. **Start daemon:**
   - Ensure model artifacts are loaded
   - Click "Start Daemon" (in Training or Status tab)

4. **Enable multi-horizon mode:**
   - Check "Enable Multi-Horizon Mode"
   - Watch prediction fan update every 5 seconds

### Option 2: Command Line (Batch)

```bash
# Run multi-horizon predictions on historical data
cd examples/
python example_multi_horizon.py

# Results saved to:
# - outputs/predictions/predictions_*.csv
# - outputs/plots/predictions_*.png
```

---

## Using the GUI

### Main Tabs

**Training Tab:**
- Configure model hyperparameters
- Train new models
- Load existing models

**Backtest Tab:**
- Test strategies on historical data
- Evaluate performance metrics

**Status Tab:**
- View live predictions
- Monitor daemon status
- Control WebSocket connection
- **Multi-horizon dashboard** (prediction fan + summary table)

**Audit Tab:**
- Inspect feature distributions
- Check for data quality issues
- Export feature datasets

### Status Tab - Multi-Horizon Dashboard

**Prediction Fan Canvas:**
- Black line: Historical price
- Red dot: Current price
- Colored lines: Predictions for each horizon
- Shaded areas: 95% confidence intervals

**Summary Table:**
- Horizon (1h, 3h, 5h, etc.)
- Predicted price
- Change in USD and %
- 95% confidence interval
- Signal direction (UP/DN)

**Controls:**
- "Enable Multi-Horizon Mode" - Toggle predictions on/off
- "Refresh Predictions" - Manual update
- "Last update" - Shows when last prediction was generated

---

## Common Workflows

### 1. Training a Model from Scratch

```bash
# Step 1: Prepare data
python csv_to_sqlite_upserter.py --csv BTCUSDT_1h.csv --symbol BTCUSDT --timeframe 1h

# Step 2: Train model
python TradeApp.py
# → Training tab → Configure → "Prepare + Train"

# Step 3: Verify artifacts
ls artifacts/
# Should see: model_best.pt, scaler.pkl, meta.json
```

### 2. Running Live Predictions

```bash
# Start GUI
python TradeApp.py

# In GUI:
# 1. Training tab → "Load artifacts model"
# 2. Status tab → "Connect Websocket"
# 3. Training tab → Click "Start Daemon"
# 4. Status tab → Enable "Multi-Horizon Mode"
# 5. Watch predictions update!
```

### 3. Analyzing Past Predictions

```bash
# Generate predictions on historical data
cd examples/
python example_multi_horizon.py

# View results
cd ../outputs/predictions/
# Open CSV in Excel or pandas
```

### 4. Comparing Feature Systems (v1 vs v2)

```bash
# In GUI:
# Training tab → Feature System dropdown → Select "v1" or "v2"
# Click "Prepare Data" → Check feature columns in logs

# v1 = 39 features (includes price levels)
# v2 = 14 clean features (stationary only)

# Recommendation: Use v2 for production
```

---

## Feature Systems

### v2 Features (Current, Recommended)

**14 Clean Features (No Price Leakage):**
- `log_ret_1`, `log_ret_5` - Log returns
- `sma_ratio_5/20/50` - Price/SMA ratios
- `ema_ratio_5/20/50` - Price/EMA ratios
- `bb_position` - Bollinger Band position
- `bb_width` - Bollinger Band width
- `rsi_14` - RSI indicator
- `atr_pct` - ATR percentage
- `raw_vol_10/30` - Rolling volatility

**Benefits:**
- ✓ No price-level leakage
- ✓ Better stationarity
- ✓ Faster computation (2.5x)
- ✓ More robust predictions

### v1 Features (Legacy)

**39 Features (Includes 26 Price Levels):**
- All v2 features +
- Fibonacci levels and distances
- Raw price indicators (log_close, SMA, EMA, BB)

**Use cases:**
- Compatibility with old models
- Research/experimentation

---

## Troubleshooting

### Issue: "No data found in database"

**Solution:**
1. Check database exists: `ls data_manager/exports/marketdata_base.db`
2. Check tables: `sqlite3 marketdata_base.db ".tables"`
3. If empty, re-run CSV upserter or wait for WebSocket data

### Issue: "Missing features" error

**Solution:**
1. Check `artifacts/meta.json` → `"feature_cols"`
2. Verify feature system matches:
   - If meta has 14 features → Use v2
   - If meta has 39 features → Use v1
3. Set correct system in Training tab dropdown

### Issue: Model training fails

**Common causes:**
- Not enough data (need 1000+ rows minimum)
- Missing required columns (close, high, low, volume)
- Insufficient RAM (increase batch_size or reduce seq_len)

**Solutions:**
- Check data: `sqlite3 marketdata_base.db "SELECT COUNT(*) FROM ohlcv WHERE symbol='BTCUSDT'"`
- Reduce batch_size: Try 32 instead of 64
- Reduce seq_len: Try 16 or 24 instead of 32

### Issue: GUI freezes during training

**Solution:**
- Training runs in background thread
- Check logs tab for progress
- Wait for "Training complete" message
- If truly frozen, restart and use smaller dataset or fewer epochs

---

## Next Steps

Once you're comfortable with the basics:

1. **Read Developer Guide:** `docs/DEVELOPER_GUIDE.md` - Technical details
2. **Explore Multi-Horizon Dashboard:** `docs/MULTI_HORIZON_DASHBOARD.md`
3. **Try Advanced Features:** Custom feature engineering, model ensembles
4. **Contribute:** See `docs/DEVELOPER_GUIDE.md` → Contribution Workflow

---

## Quick Reference

### Important Files

| File | Purpose |
|------|---------|
| `artifacts/model_best.pt` | Trained LSTM model |
| `artifacts/scaler.pkl` | Feature scaler |
| `artifacts/meta.json` | Model metadata (features, hyperparams) |
| `config/gui_config.json` | GUI settings (auto-saved) |
| `data_manager/exports/marketdata_base.db` | OHLCV database |

### Important Commands

```bash
# Start GUI
python TradeApp.py

# Import CSV data
python csv_to_sqlite_upserter.py --csv FILE.csv --symbol BTCUSDT --timeframe 1h

# Train model (CLI)
python retrain_clean_features.py --epochs 100

# Run predictions (CLI)
cd examples/ && python example_multi_horizon.py

# Run tests
cd tests/ && python test_integration.py
```

### Configuration Defaults

| Parameter | Default | Recommended Range |
|-----------|---------|-------------------|
| seq_len | 32 | 16-64 |
| horizon | 10 | 5-30 |
| hidden | 64 | 32-128 |
| epochs | 10 | 50-200 |
| batch_size | 64 | 32-128 |
| learning_rate | 0.001 | 0.0001-0.01 |
| val_fraction | 0.1 | 0.1-0.2 |

---

## Getting Help

- **Issues:** GitHub Issues (if public repo)
- **Documentation:** `/docs` directory
- **Examples:** `/examples` directory
- **Tests:** `/tests` directory

---

**Last Updated:** 2025-10-31
**Version:** 2.0
