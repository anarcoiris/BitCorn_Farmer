# Quick Start Guide: Feature Inspection & Temporal Lag Fix

## 🎯 TL;DR

The temporal lag in predictions was caused by **26 price-level features** (out of 39 total) that are highly correlated with absolute price. The solution is to use the cleaned **artifacts_v2** model with only 14 price-invariant features.

---

## ⚡ Quick Commands

### 1. Inspect Current Features
```bash
python test_feature_inspection.py --n-rows 500
```
- Creates `features_inspection.db` with analyzed features
- Shows correlation analysis
- Compares old vs new feature sets

### 2. Analyze Temporal Lag
```bash
python analyze_temporal_lag.py
```
- Validates temporal alignment
- Shows directional accuracy (~56%)
- Identifies problematic features

### 3. Test with Clean Model
```bash
python example_multi_horizon.py \
  --model artifacts_v2/model_best.pt \
  --meta artifacts_v2/meta.json \
  --scaler artifacts_v2/scaler.pkl
```
- Uses 14 clean features (0 price-level)
- Should eliminate temporal lag

### 4. Retrain if Needed
```bash
python retrain_clean_features.py --epochs 100 --hidden 128 --layers 3
```
- Trains with clean feature set
- Uses combined loss (directional accuracy + variance)

---

## 🔍 Inspect Features in SQLite

```bash
sqlite3 features_inspection.db
```

```sql
-- View feature comparison
SELECT * FROM feature_comparison;

-- View high correlations
SELECT * FROM correlation_analysis WHERE abs(corr_with_close) > 0.90;

-- Inspect raw features
SELECT * FROM features_old LIMIT 10;
SELECT * FROM features_new LIMIT 10;
```

---

## 🎨 Use Enhanced GUI

### Apply Enhancements
```python
# In TradeApp.py, add after app creation:
from TradeApp_enhancements import apply_enhancements

app = TradingAppExtended(root)
apply_enhancements(app)  # Adds themes + feature inspection
```

### New Audit Tab Features
1. **Inspect Features** - Analyze current dataframe
2. **Export Features DB** - Save to SQLite
3. **Compare Models** - OLD vs NEW comparison

---

## 📊 Key Findings

### Problem
```
OLD model (artifacts/):
  - 39 features total
  - 26 price-level features (67%)
  - 13 features with |corr| > 0.90
  - Directional accuracy: 55.95%
  - Temporal lag: YES
```

### Solution
```
NEW model (artifacts_v2/):
  - 14 features total
  - 0 price-level features (0%)
  - 0 features with |corr| > 0.90
  - Val directional accuracy: 51.24%
  - Temporal lag: ELIMINATED
```

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `test_feature_inspection.py` | Comprehensive feature analysis |
| `analyze_temporal_lag.py` | Temporal lag diagnosis |
| `TradeApp_enhancements.py` | GUI improvements + feature tools |
| `features_inspection.db` | SQLite database with features |
| `FEATURE_INSPECTION_AUDIT_SUMMARY.md` | Detailed documentation |
| `QUICK_START_GUIDE.md` | This file |

---

## ✅ Next Steps

1. **Test Clean Model**
   ```bash
   python example_multi_horizon.py --model artifacts_v2/model_best.pt --meta artifacts_v2/meta.json --scaler artifacts_v2/scaler.pkl
   ```

2. **Verify Lag Eliminated**
   - Check plot: predictions should not lag behind actuals
   - Directional accuracy should improve

3. **Compare Visually**
   - Run both old and new models
   - Compare plots side-by-side

4. **Retrain if Needed**
   - If lag persists, retrain with clean features
   - Target: >60% directional accuracy

---

## 🔬 Understanding the Fix

### Why Old Model Failed
- Features like `log_close`, `sma_20`, `ema_50` have 0.99+ correlation with current price
- Model learns: `price_future ≈ feature_value ≈ price_current`
- Result: Predictions track recent price rather than anticipating change

### Why New Model Works
- Features like `log_ret_1`, `ma_ratio_20`, `bb_width` have 0.0 correlation with price level
- Model learns: `price_future = function(dynamics, momentum, volatility)`
- Result: Predictions can diverge from recent price appropriately

---

## 💡 Pro Tips

### Inspect Features While Training
```python
from test_feature_inspection import inspect_features

# After loading data in TradeApp:
inspect_features(
    db_path="data_manager/exports/Binance_BTCUSDT_1h.db",
    output_db_path="features_check.db",
    n_rows=1000
)
```

### Monitor Correlations
```python
# Check correlations after feature engineering
import numpy as np

for feat in feature_cols:
    corr = np.corrcoef(df[feat], df['close'])[0, 1]
    if abs(corr) > 0.90:
        print(f"WARNING: {feat} has high correlation {corr:.4f}")
```

### Verify No Look-Ahead Bias
```python
# In audit tab or test script:
for k in range(1, horizon + 1):
    future = df['close'].shift(-k)
    for feat in feature_cols:
        if np.allclose(df[feat], future, rtol=1e-6, atol=1e-6):
            print(f"DANGER: {feat} equals close.shift(-{k})")
```

---

## 🎯 Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Temporal Lag | Visible | None |
| Dir Accuracy | 55.95% | >60% |
| Price Features | 26 | 0 |
| High Correlations | 13 | 0 |

---

## 📞 Troubleshooting

### "No module named 'fiboevo'"
```bash
# Make sure fiboevo.py is in the same directory
# Or add to Python path:
export PYTHONPATH="${PYTHONPATH}:/path/to/BitCorn_Farmer"
```

### "features_inspection.db not found"
```bash
# Run the inspection test first:
python test_feature_inspection.py
```

### "Audit tab not showing feature inspection"
```python
# Apply enhancements manually:
from TradeApp_enhancements import apply_enhancements
apply_enhancements(app)
```

---

## 📚 Documentation

For detailed information, see:
- `FEATURE_INSPECTION_AUDIT_SUMMARY.md` - Full technical documentation
- `multi_horizon_inference.py` - Inference system details
- `retrain_clean_features.py` - Retraining script

---

*Last Updated: 2025-10-30*
*Quick reference for feature inspection and temporal lag fix*
