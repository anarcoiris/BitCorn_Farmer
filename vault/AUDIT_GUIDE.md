# Audit Functionality Guide

## Overview

The TradeApp GUI includes a comprehensive **Audit Tab** that performs automated checks to detect data leakage and validate model training setup. This is critical for preventing common ML mistakes that lead to overfitting.

## Location

- **File**: `TradeApp.py`
- **Function**: `_run_audit()` (line 1805)
- **Tab**: "Audit" tab in the GUI Notebook

## Purpose

The audit system checks for:
1. **Data leakage** - Features that contain future information
2. **Scaler validation** - Proper scaler fitting on training data only
3. **Feature alignment** - Correct feature column ordering
4. **Suspicious patterns** - Naming and correlation red flags

---

## Audit Checks

### 1. Metadata Columns Check
**Check**: `metadata_columns`
- **Purpose**: Detects if metadata columns (timestamp, symbol, created_at, etc.) are present in feature set
- **Pass condition**: No metadata columns found
- **Fail condition**: Metadata columns detected
- **Action if failed**: Remove metadata columns before training

### 2. Data Integrity Check
**Check**: `dropna_rows`
- **Purpose**: Reports how many rows were dropped due to NaN values during feature engineering
- **Details**: Shows rows before vs after dropna
- **Info only**: Helps understand data quality

### 3. Scaler Statistics Check
**Check**: `scaled_stats` / `possible_scaler_fitted_on_all`
- **Purpose**: Detects if scaler was fitted on full dataset (data leakage)
- **Method**:
  - Checks if >80% of features have mean ≈ 0 and std ≈ 1
  - This pattern suggests scaler was fitted on all data including test set
- **Pass condition**: Scaled data NOT globally standardized
- **Fail condition**: All features perfectly standardized
- **Critical**: This indicates **DATA LEAKAGE** - scaler learned statistics from validation/test sets

### 4. Scaler Feature Names Alignment
**Check**: `scaler_feature_names_alignment`
- **Purpose**: Validates that scaler.feature_names_in_ matches actual feature columns
- **Pass condition**: All scaler features present in dataframe
- **Fail condition**: Missing features detected
- **Action if failed**:
  - Re-fit scaler with correct feature columns
  - Use `fiboevo.ensure_scaler_feature_names()` to fix

### 5. Exact Shift Matches
**Check**: `exact_shift_matches`
- **Purpose**: Detects features that are EXACTLY equal to `close.shift(-k)` (future data)
- **Method**: Tests shifts from k=1 to k=horizon
- **Pass condition**: No exact matches found
- **Fail condition**: Features match future close values
- **Critical**: This is **EXPLICIT DATA LEAKAGE** - model sees the future

### 6. High Correlation with Future Close
**Check**: `very_high_corr_with_future_close`
- **Purpose**: Flags features with correlation > 0.95 with future close
- **Method**: Computes correlation with `close.shift(-horizon)`
- **Pass condition**: No features > 0.95 correlation
- **Fail condition**: Features highly correlated with future
- **Warning**: Likely **DATA LEAKAGE** or improper feature construction

### 7. Suspicious Feature Names
**Check**: `suspicious_feature_names`
- **Purpose**: Detects feature names containing suspicious keywords
- **Keywords**: "shift", "t+1", "next", "future", "target", "_lead", "_lag"
- **Pass condition**: No suspicious names
- **Fail condition**: Suspicious names detected
- **Action if failed**: Review feature engineering - may contain forward-looking data

### 8. Recommended Train End Index
**Check**: `recommended_train_end`
- **Purpose**: Calculates the correct row index for fitting scaler
- **Output**: `train_rows_end` index
- **Usage**: Use `df.iloc[:train_rows_end+1]` to fit scaler on training data only
- **Info only**: Provides guidance for preventing data leakage

---

## How to Run Audit

### In GUI:
1. Load or prepare data (Prepare tab)
2. Navigate to **Audit** tab
3. Click **"Run Audit"** button
4. Review results in text area and table

### Audit Report Output:
- **Text area**: Full report with all checks
- **Table**: Quick summary of top 8 checks
- **Export**: Click "Save Audit Report" to export JSON

---

## Interpreting Results

### Status Indicators:
- **OK** - Check passed (green flag)
- **WARN** - Check failed (red flag)
- **N/A** - Check not applicable or insufficient data

### Critical Failures:
If any of these checks fail, **DO NOT TRAIN THE MODEL** until fixed:

1. ❌ **exact_shift_matches** - Features contain future data
2. ❌ **possible_scaler_fitted_on_all** - Scaler fitted on full dataset
3. ❌ **very_high_corr_with_future_close** - Features leak future information

### Warning Failures:
These should be investigated but may not always indicate problems:

1. ⚠️ **suspicious_feature_names** - Review feature construction
2. ⚠️ **scaler_feature_names_alignment** - Update scaler or features
3. ⚠️ **metadata_columns** - Remove from feature set

---

## Common Issues and Fixes

### Issue 1: Scaler Fitted on All Data
**Symptom**: `possible_scaler_fitted_on_all` check fails
**Cause**: Scaler fitted on full dataset before train/val/test split
**Fix**:
```python
# Use temporal_split_indexes from fiboevo
n_train_seq, n_val_seq, n_test_seq, train_rows_end = fiboevo.temporal_split_indexes(
    n_total=len(df),
    seq_len=seq_len,
    horizon=horizon,
    val_frac=0.2,
    test_frac=0.1
)

# Fit scaler ONLY on training rows
scaler.fit(df[feature_cols].iloc[:train_rows_end + 1])
```

### Issue 2: Features Match Future Close
**Symptom**: `exact_shift_matches` check fails
**Cause**: Feature engineering created forward-looking variables
**Fix**: Review `add_technical_features()` - remove any features using `.shift(-k)`

### Issue 3: High Correlation with Future
**Symptom**: `very_high_corr_with_future_close` check fails
**Cause**: Features constructed using future information
**Fix**:
- Check rolling window calculations use only past data
- Ensure indicators don't access future candles
- Review Fibonacci levels calculation

### Issue 4: Feature Names Mismatch
**Symptom**: `scaler_feature_names_alignment` check fails
**Cause**: Scaler.feature_names_in_ doesn't match current features
**Fix**:
```python
# Use new fiboevo API
scaler = fiboevo.load_scaler(
    "artifacts/scaler.pkl",
    feature_cols=meta["feature_cols"]
)

# Or manually set
scaler.feature_names_in_ = np.array(feature_cols, dtype=object)
```

---

## Integration with prepare_dataset.py

The audit checks align with best practices implemented in `prepare_dataset.py`:

### ✅ Correct Pattern (Current Implementation):
```python
# 1. Use temporal_split_indexes to get train_rows_end
n_train_seq, n_val_seq, n_test_seq, train_rows_end = fiboevo.temporal_split_indexes(
    n_total=len(df_clean),
    seq_len=seq_len,
    horizon=horizon,
    val_frac=0.2,
    test_frac=0.1
)

# 2. Fit scaler ONLY on training rows
scaler = StandardScaler()
scaler.fit(df_clean[feature_cols].iloc[:train_rows_end + 1])

# 3. Set feature names
scaler.feature_names_in_ = np.array(feature_cols, dtype=object)

# 4. Transform all data (fitted scaler only used training stats)
df_clean[feature_cols] = scaler.transform(df_clean[feature_cols])

# 5. Save metadata with train_rows_end for audit trail
meta = {
    "feature_cols": feature_cols,
    "train_rows_end": int(train_rows_end),
    "scaler_fitted_on": "train_only",  # Documentation
    # ... other meta
}
```

### ❌ Wrong Pattern (Data Leakage):
```python
# BAD: Fits scaler on ALL data
scaler.fit(df[feature_cols])  # Learns statistics from test set!
```

---

## Audit Report Structure

### JSON Export Format:
```json
{
  "timestamp": "2025-10-14 12:34:56",
  "checks": [
    {
      "name": "exact_shift_matches",
      "ok": true,
      "msg": "No exact matches to future close detected"
    },
    {
      "name": "possible_scaler_fitted_on_all",
      "ok": false,
      "msg": "Scaled data statistics suggest scaler was fit on full dataset"
    }
  ],
  "summary": {
    "n_features": 50,
    "n_rows": 10000,
    "seq_len": 32,
    "horizon": 10,
    "scaler_present": true
  }
}
```

---

## Automation

### Running Audit Programmatically:
```python
# In your training script
app = TradingAppExtended(root)
app._prepare_data_worker()  # Load and prepare data
app._run_audit()  # Run audit checks

# Check results
if hasattr(app, "_last_audit_report"):
    report = app._last_audit_report
    failures = [c for c in report["checks"] if c.get("ok") == False]
    if failures:
        print("AUDIT FAILED - Do not train model!")
        for f in failures:
            print(f"  - {f['name']}: {f['msg']}")
        sys.exit(1)
```

---

## Best Practices

### Before Every Training Run:
1. ✅ Run audit after preparing data
2. ✅ Fix all critical failures (exact_shift, scaler_fitted_on_all)
3. ✅ Investigate warning failures
4. ✅ Save audit report with training artifacts
5. ✅ Document any exceptions in training log

### During Development:
1. ✅ Run audit after modifying feature engineering
2. ✅ Check audit when model performance seems "too good"
3. ✅ Compare audit reports between training runs
4. ✅ Use audit to validate new feature additions

### Before Production:
1. ✅ Run audit on final training dataset
2. ✅ Verify scaler.feature_names_in_ is set
3. ✅ Confirm train_rows_end is in metadata
4. ✅ Archive clean audit report

---

## Related Files

- **fiboevo.py**: Core ML module with data leakage prevention
  - `temporal_split_indexes()`: Calculates correct split indices
  - `ensure_scaler_feature_names()`: Validates scaler alignment
  - `prepare_input_for_model()`: Prevents double-scaling

- **prepare_dataset.py**: Dataset preparation with audit-compliant patterns
  - Uses `temporal_split_indexes()`
  - Sets `scaler.feature_names_in_`
  - Saves `train_rows_end` in metadata

- **trading_daemon.py**: Inference daemon with feature validation
  - Checks feature_cols from meta
  - Uses `prepare_input_for_model()`
  - Validates scaler alignment

- **CLAUDE.md**: Project documentation
  - Data leakage prevention strategies
  - API migration guide
  - Critical recent updates

---

## Troubleshooting

### Audit Won't Run:
- Check that `df_features` exists (run Prepare first)
- Verify fiboevo module is imported correctly
- Check for exceptions in audit_text output

### False Positives:
- High correlation may occur naturally for strongly trending features
- Review specific features flagged, not just correlation value
- Suspicious names may be safe (e.g., "_lag_window" for rolling window)

### Missing Checks:
- Ensure GUI is up to date with latest TradeApp.py
- Verify all required libraries installed (numpy, pandas, sklearn)
- Check Python version compatibility (3.8+)

---

## Summary

The Audit Tab is a **critical safety check** that prevents expensive mistakes in ML model training. Always run the audit and address failures before training models for production use.

**Key Takeaway**: A clean audit report = confidence that your model isn't cheating by seeing the future!
