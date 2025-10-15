# Code Analysis: fiboevo.py and TradeApp.py Integration

## Executive Summary

Based on analysis of fiboevo.py (1990 lines) and TradeApp.py (3765 lines), plus the notes files, the new fiboevo.py introduces several breaking changes that require coordinated updates across the codebase.

## Key Changes in fiboevo.py

### 1. **load_model() signature change**
- **Old**: `model = load_model(path)`
- **New**: `model, meta = load_model(path)`
- **Impact**: ALL callers must update to tuple unpacking
- **Benefit**: Provides metadata for validation and feature column alignment

### 2. **load_scaler() enhanced**
- **New signature**: `load_scaler(path, feature_cols=None)`
- **Feature**: Auto-injects `feature_names_in_` from meta if missing
- **New helper**: `ensure_scaler_feature_names(scaler, meta)`
- **Purpose**: Prevents silent column misalignment at inference

### 3. **prepare_input_for_model() - NEW CRITICAL FUNCTION**
```python
prepare_input_for_model(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    seq_len: int,
    scaler: Optional[Any] = None,
    method: str = "per_row"
) -> torch.Tensor
```
- **Purpose**: Unified inference input preparation
- **Benefits**:
  - Prevents double-scaling
  - Enforces feature column order matching scaler
  - Validates `scaler.feature_names_in_` vs `feature_cols`
  - Raises RuntimeError on mismatch instead of silent failure
- **Returns**: `torch.Tensor` (1, seq_len, n_features) on CPU

### 4. **simulate_fill() enhanced with liquidity model**
- **Backward compatible**: `use_liquidity_model=False` by default
- **New mode**: Liquidity-aware fill simulation
- **Return value change**:
  - Simple mode: `(filled, price, fill_label)` - label is index
  - Liquidity mode: `(filled, price, filled_fraction)` - fraction is float
- **Impact**: Callers must check return type or mode

### 5. **Temporal split function**
```python
temporal_split_indexes(n_total, seq_len, horizon, val_frac=0.2, test_frac=0.1)
```
- Returns correct indices for fitting scaler ONLY on train data
- Prevents data leakage from validation/test into scaler statistics

## Required Changes by Module

### A. trading_daemon.py

#### Priority: **CRITICAL**

**Changes needed**:

1. **Update model/scaler loading** (line ~220):
```python
# BEFORE:
model = fiboevo.load_model(model_path)
scaler = joblib.load(scaler_path)

# AFTER:
model, model_meta = fiboevo.load_model(model_path)
scaler = fiboevo.load_scaler(scaler_path, feature_cols=model_meta.get("feature_cols"))
# OR ensure feature names:
scaler = fiboevo.ensure_scaler_feature_names(scaler, model_meta)
```

2. **Store model_meta in daemon state**:
```python
self.model = model
self.model_meta = model_meta or {}
self.model_scaler = scaler
```

3. **Update iteration_once() prediction** (line ~608):
```python
# BEFORE (manual sequence building):
seq_df = feats_df.iloc[-seq_len:].reset_index(drop=True)
X_in = seq_df[feature_cols].values.astype(np.float32)
if scaler:
    X_in = scaler.transform(X_in)
x_t = torch.from_numpy(X_in).unsqueeze(0).to(device)

# AFTER (use prepare_input_for_model):
seq_df = feats_df.iloc[-seq_len:].reset_index(drop=True)
x_t = fiboevo.prepare_input_for_model(
    seq_df, feature_cols, seq_len, scaler=self.model_scaler
).to(device)
```

4. **Add validation after load**:
```python
def _validate_model_artifacts(self):
    if not self.model_meta:
        return
    expected_cols = self.model_meta.get("feature_cols")
    if expected_cols:
        # Check current features match expectations
        missing = [c for c in expected_cols if c not in current_feats.columns]
        if missing:
            raise RuntimeError(f"Missing features: {missing}")
```

#### Files referencing load_model:
- `trading_daemon.py:224` (load_model_and_scaler method)

### B. TradeApp.py (GUI)

#### Priority: **HIGH**

**Changes needed**:

1. **Update _load_model_from_artifacts()** (line ~1688):
```python
def _load_model_from_artifacts(self):
    model_path = Path("artifacts/model_best.pt")
    if not model_path.exists():
        self._enqueue_log("Model not found")
        return
    try:
        # NEW: tuple unpacking
        self.model, self.model_meta = fibo.load_model(model_path)

        # Load scaler with meta
        scaler_path = Path("artifacts/scaler.pkl")
        if scaler_path.exists():
            self.model_scaler = fibo.load_scaler(
                scaler_path,
                feature_cols=self.model_meta.get("feature_cols")
            )

        self._enqueue_log(f"Loaded model + meta: {self.model_meta}")
    except Exception as e:
        self._enqueue_log(f"Load failed: {e}")
```

2. **Update _load_model_file_background()** (line ~1690):
```python
def _load_model_file_background(self):
    model_path = self.model_path_var.get()
    if not model_path:
        self._enqueue_log("No model path specified")
        return

    def worker():
        try:
            model, meta = fibo.load_model(Path(model_path))
            # Also load scaler if present
            scaler_path = Path(model_path).parent / "scaler.pkl"
            scaler = None
            if scaler_path.exists():
                scaler = fibo.load_scaler(scaler_path, feature_cols=meta.get("feature_cols"))

            # Update GUI state on main thread
            def ui_update():
                self.model = model
                self.model_meta = meta
                self.model_scaler = scaler
                self._enqueue_log(f"Loaded model: {meta}")

                # Sync with daemon if running
                if self.daemon:
                    self.daemon.model = model
                    self.daemon.model_meta = meta
                    self.daemon.model_scaler = scaler
                    self._enqueue_log("Synced model to daemon")

            self.root.after(0, ui_update)
        except Exception as e:
            self._enqueue_log(f"Background load failed: {e}\n{traceback.format_exc()}")

    threading.Thread(target=worker, daemon=True).start()
```

3. **Update _get_latest_prediction_thread()** (likely line ~2050-2100):
```python
def _get_latest_prediction_thread(self):
    def worker():
        try:
            # Load recent data
            df = self._load_recent_data()
            if df.empty:
                self._enqueue_log("No data for prediction")
                return

            # Get feature cols from meta
            feature_cols = self.model_meta.get("feature_cols") if self.model_meta else self.feature_cols_used
            if not feature_cols:
                self._enqueue_log("No feature_cols defined")
                return

            # Compute features
            feats = fibo.add_technical_features(
                df["close"].values,
                high=df["high"].values if "high" in df.columns else None,
                low=df["low"].values if "low" in df.columns else None
            )

            # Prepare input using new helper
            seq_len = int(self.seq_len.get())
            x_t = fibo.prepare_input_for_model(
                feats,
                feature_cols,
                seq_len,
                scaler=self.model_scaler
            )

            # Predict
            device = next(self.model.parameters()).device
            x_t = x_t.to(device)
            self.model.eval()
            with torch.no_grad():
                pred_ret, pred_vol = self.model(x_t)

            pred_ret = float(pred_ret.cpu().numpy().ravel()[0])
            pred_vol = float(pred_vol.cpu().numpy().ravel()[0])

            # Update UI
            def ui_update():
                self.pred_log_var.set(f"{pred_ret:.6f}")
                pct = (np.exp(pred_ret) - 1) * 100
                self.pred_pct_var.set(f"{pct:.4f}%")
                self.pred_vol_var.set(f"{pred_vol:.6f}")
                self.pred_ts_var.set(safe_now_str())
            self.root.after(0, ui_update)

        except Exception as e:
            self._enqueue_log(f"Prediction failed: {e}\n{traceback.format_exc()}")

    threading.Thread(target=worker, daemon=True).start()
```

4. **Update _run_forecast()** (line ~548):
```python
def _run_forecast(self):
    # ... existing setup ...

    # Use prepare_input_for_model for each step
    for step in range(N):
        try:
            # Prepare scaled input
            x_t = fibo.prepare_input_for_model(
                seq_df,  # current sequence as DataFrame
                feature_cols,
                seq_len,
                scaler=scaler,
                method="per_row"
            )

            # Move to device and predict
            device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
            x_t = x_t.to(device)

            with torch.no_grad():
                out = self.model(x_t)
            # ... rest of prediction logic ...
        except Exception as e:
            self._enqueue_log(f"Forecast step {step} failed: {e}")
```

### C. prepare_dataset.py

#### Priority: **HIGH**

**Changes needed**:

1. **Use temporal_split_indexes()**:
```python
from fiboevo import temporal_split_indexes

# Determine split indices BEFORE fitting scaler
n_train_seq, n_val_seq, n_test_seq, train_rows_end = temporal_split_indexes(
    n_total=len(df),
    seq_len=seq_len,
    horizon=horizon,
    val_frac=0.2,
    test_frac=0.1
)

# Fit scaler ONLY on train rows
scaler = StandardScaler()
scaler.fit(df[feature_cols].iloc[:train_rows_end + 1].values)

# Set feature names
scaler.feature_names_in_ = np.array(feature_cols, dtype=object)

# Transform full df (scaler learned from train only)
df[feature_cols] = scaler.transform(df[feature_cols].values)

# Create sequences and split
X_all, y_all, v_all = fibo.create_sequences_from_df(df, feature_cols, seq_len, horizon)
Xtr = X_all[:n_train_seq]
Xv = X_all[n_train_seq:n_train_seq + n_val_seq]
Xt = X_all[n_train_seq + n_val_seq:]
# ... same for y_all, v_all
```

2. **Save complete metadata**:
```python
meta = {
    "feature_cols": feature_cols,  # CRITICAL
    "input_size": len(feature_cols),
    "hidden": hidden,
    "num_layers": num_layers,
    "seq_len": seq_len,
    "horizon": horizon,
    "symbol": symbol,
    "timeframe": timeframe,
    "train_rows_end": train_rows_end,  # For audit
    "scaler_fitted_on": "train_only",  # Documentation
}

# Save with model
fibo.save_model(model, Path("artifacts/model_best.pt"), meta=meta)

# Save scaler with feature names
fibo.save_scaler(scaler, Path("artifacts/scaler.pkl"), feature_cols=feature_cols)
```

### D. Backtest Updates

#### Priority: **MEDIUM**

**In any backtest code**:

1. **Pass scaler to backtest_market_maker()**:
```python
summary, trades = fibo.backtest_market_maker(
    df_test,
    model,
    feature_cols,
    seq_len=seq_len,
    horizon=horizon,
    scaler=scaler,  # NEW: required for prepare_input_for_model
    # ... other params
)
```

2. **Handle simulate_fill() return value**:
```python
filled, entry_price, fill_info = fibo.simulate_fill(
    order_price, side, future_segment,
    use_liquidity_model=False  # or True
)

if filled:
    if isinstance(fill_info, float):
        # Liquidity mode: fill_info is filled_fraction
        filled_fraction = fill_info
        pos_fill = t  # approximate
    else:
        # Simple mode: fill_info is index label
        try:
            pos_fill = int(df.index.get_loc(fill_info))
        except Exception:
            pos_fill = t
```

## Data Leakage Prevention Checklist

### ✅ Scaler Fit Only on Train
- Use `temporal_split_indexes()` to get `train_rows_end`
- `scaler.fit(df[feature_cols].iloc[:train_rows_end + 1])`
- **Never** `scaler.fit(df[feature_cols])` on full data

### ✅ Feature Names Alignment
- Always set `scaler.feature_names_in_` when saving
- Use `ensure_scaler_feature_names()` when loading
- `prepare_input_for_model()` validates alignment

### ✅ No Future Data in Features
- Run audit checks: `_run_audit()` in GUI
- Check for suspicious feature names: "shift", "lead", "next", "future"
- Validate no exact matches to `close.shift(-k)`
- Check correlations with future close < 0.95

### ✅ Index Management
- Maintain DatetimeIndex for backtests
- Only `reset_index(drop=True)` inside `create_sequences_from_df()` (already done)
- Don't reset indices before passing to `simulate_fill()`

## Conflict Resolution

### Conflict: Double Scaling

**Problem**: Some code may pre-scale df before passing to model

**Solution**:
- **Standard**: Never scale df globally; only scale inside `prepare_input_for_model()`
- **Defense**: `prepare_input_for_model()` could check if data already standardized (mean≈0, std≈1) and warn
- **Migration**: Update all callers to pass raw df + scaler to `prepare_input_for_model()`

### Conflict: simulate_fill() return type

**Problem**: Third return value varies (label vs fraction)

**Current mitigation**:
- Default `use_liquidity_model=False` preserves backward compat
- Callers using liquidity mode must check `isinstance(fill_info, float)`

**Future**: Change to always return dict:
```python
return filled, entry_price, {
    "label": fill_label,  # or None
    "filled_fraction": fraction,  # or None
}
```

## Testing Strategy

### Phase 1: Update and Test Daemon
1. Update `trading_daemon.py` with new signatures
2. Test `load_model_and_scaler()` standalone
3. Test `iteration_once()` with `prepare_input_for_model()`
4. Verify no errors on model mismatch (should raise clear errors)

### Phase 2: Update GUI
1. Update all `load_model*` methods
2. Test loading artifacts
3. Test loading custom model file
4. Test "Get Latest Prediction"
5. Test forecast window

### Phase 3: Update Training Pipeline
1. Update `prepare_dataset.py` with temporal split
2. Verify scaler fitted only on train
3. Test that model saves with complete meta
4. Run audit checks

### Phase 4: Backtest
1. Update backtest calls to pass `scaler`
2. Test with both liquidity modes
3. Verify fills work correctly

## Verification Commands

```bash
# Test daemon smoke
python trading_daemon.py --sqlite data_manager/exports/marketdata_base.db --table ohlcv --symbol BTCUSDT --timeframe 30m --paper --seq_len 32 --poll 2.0

# Test GUI (visual check)
python TradeApp.py

# Test training with temporal split
python prepare_dataset.py  # (after updating)

# Run audit
# Use GUI Audit tab after loading/preparing data
```

## Files to Update (Priority Order)

1. **fiboevo.py** - ✅ Already updated (provided)
2. **trading_daemon.py** - 🔴 CRITICAL - Model/scaler loading, inference
3. **TradeApp.py** - 🔴 HIGH - All model load methods, prediction, forecast
4. **prepare_dataset.py** - 🔴 HIGH - Temporal split, scaler fitting
5. **Any backtest scripts** - 🟡 MEDIUM - Pass scaler, handle fill returns

## API Compatibility Matrix

| Function | Old Signature | New Signature | Breaking? |
|----------|--------------|---------------|-----------|
| `load_model()` | `model = load_model(path)` | `model, meta = load_model(path)` | ✅ YES |
| `load_scaler()` | `scaler = joblib.load(path)` | `scaler = load_scaler(path, feature_cols=...)` | ⚠️ NEW (optional use) |
| `simulate_fill()` | `(filled, price, label)` | `(filled, price, label\|fraction)` | ⚠️ CONDITIONAL |
| `prepare_input_for_model()` | N/A | NEW | ✅ NEW |
| `ensure_scaler_feature_names()` | N/A | NEW | ✅ NEW |
| `temporal_split_indexes()` | N/A | NEW | ✅ NEW |

## Migration Checklist

- [ ] Update `trading_daemon.py` load_model calls
- [ ] Update `trading_daemon.py` iteration_once() to use prepare_input_for_model
- [ ] Update `TradeApp.py` _load_model_from_artifacts
- [ ] Update `TradeApp.py` _load_model_file_background
- [ ] Update `TradeApp.py` _get_latest_prediction_thread
- [ ] Update `TradeApp.py` _run_forecast
- [ ] Update `prepare_dataset.py` to use temporal_split_indexes
- [ ] Update `prepare_dataset.py` to save complete meta
- [ ] Update backtests to pass scaler
- [ ] Run audit checks on training data
- [ ] Test daemon smoke run
- [ ] Test GUI model loading
- [ ] Test GUI prediction
- [ ] Test GUI forecast
- [ ] Verify no double-scaling issues
- [ ] Document changes in CLAUDE.md

## Notes from fiboevo_notes.md

The notes emphasize that this update is designed to:
1. **Fix data leakage** - scaler fitted on full dataset is a major issue
2. **Prevent silent misalignment** - feature column order must match scaler
3. **Enable proper validation** - metadata allows checking compatibility
4. **Simplify inference** - `prepare_input_for_model()` handles complexity

The approach is **defensive by default** - raises errors rather than producing wrong results silently.
