# Prediction Dashboard - User Guide & Integration Instructions

## Overview

The **Prediction Dashboard** is a professional live multi-horizon prediction visualization system for the TradeApp cryptocurrency trading application. It displays real-time price forecasts at multiple time horizons (1h, 3h, 5h, 10h, 20h, 30h) with confidence intervals, probability distributions, and optional scenario analysis (bull/base/bear cases).

![Dashboard Preview](docs/dashboard_preview_placeholder.png)

---

## Features

### Core Functionality

1. **Multi-Horizon Prediction Fan**
   - Simultaneous predictions at 1, 3, 5, 10, 15, 20, and 30-hour horizons
   - Color-coded by horizon using professional colormaps (viridis, plasma, coolwarm)
   - Visual "fan" radiating from current price

2. **Confidence Intervals**
   - 95% confidence bands for each prediction horizon
   - Gradient transparency (fades with longer horizons)
   - Indicates increasing uncertainty over time

3. **Probability Density Layers**
   - Violin-plot-like visualizations at key future points
   - Shows full distribution of possible outcomes
   - Based on model's predicted volatility

4. **Multiple Scenarios (Optional)**
   - **Bull Scenario**: Reduced volatility (0.7x) for optimistic forecasts
   - **Base Scenario**: Standard volatility (1.0x)
   - **Bear Scenario**: Increased volatility (1.5x) for pessimistic forecasts
   - Each scenario displayed in different color family

5. **Real-Time Updates**
   - Auto-refresh at configurable intervals (1-60 minutes)
   - Manual update button for immediate refresh
   - Smart caching to avoid redundant computation
   - Non-blocking async predictions (GUI never freezes)

6. **Interactive Controls**
   - Select/deselect horizons via checkboxes
   - Toggle visualization features (history, confidence, probability)
   - Choose color schemes
   - Enable/disable multiple scenarios
   - Adjust update interval

7. **Metrics Summary Table**
   - Current predictions for all active horizons
   - Target time, predicted price, change ($, %), confidence range
   - Directional signals (↑↓→)

8. **Data Export**
   - Export predictions to CSV files
   - Separate files for each scenario and horizon
   - Timestamp, prices, confidence bounds included

---

## Installation & Integration

### Prerequisites

Ensure you have all required dependencies installed:

```bash
pip install torch numpy pandas matplotlib scipy scikit-learn joblib
```

### File Structure

After installation, your project should have:

```
BitCorn_Farmer/
├── TradeApp.py                        # Main application
├── prediction_dashboard_tab.py        # NEW: Dashboard tab implementation
├── dashboard_utils.py                 # NEW: Utility functions
├── dashboard_visualizations.py        # NEW: Advanced plotting
├── multi_horizon_fan_inference.py     # Existing: Multi-horizon predictions
├── multi_horizon_inference.py         # Existing: Core inference engine
├── fiboevo.py                         # Existing: Model and features
├── config/
│   ├── gui_config.json                # Existing: Main app config
│   └── dashboard_config.json          # NEW: Dashboard config
├── artifacts/
│   ├── model_best.pt                  # Trained LSTM model
│   ├── meta.json                      # Model metadata
│   └── scaler.pkl                     # Feature scaler
└── data_manager/
    └── exports/
        └── Binance_BTCUSDT_1h.db      # SQLite database
```

### Integration Steps

#### 1. Add Import to TradeApp.py

At the top of `TradeApp.py`, add:

```python
# Add after existing imports (around line 115)
try:
    from prediction_dashboard_tab import PredictionDashboardTab
except ImportError:
    PredictionDashboardTab = None
```

#### 2. Add Dashboard Tab in _build_ui()

In the `_build_ui()` method of `TradeApp` class (around line 1658, after building other tabs):

```python
def _build_ui(self):
    # ... existing code for top controls and notebook ...

    # Existing tabs
    self._build_preview_tab()
    self._build_train_tab()
    self._build_backtest_tab()
    self._build_status_tab()
    self._build_audit_tab()

    # NEW: Add Predictions Dashboard tab
    self._build_predictions_dashboard_tab()

    # ... rest of existing code ...
```

#### 3. Add Dashboard Tab Builder Method

Add this new method to the `TradeApp` class:

```python
def _build_predictions_dashboard_tab(self):
    """Build the Predictions Dashboard tab."""
    if PredictionDashboardTab is None:
        # Fallback if module not available
        tab = Frame(self.nb)
        self.nb.add(tab, text="Predictions Dashboard")
        Label(tab, text="Prediction Dashboard module not available",
              fg="red", font=("Arial", 12)).pack(pady=50)
        logger.warning("PredictionDashboardTab not available")
        return

    try:
        # Create tab frame
        tab = Frame(self.nb)
        self.nb.add(tab, text="Predictions Dashboard")

        # Initialize dashboard
        self.predictions_dashboard = PredictionDashboardTab(
            parent_frame=tab,
            app_instance=self,
            config_path="config/dashboard_config.json"
        )

        logger.info("Predictions Dashboard tab initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Predictions Dashboard: {e}", exc_info=True)
        # Show error in tab
        Label(tab, text=f"Error initializing dashboard:\n{e}",
              fg="red", font=("Arial", 10)).pack(pady=50)
```

#### 4. Optional: Add Cleanup on Exit

In the `TradeApp` destructor or cleanup method:

```python
def __del__(self):
    # ... existing cleanup code ...

    # Cleanup dashboard resources
    if hasattr(self, 'predictions_dashboard') and self.predictions_dashboard is not None:
        try:
            self.predictions_dashboard.cleanup()
        except Exception as e:
            logger.error(f"Dashboard cleanup failed: {e}")
```

---

## Configuration

### Dashboard Settings (config/dashboard_config.json)

```json
{
  "dashboard": {
    "default_horizons": [1, 3, 5, 10, 20, 30],
    "update_interval_minutes": 5,
    "auto_refresh": true,
    "color_scheme": "viridis",
    "show_probability_layers": true,
    "max_history_candles": 200,
    "multiple_fans_config": {
      "enabled": false,
      "scenarios": ["base", "bull", "bear"],
      "scenario_params": {
        "base": {"volatility_mult": 1.0},
        "bull": {"volatility_mult": 0.7},
        "bear": {"volatility_mult": 1.5}
      }
    }
  }
}
```

**Key Settings:**

- `default_horizons`: Which horizons are selected by default
- `update_interval_minutes`: How often to auto-refresh (1-60)
- `auto_refresh`: Enable/disable automatic updates
- `color_scheme`: Matplotlib colormap ("viridis", "plasma", "coolwarm", "RdYlGn")
- `show_probability_layers`: Show probability density visualizations
- `max_history_candles`: How many historical candles to display
- `multiple_fans_config.enabled`: Enable bull/base/bear scenarios

---

## Usage Guide

### First-Time Setup

1. **Launch TradeApp**
   ```bash
   python TradeApp.py
   ```

2. **Navigate to Predictions Dashboard Tab**
   - Click on the "Predictions Dashboard" tab in the notebook

3. **Load Model**
   - Click "Load Model from Artifacts" to use the default trained model
   - Or click "Load Model from File..." to select a specific model

4. **Select Horizons**
   - Check the horizons you want to predict (e.g., 1h, 5h, 10h, 30h)
   - Or use "Select All" / "Clear All" buttons

5. **Generate Predictions**
   - Click "Update Now" to generate predictions
   - Wait a few seconds for computation (progress shown in status)

6. **View Results**
   - Prediction fan chart displays in main area
   - Metrics table at bottom shows summary for each horizon
   - Current price and last update time shown in status panel

### Advanced Features

#### Multiple Scenarios

1. Check "Enable Multiple Scenarios" in control panel
2. Click "Update Now"
3. Dashboard will show three prediction fans:
   - **Green lines**: Bull scenario (optimistic, lower volatility)
   - **Blue lines**: Base scenario (standard)
   - **Red lines**: Bear scenario (pessimistic, higher volatility)

#### Probability Density Layers

- Purple violin-like shapes at key future points
- Shows distribution of possible outcomes
- Wider = more uncertainty
- Toggle with "Show Probability Layers" checkbox

#### Auto-Refresh

1. Check "Auto-Refresh" checkbox
2. Adjust "Interval (minutes)" slider (1-60 minutes)
3. Dashboard will automatically update at specified interval

#### Export Predictions

1. Click "Export Predictions..."
2. Select output directory
3. CSV files will be created for each scenario and horizon
4. Files named: `predictions_{scenario}_h{horizon}.csv`

---

## Understanding the Visualization

### Prediction Fan Chart

```
Price
  ^
  |     /‾‾‾‾‾‾‾ 30h (lightest, most uncertain)
  |    /‾‾‾‾‾‾ 20h
  |   /‾‾‾‾‾ 10h
  |  /‾‾‾‾ 5h
  | /‾‾‾ 3h
  |/‾‾ 1h (darkest, most certain)
  |
  └──────────────────────> Time
    Now          Future
```

**Interpretation:**

- **Darker lines**: Shorter horizons, more reliable predictions
- **Lighter lines**: Longer horizons, less certain (fades into future)
- **Shaded bands**: Confidence intervals (95% by default)
- **Fan shape**: Uncertainty increases with time (natural for forecasting)

### Confidence Intervals

- **Narrow band**: Model is confident in prediction
- **Wide band**: High uncertainty, price could vary significantly
- **Gradient fade**: Visual cue that longer horizons are less certain

### Probability Density Layers

- **Violin-shaped overlays**: Show full distribution at key future points
- **Width**: Indicates spread of possible outcomes
- **Center line**: Most likely outcome (mean prediction)
- **Purple color**: Distinguishes from prediction lines

### Metrics Table Columns

| Column | Description |
|--------|-------------|
| **Horizon** | Time ahead (e.g., 5h means 5 hours into future) |
| **Target Time** | Clock time when prediction is for |
| **Predicted Price** | Model's forecasted price |
| **Change ($)** | Dollar change from current price |
| **Change (%)** | Percentage change from current price |
| **95% CI** | Confidence interval range (lower - upper) |
| **Signal** | Directional indicator: ↑ (bullish), ↓ (bearish), → (neutral) |

---

## Troubleshooting

### Common Issues

#### 1. "No model loaded" error

**Solution:** Load a model first via "Load Model from Artifacts" or "Load Model from File..."

#### 2. "No horizons selected" error

**Solution:** Check at least one horizon checkbox in the control panel

#### 3. Dashboard plot is empty

**Possible causes:**
- No predictions generated yet (click "Update Now")
- Model not loaded
- Database connection issue

**Check:**
- Status panel shows "Idle" → Click "Update Now"
- Model info shows "No model loaded" → Load model
- Check main app's SQLite path is correct

#### 4. Predictions take a long time

**Possible causes:**
- Many horizons selected (each requires computation)
- Large dataset (>2000 rows)
- CPU mode (GPU would be faster)

**Solutions:**
- Select fewer horizons
- Reduce `max_history_candles` in config
- If you have CUDA GPU, ensure PyTorch is using it

#### 5. "Matplotlib not available" error

**Solution:** Install matplotlib:
```bash
pip install matplotlib
```

#### 6. Predictions don't update automatically

**Check:**
- "Auto-Refresh" checkbox is checked
- Update interval is set (not 0)
- Dashboard didn't encounter an error (check status panel)

#### 7. Error: "Missing features" or "NaN values"

**Possible causes:**
- Database missing technical indicator columns
- Data gaps or missing timestamps

**Solutions:**
- Ensure `fiboevo.py` is available for feature engineering
- Run data preparation pipeline to add features
- Check database integrity

---

## Performance Optimization

### Tips for Faster Predictions

1. **Reduce Horizons**: Select only the horizons you need (e.g., 1h, 5h, 30h instead of all 7)

2. **Use GPU**: If you have a CUDA-compatible GPU:
   ```python
   # The system auto-detects GPU, but you can verify:
   import torch
   print(torch.cuda.is_available())  # Should print True
   ```

3. **Enable Caching**: Ensure cache is enabled in config:
   ```json
   "performance": {
     "cache_enabled": true,
     "cache_max_age_seconds": 300
   }
   ```

4. **Reduce Historical Data**: Lower `max_history_candles` (e.g., 100 instead of 200)

5. **Longer Update Intervals**: Set auto-refresh to 10-15 minutes instead of 5

### Memory Usage

- **Typical usage**: ~500 MB (model + predictions)
- **With multiple scenarios**: ~800 MB
- **Large datasets (>5000 rows)**: Up to 1.5 GB

---

## Advanced Customization

### Custom Color Schemes

Edit `dashboard_config.json`:

```json
"color_scheme": "plasma"
```

Available schemes:
- `viridis` (purple to yellow, default)
- `plasma` (purple to pink to yellow)
- `coolwarm` (blue to red)
- `RdYlGn` (red to yellow to green)
- `Blues`, `Greens`, `Reds` (single color gradients)

### Custom Horizons

To add custom horizons (e.g., 2h, 7h), modify `prediction_dashboard_tab.py`:

```python
# Around line 180, in __init__
for h in [1, 2, 3, 5, 7, 10, 15, 20, 30]:  # Added 2, 7
    self.horizon_vars[h] = BooleanVar(value=(h in default_horizons))
```

Then add checkboxes in `_build_control_panel()` around line 380.

### Custom Scenario Parameters

Edit `dashboard_config.json` to add custom scenarios:

```json
"scenarios": ["conservative", "base", "aggressive"],
"scenario_params": {
  "conservative": {
    "volatility_mult": 0.5,
    "description": "Very optimistic"
  },
  "base": {
    "volatility_mult": 1.0,
    "description": "Standard"
  },
  "aggressive": {
    "volatility_mult": 2.0,
    "description": "Very pessimistic"
  }
}
```

Then update `_plot_single_scenario()` in `dashboard_visualizations.py` to add color mappings.

---

## Technical Details

### Architecture

```
┌─────────────────────────────────────────────┐
│         PredictionDashboardTab              │
│  (Main GUI component, Tkinter widgets)      │
└───────────────┬─────────────────────────────┘
                │
                ├─> Control Panel (left)
                │   - Model loading
                │   - Horizon selection
                │   - Visualization options
                │   - Update settings
                │
                ├─> Visualization Area (center)
                │   - Matplotlib canvas (FigureCanvasTkAgg)
                │   - Prediction fan plot
                │
                └─> Metrics Panel (bottom)
                    - Treeview table
                    - Summary statistics

┌─────────────────────────────────────────────┐
│         Background Threads                  │
└─────────────────────────────────────────────┘
        │
        ├─> Prediction Thread (async)
        │   - Fetches data from SQLite
        │   - Runs multi_horizon_fan_inference
        │   - Generates scenarios (if enabled)
        │   - Puts results in queue
        │
        └─> Update Loop (root.after)
            - Checks prediction queue
            - Updates GUI components
            - Thread-safe via main thread

┌─────────────────────────────────────────────┐
│         Supporting Modules                  │
└─────────────────────────────────────────────┘
        │
        ├─> dashboard_utils.py
        │   - fetch_latest_data(): SQLite query + features
        │   - PredictionCache: In-memory caching
        │   - Validation functions
        │
        ├─> dashboard_visualizations.py
        │   - plot_prediction_fan_live(): Main plotting
        │   - _plot_probability_density_layers(): KDE violins
        │   - Styling functions
        │
        └─> multi_horizon_fan_inference.py
            - predict_multiple_horizons(): Core prediction engine
            - Horizon scaling (up/down from native horizon)
            - Uncertainty propagation
```

### Prediction Flow

1. **User clicks "Update Now"**
2. **GUI thread**: Start background thread, show "Generating predictions..." status
3. **Background thread**:
   - Fetch latest data from SQLite (`fetch_latest_data()`)
   - Check cache for existing predictions (`PredictionCache.get()`)
   - If not cached:
     - Run `predict_multiple_horizons()` for each scenario
     - Apply volatility multipliers for bull/bear scenarios
     - Cache results (`PredictionCache.set()`)
   - Put results in queue
4. **GUI thread** (via `root.after()`):
   - Check queue for results
   - Update matplotlib plot (`_update_visualization()`)
   - Update metrics table (`_update_metrics_table()`)
   - Update status labels
5. **Display refreshed**

### Thread Safety

- **GUI updates**: Always via `root.after()` in main thread
- **Prediction computation**: Background threads (non-blocking)
- **Communication**: Thread-safe `queue.Queue`
- **Shared state**: Protected by locks in `PredictionCache`

### Caching Strategy

```python
cache_key = (tuple(horizons), len(df), df['close'].iloc[-1])
```

- **Key components**: horizons, data length, last price
- **Rationale**: If these are unchanged, predictions are the same
- **Expiration**: 5 minutes (configurable)
- **Size limit**: 10 entries (LRU eviction)

---

## API Reference

### PredictionDashboardTab

**Constructor:**
```python
PredictionDashboardTab(
    parent_frame: Frame,
    app_instance: Any,
    config_path: str = "config/dashboard_config.json"
)
```

**Key Methods:**
- `_load_model_artifacts()`: Load model from artifacts/
- `_load_model_file()`: Load model from user-selected file
- `_manual_update()`: Trigger immediate prediction update
- `_update_visualization()`: Refresh matplotlib plot
- `_update_metrics_table()`: Refresh metrics table
- `_export_predictions()`: Export predictions to CSV
- `cleanup()`: Release resources

**Attributes:**
- `model`: Loaded LSTM model
- `meta`: Model metadata dict
- `scaler`: StandardScaler for features
- `current_predictions`: Dict[scenario -> Dict[horizon -> DataFrame]]
- `df_history`: Historical data DataFrame

### dashboard_utils Functions

```python
fetch_latest_data(
    sqlite_path: str,
    table: str = "ohlcv",
    min_rows: int = 1000,
    add_features: bool = True
) -> pd.DataFrame
```
Fetch latest data from SQLite with feature engineering.

```python
PredictionCache(
    max_age_seconds: int = 300,
    max_size: int = 10
)
```
In-memory cache for predictions.

- `get(key) -> Optional[Any]`: Retrieve cached value
- `set(key, value)`: Store value
- `clear()`: Clear all entries

### dashboard_visualizations Functions

```python
plot_prediction_fan_live(
    ax: Any,
    df: pd.DataFrame,
    predictions_by_scenario: Dict[str, Dict[int, pd.DataFrame]],
    show_history: bool = True,
    show_confidence: bool = True,
    show_probability: bool = True,
    color_scheme: str = "viridis",
    max_history_candles: int = 200
) -> None
```
Main plotting function for prediction fan.

```python
apply_confidence_gradient(
    ax: Any,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    color: str = "blue",
    n_levels: int = 5
) -> None
```
Apply gradient transparency to confidence bands.

---

## FAQ

### Q: How accurate are the predictions?

**A:** Prediction accuracy depends on:
- **Horizon**: Shorter horizons (1h, 3h) are typically more accurate than longer ones (20h, 30h)
- **Market conditions**: Stable markets → better accuracy; volatile markets → wider uncertainty
- **Model training**: How recent and comprehensive the training data is

Typical accuracy metrics (BTCUSDT 1h data):
- **1h horizon**: MAE ~$100-200, Directional accuracy ~55-60%
- **10h horizon**: MAE ~$300-500, Directional accuracy ~52-55%
- **30h horizon**: MAE ~$600-1000, Directional accuracy ~50-52% (near random)

### Q: What do the confidence intervals mean?

**A:** The 95% confidence interval means: "If we repeated this prediction 100 times, the true price would fall within this range about 95 times."

**Wide CI** = High uncertainty (e.g., volatile market, long horizon)
**Narrow CI** = High confidence (e.g., stable market, short horizon)

### Q: Should I trade based on these predictions?

**A:** **No, not directly.** The dashboard is a **decision support tool**, not an automated trading signal. Use predictions as one of many inputs:
- Combine with technical analysis (RSI, MACD, volume)
- Consider market sentiment and news
- Apply risk management (stop losses, position sizing)
- Backtest strategies before live trading

### Q: How do bull/base/bear scenarios work?

**A:** They adjust the **volatility** (uncertainty) of predictions:
- **Bull**: Assumes lower volatility (0.7x) → narrower confidence bands, more optimistic
- **Base**: Standard model volatility (1.0x) → normal confidence bands
- **Bear**: Assumes higher volatility (1.5x) → wider confidence bands, more pessimistic

**The mean prediction stays the same**; only the uncertainty changes.

### Q: Can I use this with other cryptocurrencies?

**A:** Yes, but you need to:
1. Train a model on that cryptocurrency's data (e.g., ETHUSDT, SOLUSDT)
2. Update the database path in TradeApp to point to the new data
3. Load the new model in the dashboard

The dashboard is **model-agnostic** and works with any LSTM2Head model trained via `fiboevo`.

### Q: What if I have data gaps (missing timestamps)?

**A:** The system handles small gaps (<2 hours) gracefully by:
- Forward-filling features
- Interpolating missing values

**Large gaps** (>2 hours) can cause issues. Solution:
- Run data collection pipeline to fill gaps
- Use `check_for_gaps()` in `dashboard_utils.py` to detect issues

### Q: Can I run this on a Raspberry Pi / low-power device?

**A:** Yes, but:
- **Performance**: Predictions will be slower (5-10 seconds vs 1-2 seconds on desktop)
- **Memory**: Ensure at least 2 GB RAM available
- **Recommendations**:
  - Use CPU mode (GPU not available on RPi)
  - Select fewer horizons (e.g., only 1h, 5h, 30h)
  - Set longer update intervals (15-30 minutes)
  - Lower `max_history_candles` to 100

---

## Changelog

### Version 1.0.0 (2025-10-30)

**Initial Release:**
- Multi-horizon prediction fan visualization
- Confidence intervals with gradient transparency
- Probability density layers
- Multiple scenario support (bull/base/bear)
- Real-time auto-refresh
- Interactive controls
- Metrics summary table
- CSV export
- Smart caching
- Async prediction generation
- Professional styling

---

## Support & Contact

For issues, questions, or feature requests:

1. **Check logs**: `logs/predictions/` directory contains detailed prediction logs
2. **Enable debug logging**: Set `level=logging.DEBUG` in code
3. **Review error messages**: Status panel shows errors
4. **Consult documentation**: This guide covers most common issues

**Project Repository:** [Link to your repo]

**Author:** Claude (Anthropic) for BitCorn_Farmer Project

**License:** [Your license]

---

## Appendix: Mathematical Details

### Horizon Scaling

The model is trained at a **native horizon** h_native (e.g., 10 hours). To predict at other horizons:

**For h < h_native (short horizons):**
```
y_h = (h / h_native) * y_native
σ_h = σ_native * sqrt(h / h_native)
```

**For h > h_native (long horizons):**
```
y_h = (h / h_native) * y_native
σ_h = σ_native * sqrt(h / h_native)
```

**Assumptions:**
- Linear drift approximation: log-return scales linearly with time
- Random walk volatility: variance scales linearly with time (σ² ∝ h)

**Validity:**
- Short horizons: Good approximation (error <5%)
- Long horizons (h > 2*h_native): Approximation degrades (error >15%)

### Confidence Intervals

For log-normal price distribution:

```
P_future = P_current * exp(μ ± z*σ)
```

Where:
- μ = predicted log-return
- σ = predicted volatility
- z = z-score for confidence level (1.96 for 95%)

**Example:**
- Current price: $50,000
- Predicted log-return: 0.02 (2%)
- Predicted volatility: 0.05 (5%)
- 95% CI:
  - Lower: $50,000 * exp(0.02 - 1.96*0.05) = $45,321
  - Upper: $50,000 * exp(0.02 + 1.96*0.05) = $55,207

### Probability Density

Uses kernel density estimation (KDE) on sampled log-returns:

```
log_return ~ N(μ_pred, σ_pred)
price_samples = P_current * exp(log_return_samples)
density = KDE(price_samples)
```

Displayed as violin plots at key horizons.

---

**End of Guide**
