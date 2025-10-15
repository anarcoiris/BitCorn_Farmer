# Multi-Horizon LSTM Inference System

## Overview

This module provides a rigorous implementation of multi-horizon forecasting for LSTM models trained on financial time series, with **proper inverse transformation** from model outputs (log-returns) back to original price scale.

**Key Features:**
- Two prediction modes: Jump forecasting and Autoregressive forecasting
- Mathematically correct log-return to price conversion
- Confidence interval estimation using predicted volatility
- Comprehensive visualization tools
- Prevention of data leakage and look-ahead bias
- Full support for StandardScaler feature normalization

---

## Mathematical Framework

### Model Architecture

The trained LSTM2Head model predicts two quantities at each timestep:
1. **Log-return**: $\hat{y}_t = \log(P_{t+h}) - \log(P_t) = \log(P_{t+h}/P_t)$
2. **Volatility**: $\hat{\sigma}_t$ (standard deviation of returns over prediction horizon)

where:
- $P_t$ is the close price at time $t$
- $h$ is the prediction horizon (e.g., 12 steps)
- $\hat{y}_t$ is the predicted log-return
- $\hat{\sigma}_t$ is the predicted volatility (uncertainty)

### Inverse Transformation: Log-Returns → Prices

Given a predicted log-return $\hat{y}_t$, we recover the predicted price:

$$P_{t+h} = P_t \cdot \exp(\hat{y}_t)$$

**Why this works:**
- From $\hat{y}_t = \log(P_{t+h}/P_t)$
- Apply exponential: $\exp(\hat{y}_t) = P_{t+h}/P_t$
- Solve for $P_{t+h}$: $P_{t+h} = P_t \cdot \exp(\hat{y}_t)$

### Confidence Intervals

Using the predicted volatility $\hat{\sigma}_t$, we construct confidence intervals:

**95% Confidence Interval (±2σ):**
$$\text{Upper bound} = P_t \cdot \exp(\hat{y}_t + 2\hat{\sigma}_t)$$
$$\text{Lower bound} = P_t \cdot \exp(\hat{y}_t - 2\hat{\sigma}_t)$$

This assumes log-returns are approximately normally distributed, which is a reasonable first-order approximation for short horizons.

### Multi-Step Forecasting: Two Approaches

#### 1. Jump Forecasting (Recommended)

**Method:** Slide a fixed-size window through historical data, predicting $h$ steps ahead at each position.

**Process:**
1. At time $t$, extract window $[t-s, t-1]$ where $s$ is sequence length
2. Feed to model → predict log-return for time $t+h$
3. Convert to price: $P_{t+h} = P_{t-1} \cdot \exp(\hat{y}_t)$
4. Move to time $t+1$, repeat

**Advantages:**
- Uses actual historical data for features
- No feature recomputation required
- Prediction quality remains stable
- Natural for backtesting

**Limitations:**
- Requires historical data up to prediction point
- Cannot forecast beyond available data
- Each prediction is independent (not truly autoregressive)

**When to use:** This is the **default and recommended** method for:
- Model evaluation and backtesting
- Performance analysis with ground truth
- Scenarios where historical data is available

#### 2. Autoregressive Forecasting (Experimental)

**Method:** Use model's own predictions to generate future inputs for subsequent predictions.

**Process:**
1. Start with window $[t-s, t-1]$ from actual data
2. Predict $\hat{P}_{t+h}$
3. Append $\hat{P}_{t+h}$ to history
4. Recompute features using synthetic price history
5. Predict $\hat{P}_{t+2h}$, repeat...

**Advantages:**
- Can forecast arbitrarily far into future
- Truly generative
- Useful for scenario analysis

**Limitations:**
- Feature recomputation is complex (requires full fiboevo pipeline)
- Errors compound rapidly
- Prediction quality degrades after 2-3 horizons
- Uncertainty grows exponentially

**When to use:** Only when:
- No historical data available beyond current point
- Long-term scenario analysis needed
- Accepting degraded accuracy for distant predictions

---

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch numpy pandas scikit-learn joblib

# Visualization (optional but recommended)
pip install matplotlib

# Data I/O (optional)
pip install pyarrow  # for parquet support
```

### Project Setup

The multi-horizon inference system requires:
1. Trained LSTM2Head model (.pt file)
2. Model metadata (meta.json)
3. Fitted StandardScaler (scaler.pkl)
4. Feature data with computed technical indicators

All should be available if you've trained a model using `fiboevo.py` and `prepare_dataset.py`.

---

## Usage

### Quick Start

```python
from multi_horizon_inference import (
    load_model_and_artifacts,
    predict_multi_horizon_jump,
    plot_predictions
)
import pandas as pd

# 1. Load model
model, meta, scaler, device = load_model_and_artifacts(
    model_path="artifacts/model_best.pt",
    meta_path="artifacts/meta.json",
    scaler_path="artifacts/scaler.pkl"
)

# 2. Load feature data
df = pd.read_parquet("data_manager/exports/features/binance_BTCUSDT_30m_features_v1.parquet")

# 3. Generate predictions
predictions_df = predict_multi_horizon_jump(
    df=df,
    model=model,
    meta=meta,
    scaler=scaler,
    device=device,
    n_predictions=500,
    start_idx=None  # Auto-detect
)

# 4. Visualize
fig = plot_predictions(df, predictions_df, save_path="predictions.png")

# 5. Analyze results
print(predictions_df[["timestamp", "close_pred", "prediction_error"]].head())
```

### Command-Line Interface

```bash
# Basic usage
python multi_horizon_inference.py \
    --model artifacts/model_best.pt \
    --meta artifacts/meta.json \
    --scaler artifacts/scaler.pkl \
    --data data_manager/exports/features/binance_BTCUSDT_30m_features_v1.parquet \
    --n-predictions 500 \
    --output predictions.csv \
    --plot \
    --plot-output predictions_plot.png

# Full example with custom settings
python multi_horizon_inference.py \
    --model artifacts/model_best.pt \
    --meta artifacts/meta.json \
    --scaler artifacts/scaler.pkl \
    --data data_manager/exports/features/binance_BTCUSDT_30m_features_v1.parquet \
    --n-predictions 1000 \
    --start-idx 1000 \
    --output results/predictions.csv \
    --plot \
    --plot-output results/predictions_plot.png \
    --device cuda
```

### Run Example Script

```bash
# Run complete example with default settings
python example_multi_horizon.py
```

This will:
1. Load the trained model from `artifacts/`
2. Load feature data
3. Generate 500 predictions
4. Save results to `predictions_output.csv`
5. Generate visualization plots
6. Display summary statistics

---

## API Reference

### `load_model_and_artifacts()`

Load trained model and preprocessing artifacts.

**Parameters:**
- `model_path` (str): Path to .pt model checkpoint
- `meta_path` (str): Path to meta.json with feature_cols, seq_len, horizon, etc.
- `scaler_path` (str): Path to fitted StandardScaler (.pkl)
- `device` (str, optional): Device to load model on ('cuda', 'cpu'). Auto-detect if None.

**Returns:**
- `model` (nn.Module): Loaded LSTM2Head model in eval mode
- `meta` (dict): Model metadata dictionary
- `scaler` (StandardScaler): Fitted scaler for features
- `device` (torch.device): Device model is loaded on

**Example:**
```python
model, meta, scaler, device = load_model_and_artifacts(
    "artifacts/model_best.pt",
    "artifacts/meta.json",
    "artifacts/scaler.pkl",
    device="cuda"
)
```

---

### `predict_multi_horizon_jump()`

Generate multi-horizon predictions using jump forecasting method.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with OHLCV and features. Must contain:
  - `close`: Close prices (required)
  - `timestamp`: Timestamps (optional but recommended)
  - All features listed in `meta['feature_cols']`
- `model` (nn.Module): Trained LSTM2Head model
- `meta` (dict): Model metadata
- `scaler` (StandardScaler): Fitted scaler
- `device` (torch.device): Device for inference
- `n_predictions` (int): Number of predictions to generate
- `start_idx` (int, optional): Starting index in df. If None, uses `seq_len` as minimum.
- `return_features` (bool): Include input features in output (default: False)

**Returns:**
- `predictions_df` (pd.DataFrame): DataFrame with columns:
  - `index`: Original dataframe index of prediction point
  - `timestamp`: Timestamp of prediction point
  - `close_current`: Current close price (P_t)
  - `close_actual_future`: Actual future close price (P_{t+h}) for validation
  - `close_pred`: Predicted close price (P_{t+h})
  - `log_return_pred`: Predicted log-return
  - `volatility_pred`: Predicted volatility (σ)
  - `upper_bound_2std`: Upper 95% confidence bound
  - `lower_bound_2std`: Lower 95% confidence bound
  - `horizon_steps`: Prediction horizon (h)
  - `prediction_error`: Actual - Predicted (where available)
  - `prediction_error_pct`: Percentage error

**Example:**
```python
predictions = predict_multi_horizon_jump(
    df=feature_df,
    model=model,
    meta=meta,
    scaler=scaler,
    device=device,
    n_predictions=500,
    start_idx=1000  # Start predictions from row 1000
)

# Access predictions
print(f"Mean predicted price: ${predictions['close_pred'].mean():.2f}")
print(f"MAE: ${predictions['prediction_error'].abs().mean():.2f}")
```

---

### `predict_autoregressive()`

Generate autoregressive multi-step predictions (experimental).

⚠️ **WARNING:** This method is experimental. Prediction quality degrades rapidly beyond 2-3 horizons due to:
- Feature recomputation complexity
- Error compounding
- Departure from training distribution

**Parameters:**
- `df` (pd.DataFrame): DataFrame with features (uses last `seq_len` rows as seed)
- `model` (nn.Module): Trained model
- `meta` (dict): Model metadata
- `scaler` (StandardScaler): Fitted scaler
- `device` (torch.device): Device
- `n_steps` (int): Number of future steps to predict
- `use_actual_for_features` (bool): Use actual prices for features if available (more reliable)

**Returns:**
- `predictions_df` (pd.DataFrame): DataFrame with predictions

**Example:**
```python
# Predict 50 steps into future (beyond available data)
ar_predictions = predict_autoregressive(
    df=feature_df,
    model=model,
    meta=meta,
    scaler=scaler,
    device=device,
    n_steps=50,
    use_actual_for_features=True
)
```

---

### `plot_predictions()`

Visualize historical prices with prediction overlay.

**Parameters:**
- `df` (pd.DataFrame): Historical data with 'close' and optionally 'timestamp'
- `predictions_df` (pd.DataFrame): Output from `predict_multi_horizon_jump()`
- `title` (str): Plot title
- `figsize` (tuple): Figure size (width, height) in inches
- `show_confidence` (bool): Show confidence interval bands
- `save_path` (str, optional): Path to save figure

**Returns:**
- `fig` (matplotlib.Figure): Figure object (or None if matplotlib unavailable)

**Example:**
```python
fig = plot_predictions(
    df=feature_df,
    predictions_df=predictions,
    title="BTC/USDT 30m Multi-Horizon Predictions (h=12)",
    figsize=(16, 9),
    show_confidence=True,
    save_path="predictions_plot.png"
)
```

---

### `plot_prediction_errors()`

Visualize prediction errors over time.

**Parameters:**
- `predictions_df` (pd.DataFrame): Must contain 'prediction_error' column
- `title` (str): Plot title
- `figsize` (tuple): Figure size
- `save_path` (str, optional): Save path

**Returns:**
- `fig` (matplotlib.Figure): Figure with two subplots (absolute and percentage errors)

**Example:**
```python
fig = plot_prediction_errors(
    predictions_df=predictions,
    title="Prediction Error Analysis",
    save_path="error_plot.png"
)
```

---

## Understanding the Output

### Predictions DataFrame

After running `predict_multi_horizon_jump()`, you get a DataFrame with these columns:

| Column | Description | Units |
|--------|-------------|-------|
| `timestamp` | Time of prediction point | datetime |
| `close_current` | Price at prediction time (P_t) | $ |
| `close_actual_future` | Actual price at t+h (for validation) | $ |
| `close_pred` | Predicted price at t+h | $ |
| `log_return_pred` | Predicted log-return | dimensionless |
| `volatility_pred` | Predicted volatility (std of returns) | dimensionless |
| `upper_bound_2std` | 95% CI upper bound | $ |
| `lower_bound_2std` | 95% CI lower bound | $ |
| `prediction_error` | Actual - Predicted | $ |
| `prediction_error_pct` | Percentage error | % |

### Key Metrics

**Mean Absolute Error (MAE):**
```python
mae = predictions_df["prediction_error"].abs().mean()
```
Average absolute dollar error. Lower is better.

**Mean Absolute Percentage Error (MAPE):**
```python
mape = predictions_df["prediction_error_pct"].abs().mean()
```
Average percentage error. Lower is better. Less than 1% is excellent for financial data.

**Root Mean Squared Error (RMSE):**
```python
rmse = np.sqrt((predictions_df["prediction_error"] ** 2).mean())
```
Penalizes large errors more heavily than MAE.

**Directional Accuracy:**
```python
predictions_df["predicted_direction"] = (
    predictions_df["close_pred"] > predictions_df["close_current"]
)
predictions_df["actual_direction"] = (
    predictions_df["close_actual_future"] > predictions_df["close_current"]
)
directional_accuracy = (
    predictions_df["predicted_direction"] == predictions_df["actual_direction"]
).mean() * 100
```
Percentage of times the model correctly predicted up/down direction. Above 55% is good for financial markets.

**Confidence Interval Coverage:**
```python
within_ci = (
    (predictions_df["close_actual_future"] >= predictions_df["lower_bound_2std"]) &
    (predictions_df["close_actual_future"] <= predictions_df["upper_bound_2std"])
)
coverage = within_ci.mean() * 100
```
Percentage of actual prices falling within predicted 95% CI. Should be close to 95%. Much lower indicates overconfidence; much higher indicates underconfidence.

---

## Important Considerations

### 1. Data Leakage Prevention

The system ensures no data leakage by:
- Using `prepare_input_for_model()` which validates feature alignment
- Never using future information in feature computation
- Scaler fitted only on training data (not test data)
- Proper temporal splitting in dataset preparation

### 2. Feature Scaling

All features must be scaled using the **same scaler** that was fitted during training. The system handles this automatically:

```python
# Inside predict_multi_horizon_jump()
input_tensor = prepare_input_for_model(
    window_df,
    feature_cols,  # Exact order from training
    seq_len,
    scaler=scaler,  # Training scaler
    method="per_row"
)
```

### 3. Numerical Stability

The code uses `float64` for log computations to prevent numerical errors:

```python
# Avoid underflow/overflow
logc = np.log(close.astype(np.float64))  # Use float64
price_pred = price_current * np.exp(log_return_pred)  # Safe exponentiation
```

### 4. Handling Missing Data

The system expects:
- No NaN values in feature columns during inference
- Continuous time series (no gaps in timestamps)

If your data has gaps, preprocess it:

```python
# Option 1: Drop NaN rows
df_clean = df.dropna()

# Option 2: Forward-fill missing values (use with caution)
df_filled = df.fillna(method='ffill')

# Option 3: Interpolate (for continuous features)
df_interp = df.interpolate(method='linear')
```

### 5. Computational Efficiency

For large datasets (>10k predictions):

```python
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Process in batches (future enhancement)
# Currently processes one-by-one but could be batched
```

---

## Troubleshooting

### Error: "Feature mismatch. Missing: {...}, Extra: {...}"

**Cause:** Feature columns in your data don't match those expected by the model.

**Solution:**
1. Check `meta.json` to see what features the model expects
2. Ensure your data has all required features
3. Recompute features using `fiboevo.add_technical_features()`

```python
# Recompute features
from fiboevo import add_technical_features

df_with_features = add_technical_features(
    close=df["close"].values,
    high=df["high"].values,
    low=df["low"].values,
    volume=df["volume"].values,
    decouple_from_close=False  # Match training setting
)
```

### Error: "Not enough rows for seq_len"

**Cause:** DataFrame has fewer rows than the sequence length required.

**Solution:**
```python
# Check requirements
seq_len = meta["seq_len"]
horizon = meta["horizon"]
min_rows = seq_len + horizon

if len(df) < min_rows:
    print(f"Need at least {min_rows} rows, got {len(df)}")
    # Load more data or reduce n_predictions
```

### Warning: "Prediction quality degrades rapidly"

**Cause:** Using autoregressive mode for many steps.

**Solution:**
- Use jump forecasting instead (requires historical data)
- Limit AR predictions to 2-3 horizons maximum
- Retrain model with longer horizon if needed

### Poor Prediction Accuracy

**Possible causes:**
1. **Model overfitting:** Check validation loss during training
2. **Data distribution shift:** Market regime changed since training
3. **Feature drift:** Technical indicators behaving differently
4. **Insufficient training data:** Need more diverse examples
5. **Wrong timeframe:** Model trained on different data frequency

**Diagnostic steps:**
```python
# 1. Check prediction distribution
print(predictions_df["log_return_pred"].describe())

# 2. Plot predicted vs actual
import matplotlib.pyplot as plt
plt.scatter(
    predictions_df["close_actual_future"],
    predictions_df["close_pred"],
    alpha=0.3
)
plt.plot([df["close"].min(), df["close"].max()],
         [df["close"].min(), df["close"].max()],
         'r--', label='Perfect prediction')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.show()

# 3. Check residual autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(predictions_df["prediction_error"].dropna(), lags=10)
print(lb_test)  # p-value < 0.05 indicates autocorrelation (model missing signal)
```

---

## Advanced Usage

### Custom Confidence Intervals

Adjust confidence level:

```python
# 99% confidence interval (±3σ)
predictions_df["upper_99"] = (
    predictions_df["close_current"] *
    np.exp(predictions_df["log_return_pred"] + 3 * predictions_df["volatility_pred"])
)
predictions_df["lower_99"] = (
    predictions_df["close_current"] *
    np.exp(predictions_df["log_return_pred"] - 3 * predictions_df["volatility_pred"])
)
```

### Ensemble Predictions

Combine multiple models:

```python
models = []
for model_path in ["model1.pt", "model2.pt", "model3.pt"]:
    model, _, _, _ = load_model_and_artifacts(model_path, meta_path, scaler_path)
    models.append(model)

# Generate predictions from each
all_predictions = []
for model in models:
    preds = predict_multi_horizon_jump(df, model, meta, scaler, device, n_predictions)
    all_predictions.append(preds["close_pred"])

# Ensemble: simple average
ensemble_pred = np.mean(all_predictions, axis=0)

# Or: weighted by inverse validation error
weights = [0.4, 0.35, 0.25]  # Based on validation performance
ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
```

### Integration with Trading System

```python
# Get latest prediction
latest_pred = predict_multi_horizon_jump(
    df=df.tail(1000),  # Use recent data
    model=model,
    meta=meta,
    scaler=scaler,
    device=device,
    n_predictions=1
).iloc[0]

current_price = latest_pred["close_current"]
predicted_price = latest_pred["close_pred"]
confidence_width = latest_pred["upper_bound_2std"] - latest_pred["lower_bound_2std"]

# Trading decision logic
if predicted_price > current_price * 1.005:  # Expect >0.5% gain
    if confidence_width / current_price < 0.02:  # Confidence interval < 2%
        action = "BUY"
    else:
        action = "HOLD"  # Too uncertain
elif predicted_price < current_price * 0.995:  # Expect >0.5% loss
    action = "SELL"
else:
    action = "HOLD"

print(f"Current: ${current_price:.2f}, Predicted: ${predicted_price:.2f}")
print(f"Action: {action}")
```

---

## References

### Mathematical Background

1. **Log-returns in Finance:**
   - Campbell, J.Y., Lo, A.W., MacKinlay, A.C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.

2. **LSTM for Time Series:**
   - Hochreiter, S., Schmidhuber, J. (1997). "Long short-term memory". *Neural Computation*, 9(8), 1735-1780.

3. **Volatility Forecasting:**
   - Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation". *Econometrica*, 50(4), 987-1007.

4. **Multi-step Forecasting:**
   - Taieb, S.B., Bontempi, G., Atiya, A.F., Sorjamaa, A. (2012). "A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition". *Expert Systems with Applications*, 39(8), 7067-7083.

### Related Documentation

- [fiboevo.py](./fiboevo.py) - Core feature engineering and model training
- [prepare_dataset.py](./prepare_dataset.py) - Dataset preparation with proper scaling
- [DECOUPLE_FEATURES.md](./vault/DECOUPLE_FEATURES.md) - Feature engineering methodology

---

## License

This code is part of the cow_lvl-2oct trading system project.

## Author

Implementation by Claude (Anthropic), October 2025

## Support

For issues or questions:
1. Check this documentation first
2. Review example scripts
3. Examine the code comments in `multi_horizon_inference.py`
4. Open an issue with reproducible example
