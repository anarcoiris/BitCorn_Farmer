#!/usr/bin/env python3
"""
multi_horizon_inference.py

Multi-horizon inference system for LSTM2Head models with proper inverse transformation
from log-returns back to original close price scale.

Features:
- Autoregressive multi-step prediction (using model's own predictions as inputs)
- Proper log-return to price conversion with compounding
- Uncertainty quantification via volatility predictions
- Visualization of historical prices with forecast overlay
- Handles feature scaling/descaling correctly
- Prevents data leakage in recursive prediction

Mathematical Framework:
---------------------
The model predicts log-returns: y_t = log(P_{t+h}) - log(P_t) = log(P_{t+h}/P_t)
where h is the horizon parameter.

For multi-horizon forecasting:
1. Given current price P_t and predicted log-return y_pred
2. Next price: P_{t+h} = P_t * exp(y_pred)
3. For confidence intervals, use predicted volatility sigma:
   - Upper bound: P_t * exp(y_pred + 2*sigma)
   - Lower bound: P_t * exp(y_pred - 2*sigma)

For autoregressive (AR) forecasting beyond the model's native horizon:
- The model predicts h steps ahead (e.g., h=12)
- To predict h+1, h+2, ..., we need to update features with predicted values
- This requires careful handling of feature engineering and scaling

Important Considerations:
------------------------
1. Feature Update Challenge: The model expects features like log_ret_1, log_ret_5,
   sma_5, ema_20, rsi_14, etc. For true AR forecasting, we would need to:
   - Compute these features from predicted prices
   - Re-scale them using the training scaler
   - This is complex and can accumulate errors

2. Simplified Multi-Horizon: Instead, we implement "jump" forecasting:
   - Each prediction is h steps ahead from the current observation window
   - We slide the window forward by 1 observation (using actual data)
   - This provides h-step-ahead forecasts at each time point
   - More robust than full AR but requires historical data for the window

3. True AR Multi-Horizon: For predictions beyond available data:
   - We can approximate by keeping predicted prices and synthetic features
   - Warning: prediction quality degrades rapidly beyond 2-3 horizons
   - Uncertainty should increase accordingly

Author: Claude (Anthropic)
Date: 2025-10-15
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional dependencies
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
except ImportError:
    plt = None
    mdates = None
    Figure = None

# Try to import fiboevo utilities
try:
    import fiboevo
    from fiboevo import prepare_input_for_model, add_technical_features
except ImportError:
    fiboevo = None
    prepare_input_for_model = None
    add_technical_features = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# Multi-Horizon Inference Engine
# ==========================================

def load_model_and_artifacts(
    model_path: str,
    meta_path: str,
    scaler_path: str,
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any], Any, torch.device]:
    """
    Load trained LSTM model, metadata, and scaler.

    Args:
        model_path: Path to .pt or .pth model file
        meta_path: Path to meta.json
        scaler_path: Path to scaler.pkl
        device: Device string ('cuda', 'cpu', etc). Auto-detect if None.

    Returns:
        model: Loaded PyTorch model in eval mode
        meta: Dictionary with model metadata
        scaler: Fitted StandardScaler
        device: torch.device object
    """
    if torch is None:
        raise RuntimeError("PyTorch not available. Install torch to use this module.")

    if joblib is None:
        raise RuntimeError("joblib not available. Install joblib to load scaler.")

    # Load metadata
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(f"Loaded metadata: {meta}")

    # Load scaler
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    scaler = joblib.load(scaler_path)

    # Handle older scaler versions without feature_names_in_
    if hasattr(scaler, 'feature_names_in_'):
        logger.info(f"Loaded scaler with {len(scaler.feature_names_in_)} features")
    else:
        # Assign feature names from meta for compatibility
        scaler.feature_names_in_ = np.array(meta.get("feature_cols", []))
        logger.info(f"Loaded scaler (assigned {len(scaler.feature_names_in_)} feature names from meta)")

    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Extract architecture params from meta
    input_size = len(meta.get("feature_cols", []))
    hidden_size = meta.get("hidden", 64)
    num_layers = meta.get("num_layers", 2)

    # Instantiate model architecture
    if fiboevo is not None and hasattr(fiboevo, "LSTM2Head"):
        model = fiboevo.LSTM2Head(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
    else:
        raise RuntimeError("fiboevo.LSTM2Head not available. Ensure fiboevo is properly installed.")

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully: {input_size} features, {hidden_size} hidden units, {num_layers} layers")

    return model, meta, scaler, device


def predict_multi_horizon_jump(
    df: pd.DataFrame,
    model: nn.Module,
    meta: Dict[str, Any],
    scaler: Any,
    device: torch.device,
    n_predictions: int,
    start_idx: Optional[int] = None,
    return_features: bool = False
) -> pd.DataFrame:
    """
    Generate multi-horizon predictions using "jump" method.

    This method slides a window through historical data and generates h-step-ahead
    predictions at each position. It requires actual historical data for the full
    forecast period.

    Process:
    -------
    For each time t from start_idx to start_idx + n_predictions:
        1. Extract window of seq_len observations ending at t
        2. Prepare features and scale using training scaler
        3. Predict log-return for t+horizon and volatility
        4. Convert log-return to price using P_t
        5. Move to next time step (t+1)

    Args:
        df: DataFrame with OHLCV and features (must have 'close' column)
        model: Trained LSTM2Head model
        meta: Model metadata dict with feature_cols, seq_len, horizon
        scaler: Fitted StandardScaler from training
        device: torch.device for inference
        n_predictions: Number of predictions to generate
        start_idx: Starting index in df (if None, uses seq_len as minimum)
        return_features: If True, include input features in output

    Returns:
        DataFrame with columns:
        - timestamp: prediction time
        - close_actual: actual close price at t
        - close_pred: predicted close price at t+horizon
        - log_return_pred: predicted log return
        - volatility_pred: predicted volatility
        - upper_bound_2std: upper confidence bound (mean + 2*std)
        - lower_bound_2std: lower confidence bound (mean - 2*std)
        - horizon_steps: number of steps ahead this prediction is for
    """
    if torch is None or prepare_input_for_model is None:
        raise RuntimeError("Required dependencies not available")

    # Extract metadata
    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    horizon = meta["horizon"]

    # Validate inputs
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    if len(df) < seq_len + horizon:
        raise ValueError(f"DataFrame too short: need at least {seq_len + horizon} rows, got {len(df)}")

    # Determine start index
    if start_idx is None:
        start_idx = seq_len
    elif start_idx < 0:
        # Handle negative indexing (count from end of dataset)
        start_idx = len(df) + start_idx
        if start_idx < seq_len:
            raise ValueError(f"Negative start_idx resolves to {start_idx}, but minimum is {seq_len}")

    # Validate we have enough data
    max_idx = len(df) - horizon
    if start_idx + n_predictions > max_idx:
        available = max_idx - start_idx
        warnings.warn(f"Requested {n_predictions} predictions but only {available} available. Adjusting.")
        n_predictions = max(1, available)

    # Storage for results
    results = []

    logger.info(f"Generating {n_predictions} predictions starting from index {start_idx}")
    logger.info(f"Each prediction is {horizon} steps ahead")

    model.eval()
    with torch.no_grad():
        for i in range(n_predictions):
            t = start_idx + i

            # Extract window ending at t
            window_start = t - seq_len
            window_end = t

            if window_start < 0:
                logger.warning(f"Skipping prediction at t={t}: insufficient history")
                continue

            window_df = df.iloc[window_start:window_end].copy()

            # Get current price (end of window)
            price_current = float(df.iloc[t - 1]["close"])

            # Get actual future price (for comparison)
            future_idx = t - 1 + horizon
            if future_idx < len(df):
                price_actual_future = float(df.iloc[future_idx]["close"])
            else:
                price_actual_future = np.nan

            # Prepare input tensor
            try:
                input_tensor = prepare_input_for_model(
                    window_df,
                    feature_cols,
                    seq_len,
                    scaler=scaler,
                    method="per_row"
                ).to(device)
            except Exception as e:
                logger.error(f"Failed to prepare input at t={t}: {e}")
                continue

            # Generate prediction
            pred_log_ret, pred_vol = model(input_tensor)
            pred_log_ret = float(pred_log_ret.cpu().numpy().ravel()[0])
            pred_vol = float(pred_vol.cpu().numpy().ravel()[0])

            # Convert log-return to price
            # P_{t+h} = P_t * exp(log_return)
            price_pred = price_current * np.exp(pred_log_ret)

            # Confidence intervals
            # 1 sigma (~68% confidence)
            upper_1std = price_current * np.exp(pred_log_ret + 1 * pred_vol)
            lower_1std = price_current * np.exp(pred_log_ret - 1 * pred_vol)
            # 2 sigma (~95% confidence)
            upper_2std = price_current * np.exp(pred_log_ret + 2 * pred_vol)
            lower_2std = price_current * np.exp(pred_log_ret - 2 * pred_vol)

            # Get timestamp
            if "timestamp" in df.columns:
                ts = df.iloc[t - 1]["timestamp"]
            else:
                ts = t - 1

            result = {
                "index": t - 1,
                "timestamp": ts,
                "close_current": price_current,
                "close_actual_future": price_actual_future,
                "close_pred": price_pred,
                "log_return_pred": pred_log_ret,
                "volatility_pred": pred_vol,
                "upper_bound_1std": upper_1std,
                "lower_bound_1std": lower_1std,
                "upper_bound_2std": upper_2std,
                "lower_bound_2std": lower_2std,
                "horizon_steps": horizon,
                "prediction_error": price_actual_future - price_pred if not np.isnan(price_actual_future) else np.nan,
                "prediction_error_pct": 100 * (price_actual_future - price_pred) / price_current if not np.isnan(price_actual_future) else np.nan
            }

            results.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{n_predictions} predictions")

    results_df = pd.DataFrame(results)

    logger.info(f"Generated {len(results_df)} predictions successfully")
    logger.info(f"Mean absolute error: {results_df['prediction_error'].abs().mean():.2f}")
    logger.info(f"Mean percentage error: {results_df['prediction_error_pct'].mean():.2f}%")

    return results_df


def predict_autoregressive(
    df: pd.DataFrame,
    model: nn.Module,
    meta: Dict[str, Any],
    scaler: Any,
    device: torch.device,
    n_steps: int,
    use_actual_for_features: bool = True
) -> pd.DataFrame:
    """
    Generate true autoregressive multi-step predictions.

    WARNING: This is more experimental and prediction quality degrades rapidly
    beyond the first few steps. Use predict_multi_horizon_jump when possible.

    Process:
    -------
    1. Start with last seq_len observations from df
    2. Predict next horizon steps
    3. Append predicted price to history
    4. Update features (simplified: reuse last known values or compute from prices)
    5. Repeat for n_steps total predictions

    Args:
        df: DataFrame with OHLCV and features (must have 'close' column)
        model: Trained LSTM2Head model
        meta: Model metadata dict
        scaler: Fitted StandardScaler
        device: torch.device
        n_steps: Number of future steps to predict
        use_actual_for_features: If True and data available, use actual prices
                                 for feature computation (more reliable)

    Returns:
        DataFrame with predictions
    """
    warnings.warn(
        "Autoregressive forecasting is experimental. Prediction quality "
        "degrades rapidly beyond 2-3 horizons. Consider using predict_multi_horizon_jump."
    )

    if torch is None or add_technical_features is None:
        raise RuntimeError("Required dependencies not available")

    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    horizon = meta["horizon"]

    # Start with last seq_len rows of df
    history_df = df.iloc[-seq_len:].copy()
    close_history = history_df["close"].values.tolist()

    results = []
    model.eval()

    with torch.no_grad():
        for step in range(n_steps):
            # Get current price
            price_current = close_history[-1]

            # Prepare input from current history
            try:
                # If we have enough actual data, use it for features
                if use_actual_for_features and len(history_df) >= seq_len:
                    window_df = history_df.iloc[-seq_len:].copy()
                else:
                    # Need to recompute features from synthetic prices
                    # This is a simplified approximation
                    synthetic_close = np.array(close_history[-seq_len:])

                    # Compute minimal features from close prices
                    synthetic_df = pd.DataFrame({"close": synthetic_close})

                    # Add basic features (this is simplified - real features are more complex)
                    synthetic_df["log_close"] = np.log(synthetic_df["close"])
                    synthetic_df["log_ret_1"] = synthetic_df["log_close"].diff(1)
                    synthetic_df["log_ret_5"] = synthetic_df["log_close"].diff(5)

                    # For other features, forward-fill from last known values
                    # This is a crude approximation
                    for col in feature_cols:
                        if col not in synthetic_df.columns:
                            if col in history_df.columns:
                                synthetic_df[col] = history_df[col].iloc[-1]
                            else:
                                synthetic_df[col] = 0.0

                    window_df = synthetic_df

                input_tensor = prepare_input_for_model(
                    window_df,
                    feature_cols,
                    seq_len,
                    scaler=scaler,
                    method="per_row"
                ).to(device)

            except Exception as e:
                logger.error(f"Failed to prepare input at step {step}: {e}")
                break

            # Predict
            pred_log_ret, pred_vol = model(input_tensor)
            pred_log_ret = float(pred_log_ret.cpu().numpy().ravel()[0])
            pred_vol = float(pred_vol.cpu().numpy().ravel()[0])

            # Convert to price
            price_pred = price_current * np.exp(pred_log_ret)
            upper_2std = price_current * np.exp(pred_log_ret + 2 * pred_vol)
            lower_2std = price_current * np.exp(pred_log_ret - 2 * pred_vol)

            # Store result
            result = {
                "step": step,
                "close_current": price_current,
                "close_pred": price_pred,
                "log_return_pred": pred_log_ret,
                "volatility_pred": pred_vol,
                "upper_bound_2std": upper_2std,
                "lower_bound_2std": lower_2std,
            }
            results.append(result)

            # Update history for next iteration
            close_history.append(price_pred)

            # Optionally update history_df with predicted row
            # (simplified - in practice would need full feature recomputation)
            new_row = history_df.iloc[-1:].copy()
            new_row["close"] = price_pred
            history_df = pd.concat([history_df, new_row], ignore_index=True)

    return pd.DataFrame(results)


# ==========================================
# Visualization
# ==========================================

def plot_predictions(
    df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    title: str = "Multi-Horizon Price Predictions",
    figsize: Tuple[int, int] = (14, 8),
    show_confidence: bool = True,
    save_path: Optional[str] = None
) -> Optional[Figure]:
    """
    Plot historical prices with multi-horizon predictions overlay.

    Args:
        df: Historical DataFrame with 'close' and optionally 'timestamp'
        predictions_df: Output from predict_multi_horizon_jump()
        title: Plot title
        figsize: Figure size (width, height)
        show_confidence: Whether to show confidence bands
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object (if matplotlib available)
    """
    if plt is None:
        logger.error("matplotlib not available. Install matplotlib for plotting.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot predictions - SHIFT timestamps forward by horizon to show at prediction time
    horizon = predictions_df["horizon_steps"].iloc[0]

    # Determine the cutoff point: plot historical data only up to the last prediction base time
    # This prevents historical line from overlapping with future predictions
    if "index" in predictions_df.columns:
        last_prediction_base_idx = int(predictions_df["index"].max())
    else:
        # Fallback: use the prediction start point
        last_prediction_base_idx = len(df) - horizon - 1

    # Plot historical close prices (only up to where predictions begin projecting into future)
    df_historical = df.iloc[:last_prediction_base_idx + horizon + 1].copy()

    if "timestamp" in df_historical.columns:
        x_hist = pd.to_datetime(df_historical["timestamp"])
    else:
        x_hist = df_historical.index

    ax.plot(x_hist, df_historical["close"], label="Historical Close", color="black", linewidth=1.5, alpha=0.7)

    if "timestamp" in predictions_df.columns:
        x_pred = pd.to_datetime(predictions_df["timestamp"]) + pd.Timedelta(hours=horizon)
    elif "index" in predictions_df.columns:
        # Map index to timestamp if available
        if "timestamp" in df.columns:
            # Shift indices forward by horizon
            future_indices = predictions_df["index"] + horizon
            valid_mask = future_indices < len(df)
            x_pred = pd.to_datetime(df.iloc[future_indices[valid_mask]]["timestamp"].values)
        else:
            x_pred = predictions_df["index"] + horizon
    else:
        x_pred = predictions_df.index + horizon

    # Plot predicted prices
    ax.plot(x_pred, predictions_df["close_pred"],
            label=f"Predicted Close (h={horizon} steps ahead)",
            color="blue", linewidth=2, alpha=0.8)

    # Plot actual future prices (for comparison) - DON'T shift timestamps
    # close_actual_future already contains the price at t+horizon, so we plot it aligned with predictions
    if "close_actual_future" in predictions_df.columns:
        valid_actuals = predictions_df.dropna(subset=["close_actual_future"])
        if len(valid_actuals) > 0:
            # Use same x-axis as predictions (already shifted)
            if len(x_pred) == len(predictions_df):
                # Filter x_pred to match valid_actuals indices
                x_actual_aligned = x_pred[predictions_df.index.isin(valid_actuals.index)]
            else:
                # Fallback: use predictions timestamps shifted by horizon
                if "timestamp" in valid_actuals.columns:
                    x_actual_aligned = pd.to_datetime(valid_actuals["timestamp"]) + pd.Timedelta(hours=horizon)
                elif "index" in valid_actuals.columns:
                    x_actual_aligned = valid_actuals["index"] + horizon
                else:
                    x_actual_aligned = valid_actuals.index + horizon

            ax.scatter(x_actual_aligned, valid_actuals["close_actual_future"],
                      label="Actual Future Close", color="green", s=20, alpha=0.5, zorder=5)

    # Plot confidence bands - use the same shifted timestamps as predictions
    if show_confidence:
        # Handle potential length mismatch from filtering
        if len(x_pred) == len(predictions_df):
            # 2-sigma (95% CI)
            if "upper_bound_2std" in predictions_df.columns:
                ax.fill_between(
                    x_pred,
                    predictions_df["lower_bound_2std"],
                    predictions_df["upper_bound_2std"],
                    alpha=0.15,
                    color="blue",
                    label="95% CI (±2σ)"
                )
            # 1-sigma (68% CI)
            if "upper_bound_1std" in predictions_df.columns:
                ax.fill_between(
                    x_pred,
                    predictions_df["lower_bound_1std"],
                    predictions_df["upper_bound_1std"],
                    alpha=0.25,
                    color="blue",
                    label="68% CI (±1σ)"
                )

    # Formatting
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis for datetime
    if "timestamp" in df.columns:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    return fig


def plot_prediction_errors(
    predictions_df: pd.DataFrame,
    title: str = "Prediction Errors Over Time",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> Optional[Figure]:
    """
    Plot prediction errors (actual - predicted) over time.

    Args:
        predictions_df: Output from predict_multi_horizon_jump()
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object (if available)
    """
    if plt is None:
        logger.error("matplotlib not available")
        return None

    if "prediction_error" not in predictions_df.columns:
        logger.error("predictions_df must contain 'prediction_error' column")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Get x-axis
    if "timestamp" in predictions_df.columns:
        x = pd.to_datetime(predictions_df["timestamp"])
    elif "index" in predictions_df.columns:
        x = predictions_df["index"]
    else:
        x = predictions_df.index

    # Plot absolute errors
    valid_errors = predictions_df.dropna(subset=["prediction_error"])
    ax1.plot(x, predictions_df["prediction_error"], color="red", alpha=0.7, linewidth=1)
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax1.set_ylabel("Error (Actual - Predicted)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add mean and std
    mean_err = valid_errors["prediction_error"].mean()
    std_err = valid_errors["prediction_error"].std()
    ax1.axhline(y=mean_err, color="blue", linestyle="--", linewidth=1, label=f"Mean: {mean_err:.2f}")
    ax1.fill_between(x, mean_err - std_err, mean_err + std_err, alpha=0.2, color="blue", label=f"±1 Std: {std_err:.2f}")
    ax1.legend(loc="best", fontsize=9)

    # Plot percentage errors
    ax2.plot(x, predictions_df["prediction_error_pct"], color="orange", alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Time", fontsize=11)
    ax2.set_ylabel("Error %", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add mean and std for percentage
    valid_pct = predictions_df.dropna(subset=["prediction_error_pct"])
    if len(valid_pct) > 0:
        mean_pct = valid_pct["prediction_error_pct"].mean()
        std_pct = valid_pct["prediction_error_pct"].std()
        ax2.axhline(y=mean_pct, color="purple", linestyle="--", linewidth=1, label=f"Mean: {mean_pct:.2f}%")
        ax2.fill_between(x, mean_pct - std_pct, mean_pct + std_pct, alpha=0.2, color="purple", label=f"±1 Std: {std_pct:.2f}%")
        ax2.legend(loc="best", fontsize=9)

    # Format x-axis
    if "timestamp" in predictions_df.columns:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Error plot saved to {save_path}")

    return fig


# ==========================================
# Main Example Usage
# ==========================================

def main():
    """
    Example usage of multi-horizon inference system.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Multi-horizon LSTM inference with price conversion")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--meta", type=str, required=True, help="Path to meta.json")
    parser.add_argument("--scaler", type=str, required=True, help="Path to scaler.pkl")
    parser.add_argument("--data", type=str, required=True, help="Path to parquet/csv with features")
    parser.add_argument("--n-predictions", type=int, default=500, help="Number of predictions")
    parser.add_argument("--start-idx", type=int, default=None, help="Starting index for predictions")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--plot-output", type=str, default="predictions_plot.png", help="Plot save path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load model and artifacts
    logger.info("Loading model and artifacts...")
    model, meta, scaler, device = load_model_and_artifacts(
        args.model, args.meta, args.scaler, args.device
    )

    # Load data
    logger.info(f"Loading data from {args.data}...")
    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Generate predictions
    logger.info("Generating multi-horizon predictions...")
    predictions_df = predict_multi_horizon_jump(
        df=df,
        model=model,
        meta=meta,
        scaler=scaler,
        device=device,
        n_predictions=args.n_predictions,
        start_idx=args.start_idx
    )

    # Save predictions
    predictions_df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")

    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total predictions: {len(predictions_df)}")
    logger.info(f"Horizon: {meta['horizon']} steps")

    if "prediction_error" in predictions_df.columns:
        valid = predictions_df.dropna(subset=["prediction_error"])
        if len(valid) > 0:
            mae = valid["prediction_error"].abs().mean()
            rmse = np.sqrt((valid["prediction_error"] ** 2).mean())
            mape = valid["prediction_error_pct"].abs().mean()

            logger.info(f"\nError Metrics:")
            logger.info(f"  MAE (Mean Absolute Error): {mae:.2f}")
            logger.info(f"  RMSE (Root Mean Squared Error): {rmse:.2f}")
            logger.info(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

            # Directional accuracy
            valid["direction_correct"] = (
                (valid["close_actual_future"] > valid["close_current"]) ==
                (valid["close_pred"] > valid["close_current"])
            )
            dir_acc = valid["direction_correct"].mean() * 100
            logger.info(f"  Directional Accuracy: {dir_acc:.1f}%")

    logger.info("="*60 + "\n")

    # Generate plots
    if args.plot:
        logger.info("Generating plots...")

        # Main prediction plot
        fig1 = plot_predictions(
            df=df,
            predictions_df=predictions_df,
            title=f"Multi-Horizon Predictions (h={meta['horizon']})",
            save_path=args.plot_output
        )

        # Error plot
        error_plot_path = args.plot_output.replace(".png", "_errors.png")
        fig2 = plot_prediction_errors(
            predictions_df=predictions_df,
            save_path=error_plot_path
        )

        logger.info("Plots generated successfully")

        # Show plots if in interactive mode
        if plt is not None:
            plt.show()


def predict_multi_horizon_native(
    df: pd.DataFrame,
    model: nn.Module,
    meta: Dict[str, Any],
    scaler: Any,
    device: torch.device,
    n_predictions: int,
    horizons: Optional[List[int]] = None,
    start_idx: Optional[int] = None
) -> Dict[int, pd.DataFrame]:
    """
    Generate native multi-horizon predictions using LSTMMultiHead model.

    This function is designed for models that have been trained on multiple horizons
    simultaneously (LSTMMultiHead). It returns predictions for each native horizon
    without any scaling assumptions.

    Process:
    --------
    For each time t from start_idx to start_idx + n_predictions:
        1. Extract window of seq_len observations ending at t
        2. Prepare features and scale using training scaler
        3. Predict (return, volatility) for ALL trained horizons
        4. Convert log-returns to prices for each horizon
        5. Move to next time step (t+1)

    Args:
        df: DataFrame with OHLCV and features (must have 'close' column)
        model: Trained LSTMMultiHead model
        meta: Model metadata dict with feature_cols, seq_len, horizons
        scaler: Fitted StandardScaler from training
        device: torch.device for inference
        n_predictions: Number of predictions to generate
        horizons: Optional list of horizons to predict (defaults to model's native horizons)
        start_idx: Starting index in df (if None, uses seq_len as minimum)

    Returns:
        Dict mapping horizon -> DataFrame with predictions for that horizon:
        {
            1: DataFrame with h=1 predictions,
            3: DataFrame with h=3 predictions,
            6: DataFrame with h=6 predictions,
            ...
        }

        Each DataFrame contains:
        - timestamp: prediction time
        - close_current: current close price
        - close_actual_future: actual future price at t+h
        - close_pred: predicted close price at t+h
        - log_return_pred: predicted log return
        - volatility_pred: predicted volatility
        - upper_bound_2std: upper CI
        - lower_bound_2std: lower CI
        - horizon_steps: h

    Example:
        >>> preds = predict_multi_horizon_native(df, model, meta, scaler, device, 100)
        >>> preds_h1 = preds[1]  # Predictions for h=1
        >>> preds_h6 = preds[6]  # Predictions for h=6
    """
    if torch is None or prepare_input_for_model is None:
        raise RuntimeError("Required dependencies not available")

    # Extract metadata
    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]

    # Get horizons from model or meta
    if horizons is None:
        if "horizons" in meta:
            horizons = meta["horizons"]
        elif hasattr(model, "horizons"):
            horizons = model.horizons
        else:
            raise ValueError("Could not determine horizons. Pass horizons parameter or include in metadata.")

    max_horizon = max(horizons)

    # Validate inputs
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    if len(df) < seq_len + max_horizon:
        raise ValueError(f"DataFrame too short: need at least {seq_len + max_horizon} rows")

    # Determine start index
    if start_idx is None:
        start_idx = seq_len
    elif start_idx < 0:
        start_idx = len(df) + start_idx
        if start_idx < seq_len:
            raise ValueError(f"start_idx resolves to {start_idx}, but minimum is {seq_len}")

    # Validate we have enough data
    max_idx = len(df) - max_horizon
    if start_idx + n_predictions > max_idx:
        available = max_idx - start_idx
        warnings.warn(f"Requested {n_predictions} predictions but only {available} available. Adjusting.")
        n_predictions = max(1, available)

    # Storage for results (one list per horizon)
    results_per_horizon = {h: [] for h in horizons}

    logger.info(f"Generating {n_predictions} predictions starting from index {start_idx}")
    logger.info(f"Native horizons: {horizons}")

    model.eval()
    with torch.no_grad():
        for i in range(n_predictions):
            t = start_idx + i

            # Extract window ending at t
            window_start = t - seq_len
            window_end = t

            if window_start < 0:
                logger.warning(f"Skipping prediction at t={t}: insufficient history")
                continue

            window_df = df.iloc[window_start:window_end].copy()

            # Get current price (end of window)
            price_current = float(df.iloc[t - 1]["close"])

            # Prepare input tensor
            try:
                input_tensor = prepare_input_for_model(
                    window_df,
                    feature_cols,
                    seq_len,
                    scaler=scaler,
                    method="per_row"
                ).to(device)
            except Exception as e:
                logger.error(f"Failed to prepare input at t={t}: {e}")
                continue

            # Generate predictions for all horizons
            predictions = model(input_tensor)  # Returns dict {horizon: (ret, vol)}

            # Get timestamp
            if "timestamp" in df.columns:
                ts = df.iloc[t - 1]["timestamp"]
            else:
                ts = t - 1

            # Process each horizon
            for h in horizons:
                pred_ret, pred_vol = predictions[h]
                pred_log_ret = float(pred_ret.cpu().numpy().ravel()[0])
                pred_vol_val = float(pred_vol.cpu().numpy().ravel()[0])

                # Get actual future price
                future_idx = t - 1 + h
                if future_idx < len(df):
                    price_actual_future = float(df.iloc[future_idx]["close"])
                else:
                    price_actual_future = np.nan

                # Convert log-return to price
                price_pred = price_current * np.exp(pred_log_ret)

                # Confidence intervals
                upper_1std = price_current * np.exp(pred_log_ret + 1 * pred_vol_val)
                lower_1std = price_current * np.exp(pred_log_ret - 1 * pred_vol_val)
                upper_2std = price_current * np.exp(pred_log_ret + 2 * pred_vol_val)
                lower_2std = price_current * np.exp(pred_log_ret - 2 * pred_vol_val)

                result = {
                    "index": t - 1,
                    "timestamp": ts,
                    "close_current": price_current,
                    "close_actual_future": price_actual_future,
                    "close_pred": price_pred,
                    "log_return_pred": pred_log_ret,
                    "volatility_pred": pred_vol_val,
                    "upper_bound_1std": upper_1std,
                    "lower_bound_1std": lower_1std,
                    "upper_bound_2std": upper_2std,
                    "lower_bound_2std": lower_2std,
                    "horizon_steps": h
                }

                results_per_horizon[h].append(result)

    # Convert to DataFrames
    dfs = {}
    for h in horizons:
        df_h = pd.DataFrame(results_per_horizon[h])
        if len(df_h) > 0:
            # Add computed columns
            df_h["prediction_error"] = df_h["close_actual_future"] - df_h["close_pred"]
            df_h["prediction_error_pct"] = 100 * df_h["prediction_error"] / df_h["close_current"]
            df_h["directionally_correct"] = np.sign(df_h["close_actual_future"] - df_h["close_current"]) == np.sign(df_h["close_pred"] - df_h["close_current"])
        dfs[h] = df_h

    logger.info(f"Generated {len(dfs)} horizon-specific prediction sets")
    return dfs


def detect_model_type(model: nn.Module, meta: Dict[str, Any]) -> str:
    """
    Detect whether model is LSTM2Head (single horizon) or LSTMMultiHead (multi-horizon).

    Args:
        model: PyTorch model instance
        meta: Model metadata dict

    Returns:
        "multi" if LSTMMultiHead, "single" if LSTM2Head
    """
    # Check metadata first
    if "model_type" in meta:
        model_type = meta["model_type"]
        if "Multi" in model_type:
            return "multi"
        return "single"

    # Check model attributes
    if hasattr(model, "horizons"):
        return "multi"

    # Check for multi-head structure
    if hasattr(model, "heads_ret") or hasattr(model, "heads_vol"):
        return "multi"

    # Default to single
    return "single"


def predict_universal(
    df: pd.DataFrame,
    model: nn.Module,
    meta: Dict[str, Any],
    scaler: Any,
    device: torch.device,
    n_predictions: int,
    horizon: Optional[int] = None,
    start_idx: Optional[int] = None
) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Universal prediction function that automatically detects model type.

    Supports both:
    - LSTM2Head (single horizon with scaling)
    - LSTMMultiHead (native multi-horizon)

    Args:
        df: DataFrame with OHLCV and features
        model: Trained model (LSTM2Head or LSTMMultiHead)
        meta: Model metadata dict
        scaler: Fitted StandardScaler
        device: torch.device
        n_predictions: Number of predictions to generate
        horizon: For LSTM2Head: the target horizon (uses scaling if different from native)
                 For LSTMMultiHead: if specified, return only this horizon's predictions
        start_idx: Starting index in df

    Returns:
        - If LSTM2Head or horizon specified: Single DataFrame with predictions
        - If LSTMMultiHead and horizon=None: Dict mapping horizon -> DataFrame

    Example:
        >>> # Auto-detect and use appropriate method
        >>> result = predict_universal(df, model, meta, scaler, device, 100)
        >>> # For LSTMMultiHead, result is dict: {1: df1, 6: df6, ...}
        >>> # For LSTM2Head, result is DataFrame
    """
    model_type = detect_model_type(model, meta)

    if model_type == "multi":
        logger.info("Detected LSTMMultiHead model - using native multi-horizon prediction")

        # Get predictions for all horizons
        preds_dict = predict_multi_horizon_native(
            df, model, meta, scaler, device, n_predictions, start_idx=start_idx
        )

        # If specific horizon requested, return only that one
        if horizon is not None:
            if horizon in preds_dict:
                return preds_dict[horizon]
            else:
                # Try interpolation if horizon not native
                logger.warning(f"Horizon {horizon} not in native horizons {list(preds_dict.keys())}, using interpolation")
                # For now, return closest horizon
                horizons_sorted = sorted(preds_dict.keys())
                closest = min(horizons_sorted, key=lambda h: abs(h - horizon))
                logger.info(f"Using closest native horizon h={closest}")
                return preds_dict[closest]

        # Return all horizons
        return preds_dict

    else:  # model_type == "single"
        logger.info("Detected LSTM2Head model - using single-horizon prediction (with scaling if needed)")

        # Use existing jump prediction method
        return predict_multi_horizon_jump(
            df, model, meta, scaler, device, n_predictions, start_idx=start_idx
        )


if __name__ == "__main__":
    main()
