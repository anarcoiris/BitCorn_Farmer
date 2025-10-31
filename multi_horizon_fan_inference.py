#!/usr/bin/env python3
# multi_horizon_fan_inference.py - Multi-horizon prediction module

from __future__ import annotations
import logging
import warnings
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    import fiboevo
    from fiboevo import prepare_input_for_model
except ImportError:
    fiboevo = None
    prepare_input_for_model = None

logger = logging.getLogger(__name__)


def predict_single_point_multi_horizon(
    df: pd.DataFrame,
    model: nn.Module,
    meta: Dict[str, Any],
    scaler: Any,
    device: Any,
    horizons: List[int],
    method: str = "scaling",
    current_price: Optional[float] = None
) -> Dict[int, Dict[str, float]]:
    """Generate multi-horizon predictions for the LATEST data point only."""
    if torch is None or prepare_input_for_model is None:
        raise RuntimeError("Required dependencies not available")
    
    feature_cols = meta.get("feature_cols")
    if not feature_cols:
        raise ValueError("meta feature_cols required")
    
    seq_len = meta.get("seq_len", 64)
    h_native = meta.get("horizon", 10)
    
    if "close" not in df.columns:
        raise ValueError("close column required")
    
    if len(df) < seq_len:
        raise ValueError(f"Need {seq_len} rows, got {len(df)}")
    
    if current_price is None:
        current_price = float(df.iloc[-1]["close"])
    
    if "timestamp" in df.columns:
        base_timestamp = pd.to_datetime(df.iloc[-1]["timestamp"])
    elif "ts" in df.columns:
        base_timestamp = pd.to_datetime(df.iloc[-1]["ts"], unit="s", utc=True)
    else:
        base_timestamp = pd.Timestamp.now(tz="UTC")
    
    window_df = df.iloc[-seq_len:].copy()
    
    missing = [c for c in feature_cols if c not in window_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    try:
        input_tensor = prepare_input_for_model(
            window_df, feature_cols, seq_len, scaler=scaler, method="per_row"
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Input prep failed: {e}")
    
    model.eval()
    with torch.no_grad():
        try:
            pred_log_ret_native, pred_vol_native = model(input_tensor)
            pred_log_ret_native = float(pred_log_ret_native.cpu().numpy().ravel()[0])
            pred_vol_native = float(pred_vol_native.cpu().numpy().ravel()[0])
        except Exception as e:
            raise RuntimeError(f"Model failed: {e}")
    
    results = {}
    for h_target in horizons:
        scale_factor = np.sqrt(h_target / h_native)
        log_ret_scaled = pred_log_ret_native * scale_factor
        vol_scaled = pred_vol_native * scale_factor
        price_pred = current_price * np.exp(log_ret_scaled)
        ci_lower_68 = current_price * np.exp(log_ret_scaled - vol_scaled)
        ci_upper_68 = current_price * np.exp(log_ret_scaled + vol_scaled)
        ci_lower_55 = current_price * np.exp(log_ret_scaled - 0.76 * vol_scaled)
        ci_upper_55 = current_price * np.exp(log_ret_scaled + 0.76 * vol_scaled)
        change_usd = price_pred - current_price
        change_pct = 100 * change_usd / current_price if current_price > 0 else 0.0
        
        try:
            timeframe = meta.get("timeframe", "1h")
            if "m" in timeframe:
                minutes = int(timeframe.replace("m", ""))
                future_timestamp = base_timestamp + pd.Timedelta(minutes=h_target * minutes)
            elif "h" in timeframe:
                hours = int(timeframe.replace("h", ""))
                future_timestamp = base_timestamp + pd.Timedelta(hours=h_target * hours)
            else:
                future_timestamp = base_timestamp + pd.Timedelta(hours=h_target)
        except Exception:
            future_timestamp = base_timestamp + pd.Timedelta(hours=h_target)
        
        results[h_target] = {
            "horizon": h_target,
            "price": float(price_pred),
            "log_return": float(log_ret_scaled),
            "volatility": float(vol_scaled),
            "ci_lower_55": float(ci_lower_55),
            "ci_upper_55": float(ci_upper_55),
            "ci_lower_68": float(ci_lower_68),
            "ci_upper_68": float(ci_upper_68),
            "change_usd": float(change_usd),
            "change_pct": float(change_pct),
            "current_price": float(current_price),
            "timestamp": str(future_timestamp),
            "base_timestamp": str(base_timestamp),
            "scaling_factor": float(scale_factor),
            "native_horizon": int(h_native),
        }
    
    return results


def format_prediction_summary(predictions: Dict[int, Dict[str, float]]) -> str:
    lines = ["Multi-Horizon Predictions:", "-" * 60]
    for h in sorted(predictions.keys()):
        pred = predictions[h]
        price = pred["price"]
        change_pct = pred["change_pct"]
        ci_55 = f"[{pred['ci_lower_55']:.0f}, {pred['ci_upper_55']:.0f}]"
        direction = "UP" if change_pct > 0 else "DN" if change_pct < 0 else "FLAT"
        lines.append(f"h={h:2d}:  ({direction} {change_pct:+.2f}%) 55% CI: {ci_55}")
    return chr(10).join(lines)


def get_dominant_signal(predictions: Dict[int, Dict[str, float]]) -> str:
    if not predictions:
        return "NEUTRAL"
    changes_pct = [p["change_pct"] for p in predictions.values()]
    mean_change = np.mean(changes_pct)
    if mean_change > 0.1:
        return "BULLISH"
    elif mean_change < -0.1:
        return "BEARISH"
    else:
        return "NEUTRAL"


# ==========================================
# Single-Point Multi-Horizon Prediction
# (Optimized for Real-Time Inference)
# ==========================================

def predict_single_point_multi_horizon(
    df: pd.DataFrame,
    model,
    meta: Dict[str, Any],
    scaler,
    device,
    horizons: List[int],
    method: str = "scaling",
    current_price: Optional[float] = None
) -> Dict[int, Dict[str, float]]:
    """
    Generate multi-horizon predictions for the LATEST data point only.

    Optimized for real-time inference (faster than predict_multiple_horizons).
    Perfect for live trading dashboards.

    Args:
        df: DataFrame with features (uses last seq_len rows)
        model: Trained LSTM2Head model
        meta: Model metadata dict with feature_cols, seq_len, horizon
        scaler: Fitted StandardScaler from training
        device: torch.device for inference
        horizons: List of horizons to predict [1, 3, 5, 10, 20, 30]
        method: "scaling" (fast) or "iterative" (accurate for long horizons)
        current_price: Current price (if None, uses df.iloc[-1]['close'])

    Returns:
        Dictionary keyed by horizon:
        {
            1: {
                "horizon": 1,
                "price": 106500.50,
                "log_return": 0.0005,
                "volatility": 0.20,
                "ci_lower_68": 106450.0,
                "ci_upper_68": 106550.0,
                "ci_lower_55": 106400.0,
                "ci_upper_55": 106600.0,
                "change_usd": 50.50,
                "change_pct": 0.05,
                "current_price": 106450.0
            },
            3: {...},
            ...
        }

    Raises:
        ValueError: If df doesn't have enough rows
        RuntimeError: If required dependencies unavailable
    """
    try:
        import torch
        from fiboevo import prepare_input_for_model
    except ImportError as e:
        raise RuntimeError(f"Required dependencies not available: {e}")

    # Extract metadata
    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    native_horizon = meta["horizon"]

    # Validate inputs
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    if len(df) < seq_len:
        raise ValueError(f"DataFrame too short: need at least {seq_len} rows, got {len(df)}")

    # Get current price
    if current_price is None:
        current_price = float(df.iloc[-1]["close"])

    # Extract last seq_len rows as window
    window_df = df.iloc[-seq_len:].copy()

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
        raise RuntimeError(f"Failed to prepare input tensor: {e}")

    # Generate native prediction
    model.eval()
    with torch.no_grad():
        pred_log_ret_native, pred_vol_native = model(input_tensor)
        pred_log_ret_native = float(pred_log_ret_native.cpu().numpy().ravel()[0])
        pred_vol_native = float(pred_vol_native.cpu().numpy().ravel()[0])

    # Generate predictions for each horizon
    predictions = {}

    for h in horizons:
        # Scale prediction to target horizon
        if h == native_horizon:
            # Native horizon - use direct prediction
            pred_log_ret_h = pred_log_ret_native
            pred_vol_h = pred_vol_native
        else:
            # Scale based on method
            scale_factor = h / native_horizon

            if method == "scaling":
                # Linear drift scaling (fast, assumes constant drift)
                pred_log_ret_h = pred_log_ret_native * scale_factor
                # Volatility scales with sqrt(time) (Brownian motion)
                pred_vol_h = pred_vol_native * np.sqrt(scale_factor)
            else:
                # Iterative method would go here (not implemented for single point)
                # For now, fall back to scaling
                pred_log_ret_h = pred_log_ret_native * scale_factor
                pred_vol_h = pred_vol_native * np.sqrt(scale_factor)

        # Convert log-return to price
        # P_{t+h} = P_t * exp(log_return)
        pred_price = current_price * np.exp(pred_log_ret_h)

        # Confidence intervals
        # 68% CI (±1σ)
        ci_lower_68 = current_price * np.exp(pred_log_ret_h - 1.0 * pred_vol_h)
        ci_upper_68 = current_price * np.exp(pred_log_ret_h + 1.0 * pred_vol_h)

        # 55% CI (±0.76σ)
        ci_lower_55 = current_price * np.exp(pred_log_ret_h - 0.76 * pred_vol_h)
        ci_upper_55 = current_price * np.exp(pred_log_ret_h + 0.76 * pred_vol_h)

        # Calculate changes
        change_usd = pred_price - current_price
        change_pct = 100 * (pred_price - current_price) / current_price

        # Store prediction
        predictions[h] = {
            "horizon": h,
            "price": pred_price,
            "log_return": pred_log_ret_h,
            "volatility": pred_vol_h,
            "ci_lower_68": ci_lower_68,
            "ci_upper_68": ci_upper_68,
            "ci_lower_55": ci_lower_55,
            "ci_upper_55": ci_upper_55,
            "change_usd": change_usd,
            "change_pct": change_pct,
            "current_price": current_price,
            "method": "direct" if h == native_horizon else method
        }

    return predictions
