#!/usr/bin/env python3
"""
simple_future_forecast.py

Simple and robust future price forecasting using pattern repetition.

Instead of complex autoregressive feature generation, this approach:
1. Uses the last known feature pattern
2. Makes single-step predictions extending into the future
3. Updates only the price-dependent features minimally

This is more stable than full autoregressive prediction.
"""

import logging
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multi_horizon_inference import (
    load_model_and_artifacts,
    predict_multi_horizon_jump
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def simple_future_forecast(df, model, meta, scaler, device, n_future_steps=100):
    """
    Generate simple future forecasts by repeating last known pattern.

    This is MORE STABLE than autoregressive because it doesn't try to
    regenerate complex features.
    """
    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    horizon = meta["horizon"]

    results = []

    # Get the last window of features
    last_window = df.iloc[-seq_len:].copy()
    last_price = float(df.iloc[-1]["close"])
    last_timestamp = pd.to_datetime(df.iloc[-1]["timestamp"])

    logger.info(f"Starting simple forecast from {last_timestamp} @ ${last_price:,.2f}")

    model.eval()
    with torch.no_grad():
        for step in range(n_future_steps):
            # Use last window for prediction
            try:
                # Prepare features
                X = last_window[feature_cols].values

                # Scale
                X_scaled = scaler.transform(X)

                # To tensor
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(device)

                # Predict
                pred_log_ret, pred_vol = model(X_tensor)
                pred_log_ret = float(pred_log_ret.cpu().numpy().ravel()[0])
                pred_vol = float(pred_vol.cpu().numpy().ravel()[0])

                # Convert to price
                price_pred = last_price * np.exp(pred_log_ret)
                upper_1std = last_price * np.exp(pred_log_ret + 1 * pred_vol)
                lower_1std = last_price * np.exp(pred_log_ret - 1 * pred_vol)
                upper_2std = last_price * np.exp(pred_log_ret + 2 * pred_vol)
                lower_2std = last_price * np.exp(pred_log_ret - 2 * pred_vol)

                # Future timestamp
                future_timestamp = last_timestamp + timedelta(hours=(step + 1) * horizon)

                result = {
                    "step": step,
                    "timestamp": future_timestamp,
                    "close_pred": price_pred,
                    "log_return_pred": pred_log_ret,
                    "volatility_pred": pred_vol,
                    "upper_bound_1std": upper_1std,
                    "lower_bound_1std": lower_1std,
                    "upper_bound_2std": upper_2std,
                    "lower_bound_2std": lower_2std,
                }
                results.append(result)

                # Update last price for next iteration
                last_price = price_pred
                last_timestamp = future_timestamp

                # Shift window forward (keep same features, update simple ones)
                # This is a simplification - we maintain most features constant
                # and only update log_close and simple log_returns
                new_row = last_window.iloc[-1:].copy()
                new_row["close"] = price_pred
                if "log_close" in new_row.columns:
                    new_row["log_close"] = np.log(price_pred)

                # Shift window
                last_window = pd.concat([last_window.iloc[1:], new_row], ignore_index=True)

                if (step + 1) % 50 == 0:
                    logger.info(f"  Generated {step + 1}/{n_future_steps} forecasts")

            except Exception as e:
                logger.error(f"Failed at step {step}: {e}")
                break

    return pd.DataFrame(results)


def main():
    """
    Main execution: generate recent + future predictions and visualize.
    """
    # Configuration
    MODEL_PATH = "artifacts/model_best.pt"
    META_PATH = "artifacts/meta.json"
    SCALER_PATH = "artifacts/scaler.pkl"
    DATA_PATH = "data_manager/exports/Binance_BTCUSDT_1h.db"

    N_RECENT = 168  # 1 week of recent context
    N_FUTURE = 240  # 10 days into future

    logger.info("="*60)
    logger.info("Simple Future Forecast System")
    logger.info("="*60)

    # Load model
    logger.info("Loading model...")
    model, meta, scaler, device = load_model_and_artifacts(
        MODEL_PATH, META_PATH, SCALER_PATH
    )
    horizon = meta['horizon']
    logger.info(f"Model loaded: horizon={horizon}")

    # Load data
    logger.info("Loading data...")
    import sqlite3
    from fiboevo import add_technical_features

    conn = sqlite3.connect(DATA_PATH)
    df = pd.read_sql_query("SELECT * FROM ohlcv ORDER BY ts", conn)
    conn.close()

    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    timestamps = df["timestamp"].copy()

    df_features = add_technical_features(
        df["close"].values,
        df["high"].values,
        df["low"].values,
        df["volume"].values,
        dropna_after=False
    )
    df_features["timestamp"] = timestamps
    df = df_features.dropna().reset_index(drop=True)

    last_timestamp = df["timestamp"].iloc[-1]
    last_close = df["close"].iloc[-1]

    logger.info(f"Data loaded: {len(df)} rows")
    logger.info(f"Last data: {last_timestamp} @ ${last_close:,.2f}")

    # Generate recent predictions (jump mode for validation)
    logger.info(f"\nGenerating {N_RECENT} recent predictions...")
    start_idx = -(N_RECENT + horizon)

    recent_df = predict_multi_horizon_jump(
        df, model, meta, scaler, device,
        n_predictions=N_RECENT,
        start_idx=start_idx
    )
    logger.info(f"Recent predictions: {len(recent_df)}")

    # Generate future predictions (simple forecast)
    logger.info(f"\nGenerating {N_FUTURE} future forecasts...")
    future_df = simple_future_forecast(df, model, meta, scaler, device, N_FUTURE)
    logger.info(f"Future forecasts: {len(future_df)}")

    if len(future_df) > 0:
        logger.info(f"Forecast extends to: {future_df['timestamp'].iloc[-1]}")
        logger.info(f"Final price prediction: ${future_df['close_pred'].iloc[-1]:,.2f}")

    # Save results
    logger.info("\nSaving results...")
    recent_df["prediction_type"] = "recent_validated"
    future_df["prediction_type"] = "future_forecast"

    combined = pd.concat([recent_df, future_df], ignore_index=True)
    combined.to_csv("simple_forecast.csv", index=False)
    logger.info(f"Saved {len(combined)} predictions to simple_forecast.csv")

    # Visualize
    logger.info("\nCreating visualization...")

    fig, ax = plt.subplots(figsize=(18, 10))

    # Historical data (last 500 hours for context)
    df_recent = df.iloc[-500:].copy()
    ax.plot(
        pd.to_datetime(df_recent["timestamp"]),
        df_recent["close"],
        label="Historical Close",
        color="black",
        linewidth=1.5,
        alpha=0.7
    )

    # Recent predictions (validated)
    recent_ts = pd.to_datetime(recent_df["timestamp"]) + pd.Timedelta(hours=horizon)
    ax.plot(
        recent_ts,
        recent_df["close_pred"],
        label=f"Recent Predictions (h={horizon})",
        color="blue",
        linewidth=2,
        alpha=0.7
    )

    # Actual future (where available)
    valid_actuals = recent_df.dropna(subset=["close_actual_future"])
    if len(valid_actuals) > 0:
        actual_ts = pd.to_datetime(valid_actuals["timestamp"]) + pd.Timedelta(hours=horizon)
        ax.scatter(
            actual_ts,
            valid_actuals["close_actual_future"],
            label="Actual Future",
            color="green",
            s=20,
            alpha=0.5
        )

    # FUTURE FORECAST (the key part!)
    future_ts = pd.to_datetime(future_df["timestamp"])
    ax.plot(
        future_ts,
        future_df["close_pred"],
        label=f"Future Forecast ({N_FUTURE} steps)",
        color="red",
        linewidth=2.5,
        alpha=0.9,
        linestyle="--"
    )

    # Future confidence bands
    ax.fill_between(
        future_ts,
        future_df["lower_bound_2std"],
        future_df["upper_bound_2std"],
        alpha=0.2,
        color="red",
        label="Future 95% CI"
    )

    # Mark "NOW" line
    ax.axvline(
        x=last_timestamp,
        color="orange",
        linestyle=":",
        linewidth=3,
        label=f"Last Data Point\\n{last_timestamp.strftime('%Y-%m-%d')}"
    )

    # Formatting
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("Price (USD)", fontsize=13)
    ax.set_title(
        f"BTC/USDT: Historical + True Future Forecast\\n"
        f"Last Data: {last_timestamp.strftime('%Y-%m-%d %H:%M')} @ ${last_close:,.2f}",
        fontsize=15,
        fontweight="bold"
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.savefig("simple_forecast_plot.png", dpi=150, bbox_inches="tight")
    logger.info("Plot saved to simple_forecast_plot.png")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Last historical data: {last_timestamp}")
    logger.info(f"Last historical price: ${last_close:,.2f}")
    logger.info(f"Recent predictions: {len(recent_df)}")
    logger.info(f"Future forecasts: {len(future_df)}")
    logger.info(f"Forecast extends to: {future_df['timestamp'].iloc[-1]}")
    logger.info(f"Final prediction: ${future_df['close_pred'].iloc[-1]:,.2f}")
    logger.info(f"  Upper bound (95%): ${future_df['upper_bound_2std'].iloc[-1]:,.2f}")
    logger.info(f"  Lower bound (95%): ${future_df['lower_bound_2std'].iloc[-1]:,.2f}")
    logger.info("="*60)

    plt.show()


if __name__ == "__main__":
    main()
