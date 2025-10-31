#!/usr/bin/env python3
"""
example_future_predictions.py

Generate TRUE FUTURE predictions beyond the last available historical data point.

This script demonstrates:
1. Loading the most recent historical data
2. Generating predictions extending into the future (beyond available data)
3. Visualizing historical context with future projections
4. Showing both jump predictions (on recent history) and autoregressive future predictions

Usage:
------
    python example_future_predictions.py
"""

import logging
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multi_horizon_inference import (
    load_model_and_artifacts,
    predict_multi_horizon_jump,
    predict_autoregressive
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_future_predictions():
    """
    Generate predictions extending into the true future.
    """
    # ==========================================
    # Configuration
    # ==========================================

    MODEL_PATH = "artifacts/model_best.pt"
    META_PATH = "artifacts/meta.json"
    SCALER_PATH = "artifacts/scaler.pkl"
    DATA_PATH = "data_manager/exports/Binance_BTCUSDT_1h.db"

    # Prediction parameters
    N_RECENT_PREDICTIONS = 168  # 168 hours = 1 week of recent predictions for context
    N_FUTURE_STEPS = 240  # 240 hours = 10 days into the future

    OUTPUT_CSV = "future_predictions.csv"
    PLOT_OUTPUT = "future_predictions_plot.png"

    logger.info("="*70)
    logger.info("Future Price Prediction System")
    logger.info("="*70)

    # ==========================================
    # Step 1: Load Model
    # ==========================================
    logger.info("\nStep 1: Loading model and artifacts...")

    try:
        model, meta, scaler, device = load_model_and_artifacts(
            model_path=MODEL_PATH,
            meta_path=META_PATH,
            scaler_path=SCALER_PATH,
            device=None
        )

        horizon = meta['horizon']
        seq_len = meta['seq_len']

        logger.info(f"Model loaded: {len(meta['feature_cols'])} features, horizon={horizon}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # ==========================================
    # Step 2: Load Recent Historical Data
    # ==========================================
    logger.info("\nStep 2: Loading historical data...")

    try:
        import sqlite3
        from fiboevo import add_technical_features

        conn = sqlite3.connect(DATA_PATH)
        df = pd.read_sql_query("SELECT * FROM ohlcv ORDER BY ts", conn)
        conn.close()

        # Convert timestamp
        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)

        # Compute features
        logger.info("Computing technical features...")
        timestamps = df["timestamp"].copy()

        df_features = add_technical_features(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            volume=df["volume"].values,
            dropna_after=False
        )

        df_features["timestamp"] = timestamps
        df = df_features.dropna().reset_index(drop=True)

        last_timestamp = df["timestamp"].iloc[-1]
        last_close = df["close"].iloc[-1]

        logger.info(f"Loaded {len(df)} rows")
        logger.info(f"Last data point: {last_timestamp} @ ${last_close:,.2f}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # ==========================================
    # Step 3: Generate Recent Predictions (Jump Mode)
    # ==========================================
    logger.info(f"\nStep 3: Generating {N_RECENT_PREDICTIONS} recent predictions for context...")

    try:
        start_idx = -(N_RECENT_PREDICTIONS + horizon)  # Ensure we have enough data

        recent_predictions_df = predict_multi_horizon_jump(
            df=df,
            model=model,
            meta=meta,
            scaler=scaler,
            device=device,
            n_predictions=N_RECENT_PREDICTIONS,
            start_idx=start_idx,
            return_features=False
        )

        logger.info(f"Generated {len(recent_predictions_df)} recent predictions")

    except Exception as e:
        logger.error(f"Failed to generate recent predictions: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # Step 4: Generate True Future Predictions (Autoregressive)
    # ==========================================
    logger.info(f"\nStep 4: Generating {N_FUTURE_STEPS} future predictions (autoregressive)...")
    logger.info("WARNING: Autoregressive predictions degrade in quality over time")

    try:
        future_predictions_df = predict_autoregressive(
            df=df,
            model=model,
            meta=meta,
            scaler=scaler,
            device=device,
            n_steps=N_FUTURE_STEPS,
            use_actual_for_features=False  # Pure future prediction mode
        )

        # Add timestamps for future predictions
        future_predictions_df["timestamp"] = [
            last_timestamp + timedelta(hours=(i+1) * horizon)
            for i in range(len(future_predictions_df))
        ]

        logger.info(f"Generated {len(future_predictions_df)} future predictions")
        logger.info(f"Future predictions extend to: {future_predictions_df['timestamp'].iloc[-1]}")

    except Exception as e:
        logger.error(f"Failed to generate future predictions: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # Step 5: Save Combined Results
    # ==========================================
    logger.info(f"\nStep 5: Saving results to {OUTPUT_CSV}...")

    try:
        # Mark prediction types
        recent_predictions_df["prediction_type"] = "jump_historical"
        future_predictions_df["prediction_type"] = "autoregressive_future"

        # Align columns - autoregressive has different column names
        # Rename autoregressive columns to match jump predictions for consistency
        future_predictions_df = future_predictions_df.rename(columns={
            "close_current": "close_current",  # Keep same
            "step": "horizon_steps"  # Map step number to horizon
        })

        # Add missing columns that only exist in jump predictions
        if "close_actual_future" not in future_predictions_df.columns:
            future_predictions_df["close_actual_future"] = np.nan
        if "prediction_error" not in future_predictions_df.columns:
            future_predictions_df["prediction_error"] = np.nan
        if "prediction_error_pct" not in future_predictions_df.columns:
            future_predictions_df["prediction_error_pct"] = np.nan
        if "index" not in future_predictions_df.columns:
            future_predictions_df["index"] = np.nan

        # Combine
        combined_df = pd.concat([recent_predictions_df, future_predictions_df], ignore_index=True)
        combined_df.to_csv(OUTPUT_CSV, index=False)

        logger.info(f"Saved {len(combined_df)} total predictions")
        logger.info(f"  - Recent (jump): {len(recent_predictions_df)}")
        logger.info(f"  - Future (AR): {len(future_predictions_df)}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================
    # Step 6: Visualize
    # ==========================================
    logger.info(f"\nStep 6: Creating visualization...")

    try:
        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot recent historical data (context)
        recent_window = 500  # Show last 500 hours of history
        df_recent = df.iloc[-recent_window:].copy()

        ax.plot(
            pd.to_datetime(df_recent["timestamp"]),
            df_recent["close"],
            label="Historical Close",
            color="black",
            linewidth=1.5,
            alpha=0.7
        )

        # Plot recent predictions (with actual future data for comparison)
        recent_pred_timestamps = pd.to_datetime(recent_predictions_df["timestamp"]) + pd.Timedelta(hours=horizon)

        ax.plot(
            recent_pred_timestamps,
            recent_predictions_df["close_pred"],
            label=f"Recent Predictions (h={horizon} steps)",
            color="blue",
            linewidth=2,
            alpha=0.6
        )

        # Plot actual future (where available)
        valid_actuals = recent_predictions_df.dropna(subset=["close_actual_future"])
        if len(valid_actuals) > 0:
            actual_timestamps = pd.to_datetime(valid_actuals["timestamp"]) + pd.Timedelta(hours=horizon)
            ax.scatter(
                actual_timestamps,
                valid_actuals["close_actual_future"],
                label="Actual Future Close",
                color="green",
                s=20,
                alpha=0.5,
                zorder=5
            )

        # Plot TRUE FUTURE predictions (autoregressive)
        ax.plot(
            pd.to_datetime(future_predictions_df["timestamp"]),
            future_predictions_df["close_pred"],
            label=f"Future Forecast (AR, {N_FUTURE_STEPS} steps)",
            color="red",
            linewidth=2.5,
            alpha=0.8,
            linestyle="--"
        )

        # Plot future confidence bands
        ax.fill_between(
            pd.to_datetime(future_predictions_df["timestamp"]),
            future_predictions_df["lower_bound_2std"],
            future_predictions_df["upper_bound_2std"],
            alpha=0.2,
            color="red",
            label="Future 95% CI"
        )

        # Add vertical line at "NOW" (last historical data point)
        ax.axvline(
            x=last_timestamp,
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"Last Data Point ({last_timestamp.strftime('%Y-%m-%d')})"
        )

        # Formatting
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Price (USD)", fontsize=12)
        ax.set_title(
            f"BTC/USDT: Historical Context + True Future Predictions\n"
            f"Last Data: {last_timestamp.strftime('%Y-%m-%d %H:%M')} @ ${last_close:,.2f}",
            fontsize=14,
            fontweight="bold"
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        plt.tight_layout()
        fig.savefig(PLOT_OUTPUT, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {PLOT_OUTPUT}")

        # Show plot
        plt.show()

    except Exception as e:
        logger.error(f"Failed to create plot: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================
    # Summary
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Last historical data: {last_timestamp}")
    logger.info(f"Last historical price: ${last_close:,.2f}")
    logger.info(f"Recent predictions: {len(recent_predictions_df)} (jump forecasting)")
    logger.info(f"Future predictions: {len(future_predictions_df)} (autoregressive)")
    logger.info(f"Forecast horizon: {N_FUTURE_STEPS * horizon} hours (~{N_FUTURE_STEPS * horizon / 24:.1f} days)")

    if len(future_predictions_df) > 0:
        final_pred = future_predictions_df.iloc[-1]
        logger.info(f"\nFinal prediction:")
        logger.info(f"  Time: {final_pred['timestamp']}")
        logger.info(f"  Price: ${final_pred['close_pred']:,.2f}")
        logger.info(f"  95% CI: ${final_pred['lower_bound_2std']:,.2f} - ${final_pred['upper_bound_2std']:,.2f}")

    logger.info("="*70)
    logger.info("\nDone!")


if __name__ == "__main__":
    generate_future_predictions()
