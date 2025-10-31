#!/usr/bin/env python3
"""
example_multi_horizon.py

Example script demonstrating how to use the multi-horizon inference system.

This script shows:
1. Loading a trained model and its artifacts
2. Loading or preparing feature data
3. Generating multi-horizon predictions
4. Converting predictions back to price scale
5. Visualizing results
6. Analyzing prediction quality

Usage:
------
Basic usage with default settings:
    python example_multi_horizon.py

With custom parameters:
    python example_multi_horizon.py --n-predictions 1000 --start-idx 1000

Using specific data source:
    python example_multi_horizon.py --data data_manager/exports/features/binance_BTCUSDT_30m_features_v1.parquet

Full example with plots:
    python example_multi_horizon.py --plot --output results/predictions.csv --plot-output results/plot.png
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multi_horizon_inference import (
    load_model_and_artifacts,
    predict_multi_horizon_jump,
    plot_predictions,
    plot_prediction_errors
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str, min_rows: int = 1000) -> pd.DataFrame:
    """
    Load data and compute required features.

    Args:
        data_path: Path to parquet, csv, or SQLite database file
        min_rows: Minimum number of rows required

    Returns:
        DataFrame with all computed features
    """
    import sqlite3
    from fiboevo import add_technical_features

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}")

    # Load data based on file type
    if data_path.suffix == ".db":
        # SQLite database
        conn = sqlite3.connect(str(data_path))
        df = pd.read_sql_query("SELECT * FROM ohlcv ORDER BY ts", conn)
        conn.close()
        logger.info(f"Loaded {len(df)} rows from SQLite database")
    elif data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Check for required OHLCV columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure timestamp column exists and preserve it
    if "timestamp" not in df.columns:
        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        else:
            logger.warning("No timestamp column found, using index")
            df["timestamp"] = pd.date_range(start="2020-01-01", periods=len(df), freq="1H")

    # Preserve timestamp before feature computation
    timestamps = df["timestamp"].copy()

    # Compute technical features
    logger.info("Computing technical features...")
    df_features = add_technical_features(
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        volume=df["volume"].values,
        dropna_after=False  # We'll drop NaNs ourselves later
    )

    # Re-add timestamp column
    df_features["timestamp"] = timestamps

    df = df_features
    logger.info(f"Features computed. DataFrame now has {len(df.columns)} columns")

    # Drop any NaN rows (from feature computation)
    before_len = len(df)
    df = df.dropna().reset_index(drop=True)
    after_len = len(df)

    if before_len > after_len:
        logger.info(f"Dropped {before_len - after_len} rows with NaN values (from feature computation)")

    if len(df) < min_rows:
        raise ValueError(f"Insufficient data: {len(df)} rows < {min_rows} required")

    logger.info(f"Data prepared: {len(df)} rows ready for inference")

    return df


def run_example():
    """
    Run complete example of multi-horizon inference.
    """
    # ==========================================
    # Configuration
    # ==========================================

    # Paths (adjust these to your setup)
    MODEL_PATH = "artifacts/model_best.pt"
    META_PATH = "artifacts/meta.json"
    SCALER_PATH = "artifacts/scaler.pkl"
    DATA_PATH = "data_manager/exports/Binance_BTCUSDT_1h.db"  # SQLite database

    # Prediction parameters
    N_PREDICTIONS = 100  # Number of predictions to generate
    # START_IDX determines where predictions begin in the dataset:
    # - None: Start from seq_len (64), i.e., beginning of dataset (for full backtest)
    # - Negative value: Start from end of dataset (for recent/future predictions)
    # Example: To show recent context + future predictions, start ~100 steps before end
    # This will show predictions from recent past extending into true future
    START_IDX = -100  # Start 100 steps before end to show recent predictions + future

    # Output
    OUTPUT_CSV = "predictions_output.csv"
    PLOT_OUTPUT = "predictions_plot.png"
    GENERATE_PLOTS = True

    logger.info("="*70)
    logger.info("Multi-Horizon LSTM Inference Example")
    logger.info("="*70)

    # ==========================================
    # Step 1: Load Model and Artifacts
    # ==========================================
    logger.info("\nStep 1: Loading model and artifacts...")

    try:
        model, meta, scaler, device = load_model_and_artifacts(
            model_path=MODEL_PATH,
            meta_path=META_PATH,
            scaler_path=SCALER_PATH,
            device=None  # Auto-detect
        )

        logger.info("Model loaded successfully")
        logger.info(f"  - Input features: {len(meta['feature_cols'])}")
        logger.info(f"  - Sequence length: {meta['seq_len']}")
        logger.info(f"  - Prediction horizon: {meta['horizon']} steps")
        logger.info(f"  - Device: {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Please ensure model artifacts exist in 'artifacts/' directory")
        return

    # ==========================================
    # Step 2: Load and Prepare Data
    # ==========================================
    logger.info("\nStep 2: Loading and preparing data...")

    try:
        df = load_and_prepare_data(
            data_path=DATA_PATH,
            min_rows=meta['seq_len'] + meta['horizon'] + 100
        )

        # Show data summary
        logger.info("\nData Summary:")
        logger.info(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"  - Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        logger.info(f"  - Mean close: ${df['close'].mean():.2f}")

        # Check which features are available
        available_features = [f for f in meta['feature_cols'] if f in df.columns]
        missing_features = [f for f in meta['feature_cols'] if f not in df.columns]

        logger.info(f"  - Available features: {len(available_features)}/{len(meta['feature_cols'])}")
        if missing_features:
            logger.warning(f"  - Missing features: {missing_features}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # ==========================================
    # Step 3: Generate Predictions
    # ==========================================
    logger.info("\nStep 3: Generating multi-horizon predictions...")
    logger.info(f"  - Prediction method: Jump forecasting (h={meta['horizon']} steps)")
    logger.info(f"  - Number of predictions: {N_PREDICTIONS}")

    try:
        predictions_df = predict_multi_horizon_jump(
            df=df,
            model=model,
            meta=meta,
            scaler=scaler,
            device=device,
            n_predictions=N_PREDICTIONS,
            start_idx=START_IDX,
            return_features=False
        )

        logger.info(f"Generated {len(predictions_df)} predictions successfully")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # Step 4: Analyze Results
    # ==========================================
    logger.info("\nStep 4: Analyzing prediction quality...")

    # Basic statistics
    logger.info("\nPrediction Statistics:")
    logger.info(f"  - Mean predicted price: ${predictions_df['close_pred'].mean():.2f}")
    logger.info(f"  - Std predicted price: ${predictions_df['close_pred'].std():.2f}")
    logger.info(f"  - Mean predicted log-return: {predictions_df['log_return_pred'].mean():.6f}")
    logger.info(f"  - Mean predicted volatility: {predictions_df['volatility_pred'].mean():.6f}")

    # Error metrics (where actual future prices are available)
    valid_preds = predictions_df.dropna(subset=["prediction_error"])

    if len(valid_preds) > 0:
        mae = valid_preds["prediction_error"].abs().mean()
        rmse = np.sqrt((valid_preds["prediction_error"] ** 2).mean())
        mape = valid_preds["prediction_error_pct"].abs().mean()

        logger.info("\nError Metrics (where actual prices available):")
        logger.info(f"  - MAE (Mean Absolute Error): ${mae:.2f}")
        logger.info(f"  - RMSE (Root Mean Squared Error): ${rmse:.2f}")
        logger.info(f"  - MAPE (Mean Absolute % Error): {mape:.2f}%")

        # Directional accuracy
        valid_preds["predicted_direction"] = (
            predictions_df["close_pred"] > predictions_df["close_current"]
        )
        valid_preds["actual_direction"] = (
            predictions_df["close_actual_future"] > predictions_df["close_current"]
        )
        valid_preds["direction_correct"] = (
            valid_preds["predicted_direction"] == valid_preds["actual_direction"]
        )

        directional_accuracy = valid_preds["direction_correct"].mean() * 100
        logger.info(f"  - Directional Accuracy: {directional_accuracy:.1f}%")

        # Coverage of confidence intervals
        within_ci = (
            (valid_preds["close_actual_future"] >= valid_preds["lower_bound_2std"]) &
            (valid_preds["close_actual_future"] <= valid_preds["upper_bound_2std"])
        )
        coverage = within_ci.mean() * 100
        logger.info(f"  - 95% CI Coverage: {coverage:.1f}% (target: ~95%)")

    else:
        logger.warning("No predictions with actual future prices for validation")

    # ==========================================
    # Step 5: Save Results
    # ==========================================
    logger.info(f"\nStep 5: Saving results to {OUTPUT_CSV}...")

    try:
        predictions_df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Predictions saved successfully")
        logger.info(f"  - File size: {Path(OUTPUT_CSV).stat().st_size / 1024:.1f} KB")

    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")

    # ==========================================
    # Step 6: Generate Plots
    # ==========================================
    if GENERATE_PLOTS:
        logger.info(f"\nStep 6: Generating visualizations...")

        try:
            # Main prediction plot
            logger.info("  - Creating price prediction plot...")
            fig1 = plot_predictions(
                df=df,
                predictions_df=predictions_df,
                title=f"BTC/USDT Multi-Horizon Predictions (h={meta['horizon']} steps, 1h timeframe)",
                figsize=(16, 9),
                show_confidence=True,
                save_path=PLOT_OUTPUT
            )

            # Error analysis plot
            error_plot_path = PLOT_OUTPUT.replace(".png", "_errors.png")
            logger.info("  - Creating error analysis plot...")
            fig2 = plot_prediction_errors(
                predictions_df=predictions_df,
                title="Prediction Error Analysis",
                figsize=(16, 8),
                save_path=error_plot_path
            )

            logger.info("Plots generated successfully")

            # Show plots in interactive mode
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception:
                logger.info("Non-interactive mode: plots saved to files")

        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # Summary
    # ==========================================
    logger.info("\n" + "="*70)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*70)
    logger.info(f"✓ Model loaded: {MODEL_PATH}")
    logger.info(f"✓ Data loaded: {len(df)} rows")
    logger.info(f"✓ Predictions generated: {len(predictions_df)}")
    logger.info(f"✓ Output saved: {OUTPUT_CSV}")
    if GENERATE_PLOTS:
        logger.info(f"✓ Plots saved: {PLOT_OUTPUT}")
    logger.info("="*70)
    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    run_example()
