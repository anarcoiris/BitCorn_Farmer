#!/usr/bin/env python3
"""
example_prediction_fan.py

Example demonstrating multi-horizon prediction fan system.

This script shows how to generate predictions at multiple horizons simultaneously
and visualize them as a prediction fan, useful for:
1. Understanding forecast uncertainty at different time scales
2. Comparing short-term vs long-term predictions
3. Validating model extrapolation beyond native training horizon
4. Risk assessment and portfolio planning

Usage:
------
Basic usage (default horizons: 1, 3, 5, 10, 15, 20, 30 steps):
    python example_prediction_fan.py

Custom horizons:
    python example_prediction_fan.py --horizons 1 2 5 10 20 50

With iterative method for long horizons:
    python example_prediction_fan.py --method iterative

Generate plots:
    python example_prediction_fan.py --plot

Full example:
    python example_prediction_fan.py --plot --n-predictions 200 --start-idx -200
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multi_horizon_fan_inference import (
    predict_multiple_horizons,
    plot_prediction_fan,
    plot_horizon_comparison,
    compute_summary_statistics
)
from multi_horizon_inference import load_model_and_artifacts
from example_multi_horizon import load_and_prepare_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_prediction_fan_example():
    """
    Run complete example of multi-horizon prediction fan system.
    """
    logger.info("="*80)
    logger.info("Multi-Horizon Prediction Fan Example")
    logger.info("="*80)

    # ==========================================
    # Configuration
    # ==========================================

    # Model artifacts
    MODEL_PATH = "artifacts/model_best.pt"
    META_PATH = "artifacts/meta.json"
    SCALER_PATH = "artifacts/scaler.pkl"

    # Data source
    DATA_PATH = "data_manager/exports/Binance_BTCUSDT_1h.db"

    # Horizons to predict
    # Model native horizon: 10 (from meta.json)
    # We predict at: shorter (1, 3, 5), native (10), and longer (15, 20, 30)
    HORIZONS = [1, 3, 5, 10, 15, 20, 30]

    # Prediction parameters
    N_PREDICTIONS = 100  # Number of predictions per horizon
    START_IDX = -150  # Start 150 steps before end to show predictions extending into future

    # Method for long horizons (h > native_horizon)
    # "scaling": Faster, scales prediction by time (assumes linear drift)
    # "iterative": Slower, chains predictions (better for non-linear dynamics)
    METHOD = "scaling"

    # Confidence level for intervals
    CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals

    # Output settings
    OUTPUT_DIR = Path("prediction_fan_results")
    GENERATE_PLOTS = True

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

        logger.info("Model loaded successfully:")
        logger.info(f"  - Features: {len(meta['feature_cols'])}")
        logger.info(f"  - Sequence length: {meta['seq_len']}")
        logger.info(f"  - Native horizon: {meta['horizon']} steps")
        logger.info(f"  - Hidden units: {meta['hidden']}")
        logger.info(f"  - Device: {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # ==========================================
    # Step 2: Load Data
    # ==========================================
    logger.info("\nStep 2: Loading and preparing data...")

    try:
        df = load_and_prepare_data(
            data_path=DATA_PATH,
            min_rows=meta['seq_len'] + max(HORIZONS) + 100
        )

        logger.info(f"Data loaded: {len(df)} rows")
        logger.info(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"  - Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # ==========================================
    # Step 3: Generate Multi-Horizon Predictions
    # ==========================================
    logger.info("\nStep 3: Generating predictions at multiple horizons...")
    logger.info(f"  - Horizons: {HORIZONS}")
    logger.info(f"  - Predictions per horizon: {N_PREDICTIONS}")
    logger.info(f"  - Method for long horizons: {METHOD}")
    logger.info(f"  - Confidence level: {CONFIDENCE_LEVEL*100:.0f}%")

    try:
        predictions_by_horizon = predict_multiple_horizons(
            df=df,
            model=model,
            meta=meta,
            scaler=scaler,
            device=device,
            horizons=HORIZONS,
            n_predictions=N_PREDICTIONS,
            start_idx=START_IDX,
            method=METHOD,
            confidence_level=CONFIDENCE_LEVEL
        )

        logger.info(f"\nPredictions generated for {len(predictions_by_horizon)} horizons")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # Step 4: Compute Summary Statistics
    # ==========================================
    logger.info("\nStep 4: Computing summary statistics...")

    summary_df = compute_summary_statistics(predictions_by_horizon)

    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS BY HORIZON")
    logger.info("="*80)
    logger.info("\nKey Observations:")
    logger.info("  - MAE/RMSE should increase with longer horizons (prediction becomes harder)")
    logger.info("  - Directional accuracy typically degrades for distant horizons")
    logger.info("  - Volatility predictions scale with sqrt(horizon) under random walk assumption")
    logger.info("")

    # Format summary for display
    display_df = summary_df.copy()

    # Round numeric columns
    for col in ["mae", "rmse", "mape", "directional_accuracy", "mean_pred_price", "std_pred_price", "mean_volatility"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    print(display_df.to_string(index=False))
    logger.info("="*80)

    # ==========================================
    # Step 5: Analyze Results
    # ==========================================
    logger.info("\nStep 5: Analyzing prediction quality...")

    # Check for expected patterns
    native_horizon = meta['horizon']

    # Get MAE at native horizon and longest horizon
    native_stats = summary_df[summary_df['horizon'] == native_horizon].iloc[0] if native_horizon in summary_df['horizon'].values else None
    longest_stats = summary_df[summary_df['horizon'] == max(HORIZONS)].iloc[0]

    if native_stats is not None:
        logger.info(f"\nAt native horizon (h={native_horizon}):")
        logger.info(f"  - MAE: ${native_stats['mae']:.2f}")
        logger.info(f"  - Directional Accuracy: {native_stats['directional_accuracy']:.1f}%")
        logger.info(f"  - Valid predictions: {native_stats['n_valid']}/{native_stats['n_predictions']}")

    logger.info(f"\nAt longest horizon (h={max(HORIZONS)}):")
    logger.info(f"  - MAE: ${longest_stats['mae']:.2f}")
    logger.info(f"  - Directional Accuracy: {longest_stats['directional_accuracy']:.1f}%")
    logger.info(f"  - Valid predictions: {longest_stats['n_valid']}/{longest_stats['n_predictions']}")

    if native_stats is not None and not np.isnan(native_stats['mae']) and not np.isnan(longest_stats['mae']):
        mae_ratio = longest_stats['mae'] / native_stats['mae']
        logger.info(f"\nMAE increase factor: {mae_ratio:.2f}x")
        logger.info(f"  (Expected: ~{np.sqrt(max(HORIZONS)/native_horizon):.2f}x under random walk)")

    # Check volatility scaling
    vol_native = summary_df[summary_df['horizon'] == native_horizon]['mean_volatility'].values[0] if native_horizon in summary_df['horizon'].values else None
    vol_longest = longest_stats['mean_volatility']

    if vol_native is not None and not np.isnan(vol_native):
        vol_ratio = vol_longest / vol_native
        expected_vol_ratio = np.sqrt(max(HORIZONS) / native_horizon)
        logger.info(f"\nVolatility scaling:")
        logger.info(f"  Actual: {vol_ratio:.2f}x")
        logger.info(f"  Expected (sqrt(h_ratio)): {expected_vol_ratio:.2f}x")

    # ==========================================
    # Step 6: Save Results
    # ==========================================
    logger.info(f"\nStep 6: Saving results to {OUTPUT_DIR}...")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save summary
    summary_path = OUTPUT_DIR / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  Summary saved: {summary_path}")

    # Save detailed predictions for each horizon
    for h, pred_df in predictions_by_horizon.items():
        pred_path = OUTPUT_DIR / f"predictions_horizon_{h:02d}.csv"
        pred_df.to_csv(pred_path, index=False)

    logger.info(f"  Detailed predictions saved: {len(predictions_by_horizon)} files")

    # ==========================================
    # Step 7: Generate Visualizations
    # ==========================================
    if GENERATE_PLOTS:
        logger.info("\nStep 7: Generating visualizations...")

        try:
            # Main prediction fan plot
            logger.info("  - Creating prediction fan plot...")
            fan_plot_path = OUTPUT_DIR / "prediction_fan.png"
            fig1 = plot_prediction_fan(
                df=df,
                predictions_by_horizon=predictions_by_horizon,
                title=f"Multi-Horizon Prediction Fan: BTC/USDT 1h (Native h={native_horizon})",
                figsize=(18, 10),
                show_confidence=True,
                alpha_base=0.8,
                save_path=str(fan_plot_path),
                zoom_to_predictions=False  # Show full history
            )

            # Zoomed prediction fan (recent data only)
            logger.info("  - Creating zoomed prediction fan plot...")
            fan_zoom_path = OUTPUT_DIR / "prediction_fan_zoom.png"
            fig2 = plot_prediction_fan(
                df=df,
                predictions_by_horizon=predictions_by_horizon,
                title=f"Multi-Horizon Prediction Fan (Zoomed)",
                figsize=(16, 9),
                show_confidence=True,
                alpha_base=0.85,
                save_path=str(fan_zoom_path),
                zoom_to_predictions=True  # Zoom to prediction range
            )

            # Horizon comparison plot
            logger.info("  - Creating horizon comparison plot...")
            comparison_path = OUTPUT_DIR / "horizon_metrics_comparison.png"
            fig3 = plot_horizon_comparison(
                predictions_by_horizon=predictions_by_horizon,
                metrics=["mae", "rmse", "directional_accuracy"],
                figsize=(15, 5),
                save_path=str(comparison_path)
            )

            logger.info("  Visualizations generated successfully")

            # Show plots in interactive mode
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception:
                logger.info("  Non-interactive mode: plots saved to files only")

        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # Step 8: Interpretation Guide
    # ==========================================
    logger.info("\n" + "="*80)
    logger.info("INTERPRETATION GUIDE")
    logger.info("="*80)

    logger.info("""
How to Read the Prediction Fan:
--------------------------------
1. **Color Coding**: Darker lines = shorter horizons (more reliable)
                     Lighter lines = longer horizons (more uncertain)

2. **Confidence Bands**: Shaded regions show 95% confidence intervals
                         Width increases with horizon (uncertainty grows)

3. **Divergence**: If prediction lines diverge significantly, model sees
                   different dynamics at different time scales

4. **Convergence**: If lines converge, model predicts mean reversion

Statistical Caveats:
-------------------
1. **Short Horizons (h < {native_horizon})**:
   - Predictions are scaled down from native horizon
   - Model was NOT explicitly trained for these horizons
   - Use with caution; consider training dedicated short-horizon models

2. **Native Horizon (h = {native_horizon})**:
   - Most reliable predictions (model trained exactly for this)
   - Confidence intervals properly calibrated from training

3. **Long Horizons (h > {native_horizon})**:
   - Predictions extrapolate beyond training target
   - Method "{METHOD}":
     * scaling: Assumes linear drift + random walk diffusion
     * iterative: Chains predictions (accumulates errors)
   - Quality degrades for h > 2*{native_horizon}
   - Uncertainty may be underestimated in trending markets

Quality Checks:
--------------
✓ Directional accuracy should be >50% (better than random)
✓ MAE should increase roughly with sqrt(horizon) for stationary markets
✓ Confidence interval coverage should be close to {CONFIDENCE_LEVEL*100:.0f}%
✓ Predictions should not diverge wildly from historical patterns

Recommended Actions:
-------------------
1. Focus on horizons close to native ({native_horizon} ± 5 steps)
2. For critical decisions, use only native horizon predictions
3. Treat long-horizon forecasts as scenario analysis, not point estimates
4. Consider ensemble with multiple models trained at different horizons
5. Validate predictions with out-of-sample backtesting
""".format(native_horizon=native_horizon, METHOD=METHOD, CONFIDENCE_LEVEL=CONFIDENCE_LEVEL))

    logger.info("="*80)

    # ==========================================
    # Summary
    # ==========================================
    logger.info("\n" + "="*80)
    logger.info("EXECUTION COMPLETE")
    logger.info("="*80)
    logger.info(f"✓ Model: {MODEL_PATH}")
    logger.info(f"✓ Data: {len(df)} rows")
    logger.info(f"✓ Horizons: {HORIZONS}")
    logger.info(f"✓ Predictions: {N_PREDICTIONS} per horizon")
    logger.info(f"✓ Results saved: {OUTPUT_DIR}")
    if GENERATE_PLOTS:
        logger.info(f"✓ Plots generated: {OUTPUT_DIR}")
    logger.info("="*80)
    logger.info("\nMulti-horizon prediction fan analysis completed successfully!")


if __name__ == "__main__":
    try:
        run_prediction_fan_example()
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
