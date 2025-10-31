#!/usr/bin/env python3
"""
future_forecast_fan.py

Generate true future forecasts at multiple horizons beyond available data.

This script uses the last available data point as the starting point and
generates forecasts into the unknown future using autoregressive prediction.

Usage:
    python future_forecast_fan.py --n-steps 50 --plot
"""

import logging
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_horizon_inference import load_model_and_artifacts
from multi_horizon_fan_inference import plot_prediction_fan
from example_multi_horizon import load_and_prepare_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_future_forecasts_multi_horizon(
    df, model, meta, scaler, device, horizons, base_price=None
):
    """
    Generate forecasts at multiple horizons from the last data point.

    Args:
        df: Historical dataframe (uses last seq_len rows)
        model: Trained model
        meta: Model metadata
        scaler: Feature scaler
        device: torch device
        horizons: List of horizons to forecast
        base_price: If None, uses last close price from df

    Returns:
        DataFrame with forecasts for each horizon
    """
    import torch
    from fiboevo import prepare_input_for_model

    seq_len = meta['seq_len']
    native_horizon = meta['horizon']
    feature_cols = meta['feature_cols']

    # Get last available window
    window_df = df.iloc[-seq_len:].copy()

    if base_price is None:
        base_price = float(df.iloc[-1]['close'])

    last_timestamp = df.iloc[-1]['timestamp']

    logger.info(f"Generating forecasts from base price: ${base_price:,.2f}")
    logger.info(f"Last known timestamp: {last_timestamp}")
    logger.info(f"Forecasting at horizons: {horizons}")

    # Prepare input tensor
    input_tensor = prepare_input_for_model(
        window_df,
        feature_cols,
        seq_len,
        scaler=scaler,
        method="per_row"
    ).to(device)

    # Get native prediction
    model.eval()
    with torch.no_grad():
        pred_log_ret_native, pred_vol_native = model(input_tensor)
        pred_log_ret_native = float(pred_log_ret_native.cpu().numpy().ravel()[0])
        pred_vol_native = float(pred_vol_native.cpu().numpy().ravel()[0])

    # Generate forecasts for each horizon using scaling
    forecasts = []

    for h in horizons:
        # Scale prediction to target horizon
        scale_factor = h / native_horizon

        # Log-return scales linearly (drift assumption)
        pred_log_ret_h = pred_log_ret_native * scale_factor

        # Volatility scales with sqrt(time) (Brownian motion)
        pred_vol_h = pred_vol_native * np.sqrt(scale_factor)

        # Convert to price
        pred_price = base_price * np.exp(pred_log_ret_h)

        # Confidence intervals (95% ~ 2 std)
        upper_95 = base_price * np.exp(pred_log_ret_h + 1.96 * pred_vol_h)
        lower_95 = base_price * np.exp(pred_log_ret_h - 1.96 * pred_vol_h)

        # Future timestamp
        future_time = pd.to_datetime(last_timestamp) + pd.Timedelta(hours=h)

        forecast = {
            'horizon': h,
            'timestamp': future_time,
            'base_price': base_price,
            'predicted_price': pred_price,
            'log_return': pred_log_ret_h,
            'volatility': pred_vol_h,
            'upper_bound_95': upper_95,
            'lower_bound_95': lower_95,
            'price_change': pred_price - base_price,
            'price_change_pct': 100 * (pred_price - base_price) / base_price
        }

        forecasts.append(forecast)

        logger.info(
            f"h={h:2d}: ${pred_price:>10,.2f} "
            f"({forecast['price_change_pct']:>+6.2f}%) "
            f"95%CI: [${lower_95:,.0f}, ${upper_95:,.0f}]"
        )

    return pd.DataFrame(forecasts)


def plot_future_forecasts(df, forecasts, save_path=None):
    """
    Plot historical prices with future forecasts at multiple horizons.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot recent historical data (last 200 points for context)
    recent_df = df.iloc[-200:].copy()
    if 'timestamp' in recent_df.columns:
        x_hist = pd.to_datetime(recent_df['timestamp'])
    else:
        x_hist = recent_df.index

    ax.plot(x_hist, recent_df['close'],
            label='Historical Close', color='black', linewidth=2, alpha=0.8)

    # Plot forecast for each horizon
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(forecasts)))

    base_price = forecasts['base_price'].iloc[0]
    last_timestamp = df.iloc[-1]['timestamp']

    for idx, row in forecasts.iterrows():
        h = row['horizon']
        pred_price = row['predicted_price']
        future_time = row['timestamp']
        upper = row['upper_bound_95']
        lower = row['lower_bound_95']

        color = colors[idx]

        # Draw line from last known point to prediction
        ax.plot([last_timestamp, future_time],
                [base_price, pred_price],
                color=color, linewidth=2, alpha=0.8,
                label=f'h={h} ({row["price_change_pct"]:+.1f}%)')

        # Draw confidence band
        ax.fill_between([last_timestamp, future_time],
                        [base_price, lower],
                        [base_price, upper],
                        color=color, alpha=0.15)

        # Mark the prediction point
        ax.scatter([future_time], [pred_price],
                  color=color, s=100, zorder=5, edgecolors='white', linewidths=2)

    # Formatting
    ax.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Horizon Future Price Forecasts', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    return fig


def main():
    """
    Main function: Generate and visualize future forecasts.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Generate multi-horizon future forecasts')
    parser.add_argument('--n-steps', type=int, default=50,
                       help='Maximum horizon to forecast (hours)')
    parser.add_argument('--horizons', type=int, nargs='+', default=None,
                       help='Specific horizons to forecast (e.g., 1 5 10 20 30)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', type=str, default='future_forecasts.csv',
                       help='Output CSV file')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Multi-Horizon Future Forecasting System")
    logger.info("="*80)

    # Configuration
    MODEL_PATH = "artifacts/model_best.pt"
    META_PATH = "artifacts/meta.json"
    SCALER_PATH = "artifacts/scaler.pkl"
    DATA_PATH = "data_manager/exports/Binance_BTCUSDT_1h.db"

    # Load model
    logger.info("\nStep 1: Loading model...")
    model, meta, scaler, device = load_model_and_artifacts(
        MODEL_PATH, META_PATH, SCALER_PATH
    )
    logger.info(f"Model native horizon: {meta['horizon']} steps")

    # Load data
    logger.info("\nStep 2: Loading data...")
    df = load_and_prepare_data(DATA_PATH, min_rows=1000)
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Last data point: {df.iloc[-1]['timestamp']}")
    logger.info(f"Last close price: ${df.iloc[-1]['close']:,.2f}")

    # Determine horizons
    if args.horizons:
        horizons = sorted(args.horizons)
    else:
        # Auto-generate horizons from 1 to n_steps
        native_h = meta['horizon']
        horizons = [1, 3, 5, native_h, 15, 20, 30, 40, args.n_steps]
        horizons = sorted(list(set([h for h in horizons if h <= args.n_steps])))

    logger.info(f"\nStep 3: Generating forecasts at {len(horizons)} horizons...")

    # Generate forecasts
    forecasts = generate_future_forecasts_multi_horizon(
        df=df,
        model=model,
        meta=meta,
        scaler=scaler,
        device=device,
        horizons=horizons
    )

    # Save results
    logger.info(f"\nStep 4: Saving results to {args.output}...")
    forecasts.to_csv(args.output, index=False)
    logger.info(f"Saved {len(forecasts)} forecasts")

    # Display summary
    logger.info("\n" + "="*80)
    logger.info("FORECAST SUMMARY")
    logger.info("="*80)
    logger.info(f"Base price: ${forecasts['base_price'].iloc[0]:,.2f}")
    logger.info(f"Shortest horizon (h={horizons[0]}): ${forecasts['predicted_price'].iloc[0]:,.2f}")
    logger.info(f"Longest horizon (h={horizons[-1]}): ${forecasts['predicted_price'].iloc[-1]:,.2f}")
    logger.info(f"Overall predicted change: {forecasts['price_change_pct'].iloc[-1]:+.2f}%")
    logger.info("="*80)

    # Generate plot
    if args.plot:
        logger.info("\nStep 5: Generating visualization...")
        plot_path = args.output.replace('.csv', '.png')
        plot_future_forecasts(df, forecasts, save_path=plot_path)
        logger.info("Done!")

        try:
            import matplotlib.pyplot as plt
            plt.show()
        except:
            logger.info("Non-interactive mode: plot saved to file")


if __name__ == "__main__":
    main()
