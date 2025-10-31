#!/usr/bin/env python3
"""
dashboard_visualizations_simple.py

Simplified visualization functions for real-time Status tab display.
Optimized for speed and integration with Tkinter matplotlib canvas.

Author: Claude (Anthropic)
Date: 2025-10-30
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    plt = None
    mdates = None


def plot_prediction_fan_live_simple(
    ax,
    df_history: pd.DataFrame,
    predictions_dict: Dict[int, Dict],
    show_confidence: bool = True,
    colormap: str = "viridis",
    n_history: int = 100
):
    """
    Simplified prediction fan plot for Status tab (optimized for speed).

    Args:
        ax: matplotlib axis
        df_history: Recent OHLCV data (last 100-200 rows)
        predictions_dict: {horizon: {price, ci_lower_95, ci_upper_95, ...}}
        show_confidence: Show confidence bands
        colormap: Color scheme (viridis, plasma, coolwarm, etc.)
        n_history: Number of historical candles to show

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_prediction_fan_live_simple(ax, df_history, predictions)
        >>> canvas.draw()
    """
    if plt is None:
        raise RuntimeError("matplotlib not available")

    # Limit history for performance
    df_hist = df_history.tail(n_history).copy() if len(df_history) > n_history else df_history.copy()

    # Get historical prices
    if "close" in df_hist.columns:
        hist_prices = df_hist["close"].values
    else:
        # Fallback: use last numeric column
        hist_prices = df_hist.select_dtypes(include=[np.number]).iloc[:, -1].values

    # Get timestamps
    if "timestamp" in df_hist.columns:
        hist_times = pd.to_datetime(df_hist["timestamp"])
    elif "ts" in df_hist.columns:
        hist_times = pd.to_datetime(df_hist["ts"], unit="s", utc=True)
    else:
        # Fallback: create synthetic timestamps
        hist_times = pd.date_range(end=pd.Timestamp.now(), periods=len(df_hist), freq="1H")

    # Plot historical price
    ax.plot(hist_times, hist_prices, color="black", linewidth=2.5,
            label="Historical Close", zorder=3, alpha=0.8)

    # Mark current price prominently
    current_price = hist_prices[-1]
    current_time = hist_times.iloc[-1]
    ax.scatter([current_time], [current_price], color="red", s=120,
              marker="o", edgecolors="white", linewidths=2, zorder=5,
              label=f"Current: ${current_price:,.2f}")

    # Sort horizons
    horizons = sorted(predictions_dict.keys())

    if not horizons:
        ax.text(0.5, 0.5, "No predictions available",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return

    # Color map
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / max(1, len(horizons) - 1)) for i in range(len(horizons))]

    # Plot predictions
    for i, h in enumerate(horizons):
        pred = predictions_dict[h]

        # Future time (assuming 1h timeframe)
        future_time = current_time + pd.Timedelta(hours=h)
        pred_price = pred.get("price", current_price)

        # Draw line from current to prediction
        ax.plot([current_time, future_time], [current_price, pred_price],
                color=colors[i], linewidth=2.5, alpha=0.9,
                label=f"h={h} ({pred.get('change_pct', 0):+.1f}%)",
                zorder=2)

        # Draw confidence band
        if show_confidence and "ci_lower_95" in pred and "ci_upper_95" in pred:
            ci_lower = pred["ci_lower_95"]
            ci_upper = pred["ci_upper_95"]

            # Gradient alpha (fades for longer horizons)
            alpha = max(0.05, 0.2 - (i / len(horizons)) * 0.15)

            ax.fill_between([current_time, future_time],
                           [current_price, ci_lower],
                           [current_price, ci_upper],
                           color=colors[i], alpha=alpha, zorder=1)

        # Mark prediction point
        ax.scatter([future_time], [pred_price], color=colors[i], s=80,
                  edgecolors="white", linewidths=2, zorder=4)

    # Formatting
    ax.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax.set_ylabel("Price (USD)", fontsize=11, fontweight="bold")
    ax.set_title("Multi-Horizon Price Predictions (Live)", fontsize=13, fontweight="bold")

    # Legend with smaller font
    ax.legend(loc="best", fontsize=8, framealpha=0.95, ncol=2)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Format x-axis
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=25, labelsize=9)
    except:
        pass  # Fallback if datetime formatting fails

    ax.tick_params(axis="y", labelsize=9)

    # Auto-scale y-axis with margin
    all_prices = list(hist_prices)
    for h in horizons:
        pred = predictions_dict[h]
        all_prices.append(pred.get("price", current_price))
        if show_confidence:
            all_prices.append(pred.get("ci_lower_95", current_price * 0.95))
            all_prices.append(pred.get("ci_upper_95", current_price * 1.05))

    y_min, y_max = min(all_prices), max(all_prices)
    y_margin = (y_max - y_min) * 0.12  # 12% margin
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()


# Alias for backward compatibility
plot_prediction_fan_live = plot_prediction_fan_live_simple
