#!/usr/bin/env python3
"""
dashboard_visualizations.py

Advanced visualization functions for the prediction dashboard.

Provides:
- Multi-scenario prediction fan plotting
- Probability density layers
- Confidence degradation gradients
- Professional styling and color schemes

Author: Claude (Anthropic)
Date: 2025-10-30
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib import cm
    import matplotlib.dates as mdates
    from matplotlib.patches import Polygon
    from matplotlib.collections import PolyCollection
except ImportError:
    plt = None
    Figure = None
    cm = None
    mdates = None
    Polygon = None
    PolyCollection = None

# Scipy for KDE
try:
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    stats = None
    gaussian_filter1d = None

logger = logging.getLogger(__name__)


# ==========================================
# Main Plotting Function
# ==========================================

def plot_prediction_fan_live(
    ax: Any,
    df: pd.DataFrame,
    predictions_by_scenario: Dict[str, Dict[int, pd.DataFrame]],
    show_history: bool = True,
    show_confidence: bool = True,
    show_probability: bool = True,
    color_scheme: str = "viridis",
    max_history_candles: int = 200
) -> None:
    """
    Plot live multi-horizon prediction fan with advanced features.

    This function creates a professional trading visualization with:
    - Historical price data (candlesticks or line)
    - Multiple prediction horizons as a "fan"
    - Confidence intervals with gradient transparency
    - Probability density layers at key future points
    - Support for multiple scenarios (bull/base/bear)

    Args:
        ax: Matplotlib axes object to plot on
        df: Historical DataFrame with 'close' and 'timestamp'
        predictions_by_scenario: Dict mapping scenario -> horizon -> predictions
            Example: {"base": {1: df1, 5: df5, ...}, "bull": {...}, ...}
        show_history: Whether to show historical price line
        show_confidence: Whether to show confidence bands
        show_probability: Whether to show probability density layers
        color_scheme: Matplotlib colormap name
        max_history_candles: Maximum historical candles to display
    """
    if ax is None:
        logger.error("Axes object is None")
        return

    try:
        # Get base scenario (always present)
        base_predictions = predictions_by_scenario.get("base", {})

        if not base_predictions:
            ax.text(0.5, 0.5, "No predictions available",
                   ha='center', va='center', fontsize=12, color='gray',
                   transform=ax.transAxes)
            return

        # Determine if multiple scenarios
        has_multiple_scenarios = len(predictions_by_scenario) > 1

        # --- Plot Historical Data ---
        if show_history and df is not None and len(df) > 0:
            _plot_historical_data(ax, df, max_history_candles)

        # --- Plot Predictions ---
        if has_multiple_scenarios:
            # Plot each scenario with different color family
            scenario_colors = {
                "bull": "Greens",
                "base": "Blues",
                "bear": "Reds"
            }

            for scenario, predictions in predictions_by_scenario.items():
                cmap = scenario_colors.get(scenario, color_scheme)

                _plot_single_scenario(
                    ax=ax,
                    df=df,
                    predictions_by_horizon=predictions,
                    scenario_name=scenario,
                    color_scheme=cmap,
                    show_confidence=show_confidence,
                    alpha_base=0.6
                )
        else:
            # Single scenario: use default color scheme
            _plot_single_scenario(
                ax=ax,
                df=df,
                predictions_by_horizon=base_predictions,
                scenario_name="",
                color_scheme=color_scheme,
                show_confidence=show_confidence,
                alpha_base=0.8
            )

        # --- Plot Probability Layers ---
        if show_probability and stats is not None:
            _plot_probability_density_layers(ax, df, base_predictions)

        # --- Formatting ---
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Price (USD)", fontsize=11)
        ax.set_title("Multi-Horizon Prediction Fan", fontsize=13, fontweight="bold")

        # Legend with proper positioning
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)

        ax.grid(True, alpha=0.3, linestyle="--")

        # Format x-axis for datetime
        if "timestamp" in df.columns and mdates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    except Exception as e:
        logger.error(f"Failed to plot prediction fan: {e}", exc_info=True)
        ax.text(0.5, 0.5, f"Plotting error:\n{e}",
               ha='center', va='center', fontsize=10, color='red',
               transform=ax.transAxes)


def _plot_historical_data(
    ax: Any,
    df: pd.DataFrame,
    max_candles: int
) -> None:
    """Plot historical price data."""
    # Use last N candles
    df_plot = df.iloc[-max_candles:]

    if "timestamp" in df_plot.columns:
        x_hist = pd.to_datetime(df_plot["timestamp"])
    else:
        x_hist = df_plot.index

    # Plot close price line
    ax.plot(x_hist, df_plot["close"],
           label="Historical Close",
           color="black",
           linewidth=2,
           alpha=0.7,
           zorder=10)

    # Optionally add high/low shadow
    if "high" in df_plot.columns and "low" in df_plot.columns:
        ax.fill_between(
            x_hist,
            df_plot["low"],
            df_plot["high"],
            alpha=0.1,
            color="gray",
            zorder=1
        )


def _plot_single_scenario(
    ax: Any,
    df: pd.DataFrame,
    predictions_by_horizon: Dict[int, pd.DataFrame],
    scenario_name: str,
    color_scheme: str,
    show_confidence: bool,
    alpha_base: float
) -> None:
    """Plot predictions for a single scenario."""
    horizons_sorted = sorted(predictions_by_horizon.keys())
    n_horizons = len(horizons_sorted)

    if n_horizons == 0:
        return

    # Get colormap
    if cm is not None:
        try:
            cmap = cm.get_cmap(color_scheme)
            colors = cmap(np.linspace(0.3, 0.9, n_horizons))
        except:
            # Fallback
            colors = cm.viridis(np.linspace(0.3, 0.9, n_horizons))
    else:
        colors = ["blue"] * n_horizons

    # Determine label prefix
    label_prefix = f"{scenario_name.capitalize()} " if scenario_name else ""

    # Plot each horizon
    for i, h in enumerate(horizons_sorted):
        pred_df = predictions_by_horizon[h]

        if len(pred_df) == 0:
            continue

        color = colors[i]

        # Calculate alpha based on horizon (fade for longer horizons)
        alpha_line = alpha_base * (1.0 - 0.3 * (i / max(n_horizons - 1, 1)))
        alpha_band = 0.15 * (1.0 - 0.4 * (i / max(n_horizons - 1, 1)))

        # Calculate line width (thinner for longer horizons)
        linewidth = 1.8 - 0.5 * (i / max(n_horizons - 1, 1))

        # Get x-coordinates (shifted by horizon)
        x_pred = _get_prediction_x_coords(df, pred_df, h)

        if len(x_pred) != len(pred_df):
            # Trim to match
            min_len = min(len(x_pred), len(pred_df))
            x_pred = x_pred[:min_len]
            pred_df_plot = pred_df.iloc[:min_len]
        else:
            pred_df_plot = pred_df

        # Plot prediction line
        ax.plot(x_pred, pred_df_plot["close_pred"],
               label=f"{label_prefix}h={h}",
               color=color,
               linewidth=linewidth,
               alpha=alpha_line,
               zorder=5)

        # Plot confidence bands
        if show_confidence and "upper_bound" in pred_df_plot.columns:
            ax.fill_between(
                x_pred,
                pred_df_plot["lower_bound"],
                pred_df_plot["upper_bound"],
                alpha=alpha_band,
                color=color,
                zorder=2
            )


def _get_prediction_x_coords(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    horizon: int
) -> np.ndarray:
    """Get x-coordinates for predictions (shifted by horizon)."""
    if "timestamp" in pred_df.columns:
        # Shift timestamps forward by horizon
        x_pred = pd.to_datetime(pred_df["timestamp"]) + pd.Timedelta(hours=horizon)
        return x_pred.values

    elif "index" in pred_df.columns and "timestamp" in df.columns:
        # Map indices to timestamps
        future_indices = pred_df["index"].values + horizon
        valid_mask = future_indices < len(df)

        if valid_mask.any():
            x_pred = pd.to_datetime(df.iloc[future_indices[valid_mask]]["timestamp"].values)
            return x_pred
        else:
            # Extrapolate timestamps
            base_ts = pd.to_datetime(df.iloc[pred_df["index"].values]["timestamp"].values)
            x_pred = base_ts + pd.Timedelta(hours=horizon)
            return x_pred

    else:
        # Fallback: use indices
        return pred_df["index"].values + horizon


def _plot_probability_density_layers(
    ax: Any,
    df: pd.DataFrame,
    predictions_by_horizon: Dict[int, pd.DataFrame],
    key_horizons: Optional[List[int]] = None,
    alpha: float = 0.3
) -> None:
    """
    Plot probability density layers at key future time points.

    Shows a violin-plot-like visualization of the distribution of possible
    outcomes at selected horizons.

    Args:
        ax: Matplotlib axes
        df: Historical data
        predictions_by_horizon: Predictions dict
        key_horizons: Horizons to show density for (default: 3 longest)
        alpha: Transparency for density plots
    """
    try:
        # Select key horizons (default: 3 longest)
        if key_horizons is None:
            all_horizons = sorted(predictions_by_horizon.keys())
            if len(all_horizons) <= 3:
                key_horizons = all_horizons
            else:
                # Select evenly spaced
                indices = np.linspace(0, len(all_horizons) - 1, 3, dtype=int)
                key_horizons = [all_horizons[i] for i in indices]

        # Get y-axis limits
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]

        for h in key_horizons:
            if h not in predictions_by_horizon:
                continue

            pred_df = predictions_by_horizon[h]

            if len(pred_df) == 0:
                continue

            # Get last prediction's distribution
            last_pred = pred_df.iloc[-1]

            mean_price = last_pred["close_pred"]
            volatility = last_pred.get("volatility_pred", 0.05)

            # Get x-coordinate
            x_coord = _get_prediction_x_coords(df, pred_df.iloc[-1:], h)[0]

            # Generate distribution samples (log-normal)
            # log_return ~ N(mean_log_ret, volatility)
            # price = current_price * exp(log_return)

            mean_log_ret = last_pred["log_return_pred"]
            current_price = last_pred["close_current"]

            # Sample log-returns
            log_returns = np.random.normal(mean_log_ret, volatility, 1000)
            price_samples = current_price * np.exp(log_returns)

            # Compute KDE
            if stats is not None:
                try:
                    kde = stats.gaussian_kde(price_samples)

                    # Evaluate KDE on a grid
                    price_min = max(ylim[0], price_samples.min())
                    price_max = min(ylim[1], price_samples.max())
                    price_grid = np.linspace(price_min, price_max, 100)
                    density = kde(price_grid)

                    # Normalize density to fit in plot
                    # Scale so max density corresponds to ~3% of x-axis range
                    max_density = density.max()
                    if max_density > 0:
                        # Convert time to numeric for scaling
                        if isinstance(x_coord, pd.Timestamp):
                            x_numeric = mdates.date2num(x_coord)
                            xlim = ax.get_xlim()
                            x_range = xlim[1] - xlim[0]
                            width_scale = 0.03 * x_range  # 3% of x-axis
                            density_scaled = (density / max_density) * width_scale

                            # Create violin-like shape
                            x_left = x_numeric - density_scaled / 2
                            x_right = x_numeric + density_scaled / 2

                            # Plot filled polygon
                            ax.fill_betweenx(
                                price_grid,
                                x_left,
                                x_right,
                                alpha=alpha,
                                color="purple",
                                zorder=3
                            )

                            # Add center line
                            ax.plot([x_numeric, x_numeric], [price_grid[0], price_grid[-1]],
                                   color="purple", linewidth=0.5, alpha=0.5, zorder=4)

                except Exception as e:
                    logger.warning(f"Failed to plot density for h={h}: {e}")

    except Exception as e:
        logger.warning(f"Failed to plot probability layers: {e}")


# ==========================================
# Confidence Gradient Functions
# ==========================================

def apply_confidence_gradient(
    ax: Any,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    color: str = "blue",
    n_levels: int = 5
) -> None:
    """
    Apply gradient transparency to confidence bands.

    Creates multiple nested confidence bands with decreasing alpha,
    giving a smooth gradient effect.

    Args:
        ax: Matplotlib axes
        x: X-coordinates
        y_mean: Mean prediction
        y_lower: Lower confidence bound
        y_upper: Upper confidence bound
        color: Color for gradient
        n_levels: Number of gradient levels
    """
    try:
        alphas = np.linspace(0.3, 0.05, n_levels)

        for i in range(n_levels):
            # Interpolate between mean and bounds
            factor = (i + 1) / n_levels
            y_lower_level = y_mean + factor * (y_lower - y_mean)
            y_upper_level = y_mean + factor * (y_upper - y_mean)

            ax.fill_between(
                x,
                y_lower_level,
                y_upper_level,
                alpha=alphas[i],
                color=color,
                zorder=2
            )

    except Exception as e:
        logger.error(f"Failed to apply confidence gradient: {e}")


# ==========================================
# Specialized Plotting Functions
# ==========================================

def plot_horizon_error_heatmap(
    ax: Any,
    predictions_by_horizon: Dict[int, pd.DataFrame],
    metric: str = "error_pct"
) -> None:
    """
    Plot heatmap of prediction errors across time and horizons.

    Args:
        ax: Matplotlib axes
        predictions_by_horizon: Predictions dict
        metric: Metric to plot ("error_pct", "error_abs", "volatility")
    """
    try:
        horizons = sorted(predictions_by_horizon.keys())

        # Collect data
        data_matrix = []
        timestamps = None

        for h in horizons:
            pred_df = predictions_by_horizon[h]

            if len(pred_df) == 0:
                continue

            if metric == "error_pct":
                values = pred_df["prediction_error_pct"].values
            elif metric == "error_abs":
                values = pred_df["prediction_error"].abs().values
            elif metric == "volatility":
                values = pred_df["volatility_pred"].values
            else:
                values = pred_df["prediction_error_pct"].values

            data_matrix.append(values)

            if timestamps is None:
                if "timestamp" in pred_df.columns:
                    timestamps = pd.to_datetime(pred_df["timestamp"])
                else:
                    timestamps = pred_df["index"]

        if not data_matrix:
            return

        # Pad arrays to same length
        max_len = max(len(arr) for arr in data_matrix)
        data_matrix_padded = []

        for arr in data_matrix:
            if len(arr) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(arr)] = arr
                data_matrix_padded.append(padded)
            else:
                data_matrix_padded.append(arr)

        data_matrix_array = np.array(data_matrix_padded)

        # Plot heatmap
        im = ax.imshow(
            data_matrix_array,
            aspect='auto',
            cmap='RdYlGn_r',
            interpolation='nearest'
        )

        # Set ticks
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels([f"{h}h" for h in horizons])

        # X-axis (timestamps)
        n_xticks = 10
        xtick_indices = np.linspace(0, len(timestamps) - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_indices)

        if timestamps is not None and hasattr(timestamps, 'strftime'):
            xtick_labels = [timestamps.iloc[i].strftime("%m-%d") for i in xtick_indices]
            ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

        ax.set_xlabel("Time")
        ax.set_ylabel("Horizon")
        ax.set_title(f"Prediction {metric.replace('_', ' ').title()} Heatmap")

        # Colorbar
        plt.colorbar(im, ax=ax, label=metric)

    except Exception as e:
        logger.error(f"Failed to plot heatmap: {e}")


def plot_directional_accuracy_gauge(
    ax: Any,
    predictions_by_horizon: Dict[int, pd.DataFrame]
) -> None:
    """
    Plot gauge chart showing directional accuracy for each horizon.

    Args:
        ax: Matplotlib axes
        predictions_by_horizon: Predictions dict
    """
    try:
        horizons = []
        accuracies = []

        for h in sorted(predictions_by_horizon.keys()):
            pred_df = predictions_by_horizon[h]

            valid = pred_df.dropna(subset=["prediction_error"])

            if len(valid) == 0:
                continue

            # Calculate directional accuracy
            direction_correct = (
                (valid["close_actual_future"] > valid["close_current"]) ==
                (valid["close_pred"] > valid["close_current"])
            )

            accuracy = direction_correct.mean() * 100

            horizons.append(h)
            accuracies.append(accuracy)

        if not horizons:
            return

        # Bar chart
        colors = ['green' if a >= 50 else 'red' for a in accuracies]

        ax.barh(range(len(horizons)), accuracies, color=colors, alpha=0.7)
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels([f"{h}h" for h in horizons])
        ax.set_xlabel("Directional Accuracy (%)")
        ax.set_title("Directional Accuracy by Horizon")
        ax.axvline(x=50, color='gray', linestyle='--', linewidth=1, label="Random (50%)")
        ax.set_xlim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

    except Exception as e:
        logger.error(f"Failed to plot directional accuracy: {e}")


# ==========================================
# Styling Functions
# ==========================================

def apply_dark_theme(fig: Figure, ax: Any) -> None:
    """
    Apply dark theme to matplotlib figure.

    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
    """
    try:
        # Background colors
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#2d2d2d')

        # Text colors
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        # Tick colors
        ax.tick_params(colors='white')

        # Spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        # Grid color
        ax.grid(True, alpha=0.2, color='gray')

        # Legend
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor('#2d2d2d')
            for text in legend.get_texts():
                text.set_color('white')

    except Exception as e:
        logger.error(f"Failed to apply dark theme: {e}")


def apply_professional_style(ax: Any) -> None:
    """
    Apply professional trading chart styling.

    Args:
        ax: Matplotlib axes
    """
    try:
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Tick parameters
        ax.tick_params(width=1.5, length=5)

        # Font sizes
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)

    except Exception as e:
        logger.error(f"Failed to apply professional style: {e}")


# ==========================================
# Test Function
# ==========================================

def test_visualizations():
    """Test visualization functions with dummy data."""
    try:
        # Create dummy data
        n_points = 200
        timestamps = pd.date_range(start="2025-01-01", periods=n_points, freq="1h")

        df = pd.DataFrame({
            "timestamp": timestamps,
            "close": 50000 + np.cumsum(np.random.randn(n_points) * 100),
            "high": 50500 + np.cumsum(np.random.randn(n_points) * 100),
            "low": 49500 + np.cumsum(np.random.randn(n_points) * 100)
        })

        # Create dummy predictions
        predictions = {}
        for h in [1, 5, 10, 20]:
            n_pred = 50
            pred_df = pd.DataFrame({
                "index": range(150, 150 + n_pred),
                "timestamp": timestamps[150:150 + n_pred],
                "close_current": df["close"].iloc[150:150 + n_pred].values,
                "close_pred": df["close"].iloc[150:150 + n_pred].values * (1 + np.random.randn(n_pred) * 0.02),
                "log_return_pred": np.random.randn(n_pred) * 0.01,
                "volatility_pred": np.abs(np.random.randn(n_pred) * 0.02),
                "upper_bound": df["close"].iloc[150:150 + n_pred].values * 1.05,
                "lower_bound": df["close"].iloc[150:150 + n_pred].values * 0.95
            })
            predictions[h] = pred_df

        predictions_by_scenario = {"base": predictions}

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        plot_prediction_fan_live(
            ax=ax,
            df=df,
            predictions_by_scenario=predictions_by_scenario,
            show_history=True,
            show_confidence=True,
            show_probability=True,
            color_scheme="viridis"
        )

        plt.tight_layout()
        plt.savefig("test_dashboard_viz.png", dpi=150, bbox_inches="tight")
        print("Test plot saved to test_dashboard_viz.png")

        plt.show()

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_visualizations()
