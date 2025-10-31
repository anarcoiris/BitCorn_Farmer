#!/usr/bin/env python3
"""
prediction_dashboard_tab.py

Professional live multi-horizon prediction dashboard for TradeApp.

This module implements a real-time trading dashboard that displays:
- Multi-horizon prediction fans at configurable horizons [1h, 3h, 5h, 10h, 20h, 30h]
- Confidence intervals with degradation gradients
- Probability density layers for uncertainty visualization
- Multiple scenario support (bull/base/bear cases)
- Interactive controls for customization
- Real-time metrics and performance indicators

Mathematical Framework:
-----------------------
The dashboard uses the multi_horizon_fan_inference system to generate predictions
at multiple horizons simultaneously. Key features:

1. **Horizon Scaling**:
   - For h < h_native: Predictions scaled down with sqrt(h/h_native) uncertainty
   - For h == h_native: Direct model predictions (most reliable)
   - For h > h_native: Predictions scaled up with increased uncertainty

2. **Uncertainty Visualization**:
   - Confidence bands: 68%, 95%, 99% intervals
   - Gradient alpha: Decreases with distance into future
   - Line thickness: Decreases for longer horizons

3. **Multiple Scenarios**:
   - Base: Standard volatility (1.0x)
   - Bull: Reduced volatility (0.7x) - optimistic
   - Bear: Increased volatility (1.5x) - pessimistic

4. **Real-Time Updates**:
   - Asynchronous data fetching from SQLite
   - Smart caching to avoid redundant computation
   - Thread-safe GUI updates via root.after()

Author: Claude (Anthropic)
Date: 2025-10-30
"""

from __future__ import annotations

import logging
import threading
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import queue

import numpy as np
import pandas as pd

# Tkinter imports
from tkinter import *
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

# Matplotlib imports
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Ensure TkAgg backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib import cm
    import matplotlib.dates as mdates
except ImportError:
    matplotlib = None
    plt = None
    Figure = None
    FigureCanvasTkAgg = None
    cm = None
    mdates = None

# Optional dependencies
try:
    import torch
except ImportError:
    torch = None

try:
    import joblib
except ImportError:
    joblib = None

# Local modules
try:
    from dashboard_utils import (
        fetch_latest_data,
        PredictionCache,
        AsyncPredictionRunner
    )
except ImportError:
    fetch_latest_data = None
    PredictionCache = None
    AsyncPredictionRunner = None

try:
    from dashboard_visualizations import (
        plot_prediction_fan_live,
        plot_probability_density_layers,
        apply_confidence_gradient
    )
except ImportError:
    plot_prediction_fan_live = None
    plot_probability_density_layers = None
    apply_confidence_gradient = None

try:
    from multi_horizon_fan_inference import (
        predict_multiple_horizons,
        load_model_and_artifacts
    )
except ImportError:
    predict_multiple_horizons = None
    load_model_and_artifacts = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionDashboardTab:
    """
    Professional live multi-horizon prediction dashboard.

    Integrates into TradeApp as a new tab, providing real-time visualization
    of multi-horizon price predictions with confidence intervals, probability
    layers, and multiple scenario support.

    Features:
    ---------
    - Multi-horizon prediction fans at customizable horizons
    - Real-time data updates from SQLite database
    - Asynchronous prediction generation (non-blocking GUI)
    - Smart caching to avoid redundant computation
    - Interactive controls for customization
    - Advanced visualizations with confidence gradients
    - Multiple scenario support (bull/base/bear)
    - Comprehensive metrics display

    Architecture:
    -------------
    - Main thread: GUI event loop
    - Background thread: Prediction generation
    - Update loop: Periodic data refresh via root.after()
    - Thread-safe communication via queues
    """

    def __init__(
        self,
        parent_frame: Frame,
        app_instance: Any,
        config_path: str = "config/dashboard_config.json"
    ):
        """
        Initialize the prediction dashboard tab.

        Args:
            parent_frame: Parent Frame widget (the tab container)
            app_instance: Reference to main TradeApp instance
            config_path: Path to dashboard configuration JSON
        """
        self.parent = parent_frame
        self.app = app_instance
        self.config_path = Path(config_path)

        # Load configuration
        self.config = self._load_config()

        # State variables
        self.is_running = False
        self.last_update_time: Optional[float] = None
        self.current_predictions: Optional[Dict[int, pd.DataFrame]] = None
        self.df_history: Optional[pd.DataFrame] = None

        # Model artifacts (loaded from app or file)
        self.model = None
        self.meta = None
        self.scaler = None
        self.device = None

        # Prediction cache and async runner
        self.cache = PredictionCache() if PredictionCache is not None else None
        self.async_runner: Optional[AsyncPredictionRunner] = None

        # Thread-safe communication
        self.prediction_queue: queue.Queue = queue.Queue()
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # UI components (will be created in _build_ui)
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.fig: Optional[Figure] = None
        self.ax: Optional[Any] = None
        self.metrics_tree: Optional[ttk.Treeview] = None
        self.status_label: Optional[Label] = None
        self.last_update_label: Optional[Label] = None
        self.current_price_label: Optional[Label] = None

        # Control variables
        self.auto_refresh_var = BooleanVar(value=self.config.get("auto_refresh", True))
        self.update_interval_var = IntVar(value=self.config.get("update_interval_minutes", 5))
        self.show_confidence_var = BooleanVar(value=True)
        self.show_probability_var = BooleanVar(value=self.config.get("show_probability_layers", True))
        self.show_history_var = BooleanVar(value=True)
        self.color_scheme_var = StringVar(value=self.config.get("color_scheme", "viridis"))

        # Horizon selection (checkboxes)
        self.horizon_vars: Dict[int, BooleanVar] = {}
        default_horizons = self.config.get("default_horizons", [1, 3, 5, 10, 20, 30])
        for h in [1, 3, 5, 10, 15, 20, 30]:
            self.horizon_vars[h] = BooleanVar(value=(h in default_horizons))

        # Scenario selection
        self.scenarios_enabled_var = BooleanVar(
            value=self.config.get("multiple_fans_config", {}).get("enabled", False)
        )
        self.active_scenarios = ["base"]  # Always include base

        # Build UI
        self._build_ui()

        # Start update loop if auto-refresh is enabled
        if self.auto_refresh_var.get():
            self._schedule_next_update()

        logger.info("PredictionDashboardTab initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load dashboard configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded dashboard config from {self.config_path}")
                return config.get("dashboard", {})
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _save_config(self):
        """Save current configuration to JSON file."""
        try:
            config = {
                "dashboard": {
                    "auto_refresh": self.auto_refresh_var.get(),
                    "update_interval_minutes": self.update_interval_var.get(),
                    "color_scheme": self.color_scheme_var.get(),
                    "show_probability_layers": self.show_probability_var.get(),
                    "default_horizons": [h for h, var in self.horizon_vars.items() if var.get()],
                    "multiple_fans_config": {
                        "enabled": self.scenarios_enabled_var.get()
                    }
                }
            }

            # Merge with existing config if it exists
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                existing.update(config)
                config = existing

            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved dashboard config to {self.config_path}")
            messagebox.showinfo("Config Saved", "Dashboard configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            messagebox.showerror("Error", f"Failed to save config: {e}")

    def _build_ui(self):
        """Build the complete dashboard UI."""
        # Main container with PanedWindow for resizable sections
        self.paned_window = ttk.PanedWindow(self.parent, orient=HORIZONTAL)
        self.paned_window.pack(fill=BOTH, expand=True)

        # Left side: Control panel
        self.control_frame = Frame(self.paned_window, width=300)
        self.paned_window.add(self.control_frame, weight=0)

        # Right side: Visualization area
        self.viz_frame = Frame(self.paned_window)
        self.paned_window.add(self.viz_frame, weight=1)

        # Build components
        self._build_control_panel()
        self._build_visualization_area()
        self._build_metrics_panel()

    def _build_control_panel(self):
        """Build the left-side control panel."""
        # Scrollable frame for controls
        canvas_controls = Canvas(self.control_frame)
        scrollbar = ttk.Scrollbar(self.control_frame, orient=VERTICAL, command=canvas_controls.yview)
        scrollable_frame = Frame(canvas_controls)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_controls.configure(scrollregion=canvas_controls.bbox("all"))
        )

        canvas_controls.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_controls.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=RIGHT, fill=Y)
        canvas_controls.pack(side=LEFT, fill=BOTH, expand=True)

        # --- Title ---
        Label(scrollable_frame, text="Prediction Dashboard", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        # --- Model Selection ---
        model_frame = LabelFrame(scrollable_frame, text="Model Configuration", padx=10, pady=10)
        model_frame.pack(fill=X, padx=10, pady=5)

        Button(model_frame, text="Load Model from Artifacts", command=self._load_model_artifacts).pack(
            fill=X, pady=2
        )
        Button(model_frame, text="Load Model from File...", command=self._load_model_file).pack(
            fill=X, pady=2
        )

        # Model info display
        self.model_info_label = Label(model_frame, text="No model loaded", fg="gray", anchor="w")
        self.model_info_label.pack(fill=X, pady=(5, 0))

        # --- Horizon Selection ---
        horizon_frame = LabelFrame(scrollable_frame, text="Horizons (Hours)", padx=10, pady=10)
        horizon_frame.pack(fill=X, padx=10, pady=5)

        for h in sorted(self.horizon_vars.keys()):
            cb = Checkbutton(horizon_frame, text=f"{h}h", variable=self.horizon_vars[h])
            cb.pack(anchor="w")

        Button(horizon_frame, text="Select All", command=self._select_all_horizons).pack(fill=X, pady=2)
        Button(horizon_frame, text="Clear All", command=self._clear_all_horizons).pack(fill=X, pady=2)

        # --- Visualization Options ---
        viz_options_frame = LabelFrame(scrollable_frame, text="Visualization", padx=10, pady=10)
        viz_options_frame.pack(fill=X, padx=10, pady=5)

        Checkbutton(viz_options_frame, text="Show Historical Data", variable=self.show_history_var).pack(anchor="w")
        Checkbutton(viz_options_frame, text="Show Confidence Bands", variable=self.show_confidence_var).pack(anchor="w")
        Checkbutton(viz_options_frame, text="Show Probability Layers", variable=self.show_probability_var).pack(anchor="w")

        Label(viz_options_frame, text="Color Scheme:").pack(anchor="w", pady=(5, 0))
        color_schemes = ["viridis", "plasma", "coolwarm", "RdYlGn", "Blues"]
        OptionMenu(viz_options_frame, self.color_scheme_var, *color_schemes).pack(fill=X)

        # --- Scenarios (Multiple Fans) ---
        scenario_frame = LabelFrame(scrollable_frame, text="Multiple Scenarios", padx=10, pady=10)
        scenario_frame.pack(fill=X, padx=10, pady=5)

        Checkbutton(scenario_frame, text="Enable Multiple Scenarios", variable=self.scenarios_enabled_var).pack(anchor="w")
        Label(scenario_frame, text="(Bull/Base/Bear volatility)", fg="gray", font=("Arial", 8)).pack(anchor="w")

        # --- Update Settings ---
        update_frame = LabelFrame(scrollable_frame, text="Update Settings", padx=10, pady=10)
        update_frame.pack(fill=X, padx=10, pady=5)

        Checkbutton(update_frame, text="Auto-Refresh", variable=self.auto_refresh_var,
                   command=self._toggle_auto_refresh).pack(anchor="w")

        Label(update_frame, text="Interval (minutes):").pack(anchor="w", pady=(5, 0))
        Scale(update_frame, from_=1, to=60, orient=HORIZONTAL, variable=self.update_interval_var).pack(fill=X)

        # --- Action Buttons ---
        action_frame = Frame(scrollable_frame, padx=10, pady=10)
        action_frame.pack(fill=X, padx=10, pady=5)

        Button(action_frame, text="Update Now", bg="#4CAF50", fg="white", command=self._manual_update).pack(
            fill=X, pady=2
        )
        Button(action_frame, text="Clear Plot", command=self._clear_plot).pack(fill=X, pady=2)
        Button(action_frame, text="Save Config", command=self._save_config).pack(fill=X, pady=2)
        Button(action_frame, text="Export Predictions...", command=self._export_predictions).pack(fill=X, pady=2)

        # --- Status Display ---
        status_frame = LabelFrame(scrollable_frame, text="Status", padx=10, pady=10)
        status_frame.pack(fill=X, padx=10, pady=5)

        Label(status_frame, text="Current Price:").pack(anchor="w")
        self.current_price_label = Label(status_frame, text="N/A", font=("Arial", 14, "bold"), fg="blue")
        self.current_price_label.pack(anchor="w")

        Label(status_frame, text="Last Update:").pack(anchor="w", pady=(5, 0))
        self.last_update_label = Label(status_frame, text="Never", fg="gray")
        self.last_update_label.pack(anchor="w")

        Label(status_frame, text="Status:").pack(anchor="w", pady=(5, 0))
        self.status_label = Label(status_frame, text="Idle", fg="gray")
        self.status_label.pack(anchor="w")

    def _build_visualization_area(self):
        """Build the main visualization canvas."""
        # Title and toolbar
        top_frame = Frame(self.viz_frame)
        top_frame.pack(side=TOP, fill=X, padx=5, pady=5)

        Label(top_frame, text="Multi-Horizon Prediction Fan", font=("Arial", 13, "bold")).pack(side=LEFT)

        # Create matplotlib figure
        if Figure is not None and FigureCanvasTkAgg is not None:
            self.fig = Figure(figsize=(12, 7), dpi=100)
            self.ax = self.fig.add_subplot(111)

            # Initial empty plot
            self.ax.text(0.5, 0.5, "No predictions yet.\nClick 'Update Now' to generate predictions.",
                        ha='center', va='center', fontsize=12, color='gray',
                        transform=self.ax.transAxes)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price (USD)")
            self.ax.grid(True, alpha=0.3)

            # Embed in Tkinter
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True, padx=5, pady=5)
        else:
            Label(self.viz_frame, text="Matplotlib not available", fg="red").pack(pady=20)

    def _build_metrics_panel(self):
        """Build the bottom metrics table."""
        metrics_frame = LabelFrame(self.viz_frame, text="Prediction Summary", padx=5, pady=5)
        metrics_frame.pack(side=BOTTOM, fill=X, padx=5, pady=5)

        # Treeview for metrics table
        columns = ("Horizon", "Target Time", "Predicted Price", "Change ($)", "Change (%)", "95% CI", "Signal")
        self.metrics_tree = ttk.Treeview(metrics_frame, columns=columns, show="headings", height=5)

        for col in columns:
            self.metrics_tree.heading(col, text=col)
            if col == "Horizon":
                self.metrics_tree.column(col, width=80, anchor="center")
            elif col == "Target Time":
                self.metrics_tree.column(col, width=150, anchor="center")
            elif col == "Signal":
                self.metrics_tree.column(col, width=60, anchor="center")
            else:
                self.metrics_tree.column(col, width=120, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(metrics_frame, orient=VERTICAL, command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)

        self.metrics_tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

    # ==========================================
    # Model Loading
    # ==========================================

    def _load_model_artifacts(self):
        """Load model from default artifacts directory."""
        try:
            self._update_status("Loading model from artifacts...")

            artifacts_dir = Path("artifacts")
            model_path = artifacts_dir / "model_best.pt"
            meta_path = artifacts_dir / "meta.json"
            scaler_path = artifacts_dir / "scaler.pkl"

            if not all([model_path.exists(), meta_path.exists(), scaler_path.exists()]):
                raise FileNotFoundError("One or more artifact files not found in artifacts/")

            self._load_model_from_paths(str(model_path), str(meta_path), str(scaler_path))

        except Exception as e:
            logger.error(f"Failed to load model from artifacts: {e}")
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self._update_status("Error loading model")

    def _load_model_file(self):
        """Load model from user-selected file."""
        try:
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("PyTorch Model", "*.pt *.pth"), ("All Files", "*.*")]
            )

            if not model_path:
                return

            model_dir = Path(model_path).parent
            meta_path = model_dir / "meta.json"
            scaler_path = model_dir / "scaler.pkl"

            if not all([meta_path.exists(), scaler_path.exists()]):
                raise FileNotFoundError(f"meta.json or scaler.pkl not found in {model_dir}")

            self._load_model_from_paths(model_path, str(meta_path), str(scaler_path))

        except Exception as e:
            logger.error(f"Failed to load model file: {e}")
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def _load_model_from_paths(self, model_path: str, meta_path: str, scaler_path: str):
        """Load model, metadata, and scaler from specified paths."""
        try:
            if load_model_and_artifacts is None:
                raise RuntimeError("load_model_and_artifacts function not available")

            self._update_status("Loading model...")

            self.model, self.meta, self.scaler, self.device = load_model_and_artifacts(
                model_path=model_path,
                meta_path=meta_path,
                scaler_path=scaler_path,
                device=None  # Auto-detect
            )

            # Update model info display
            info_text = (f"Model loaded\n"
                        f"Seq Len: {self.meta.get('seq_len', 'N/A')}\n"
                        f"Horizon: {self.meta.get('horizon', 'N/A')}\n"
                        f"Hidden: {self.meta.get('hidden', 'N/A')}\n"
                        f"Features: {len(self.meta.get('feature_cols', []))}")
            self.model_info_label.config(text=info_text, fg="green")

            logger.info(f"Model loaded successfully from {model_path}")
            self._update_status("Model loaded successfully")
            messagebox.showinfo("Success", "Model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_info_label.config(text=f"Error: {e}", fg="red")
            raise

    # ==========================================
    # Update Logic
    # ==========================================

    def _toggle_auto_refresh(self):
        """Toggle auto-refresh on/off."""
        if self.auto_refresh_var.get():
            logger.info("Auto-refresh enabled")
            self._schedule_next_update()
        else:
            logger.info("Auto-refresh disabled")

    def _schedule_next_update(self):
        """Schedule the next automatic update."""
        if self.auto_refresh_var.get():
            interval_ms = self.update_interval_var.get() * 60 * 1000  # minutes to ms
            self.parent.after(interval_ms, self._auto_update)

    def _auto_update(self):
        """Automatic update callback."""
        if self.auto_refresh_var.get():
            self._manual_update()
            self._schedule_next_update()

    def _manual_update(self):
        """Manually trigger a prediction update."""
        if self.model is None:
            messagebox.showwarning("No Model", "Please load a model first")
            return

        # Get selected horizons
        selected_horizons = [h for h, var in self.horizon_vars.items() if var.get()]

        if not selected_horizons:
            messagebox.showwarning("No Horizons", "Please select at least one horizon")
            return

        # Start async prediction generation
        self._update_status("Generating predictions...")
        threading.Thread(target=self._run_prediction_update, args=(selected_horizons,), daemon=True).start()

    def _run_prediction_update(self, horizons: List[int]):
        """
        Run prediction update in background thread.

        This function:
        1. Fetches latest data from SQLite
        2. Generates predictions at multiple horizons
        3. Optionally generates multiple scenarios (bull/base/bear)
        4. Queues results for GUI update
        """
        try:
            # Fetch latest data
            sqlite_path = self.app.sqlite_path.get() if hasattr(self.app, 'sqlite_path') else None
            table = self.app.table.get() if hasattr(self.app, 'table') else "ohlcv"

            if not sqlite_path:
                raise ValueError("No SQLite database path configured in main app")

            if fetch_latest_data is None:
                # Fallback: use simple SQLite query
                df = self._fetch_data_simple(sqlite_path, table)
            else:
                df = fetch_latest_data(sqlite_path, table, min_rows=1000)

            logger.info(f"Fetched {len(df)} rows from database")

            # Check cache (if available)
            cache_key = (tuple(horizons), len(df), df['close'].iloc[-1] if len(df) > 0 else 0)

            if self.cache is not None:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.info("Using cached predictions")
                    self.prediction_queue.put(("success", cached_result, df))
                    return

            # Generate predictions
            scenarios_to_run = ["base"]
            if self.scenarios_enabled_var.get():
                scenarios_to_run = ["bull", "base", "bear"]

            all_scenarios = {}

            for scenario in scenarios_to_run:
                # Adjust volatility based on scenario
                if scenario == "bull":
                    vol_mult = 0.7
                elif scenario == "bear":
                    vol_mult = 1.5
                else:
                    vol_mult = 1.0

                predictions = predict_multiple_horizons(
                    df=df,
                    model=self.model,
                    meta=self.meta,
                    scaler=self.scaler,
                    device=self.device,
                    horizons=horizons,
                    n_predictions=100,  # Last 100 predictions
                    start_idx=-150,
                    method="scaling",
                    confidence_level=0.95
                )

                # Apply volatility multiplier if not base scenario
                if vol_mult != 1.0:
                    for h, pred_df in predictions.items():
                        pred_df["volatility_pred"] *= vol_mult
                        # Recalculate bounds
                        z_score = 1.96  # 95% CI
                        pred_df["upper_bound"] = pred_df["close_current"] * np.exp(
                            pred_df["log_return_pred"] + z_score * pred_df["volatility_pred"]
                        )
                        pred_df["lower_bound"] = pred_df["close_current"] * np.exp(
                            pred_df["log_return_pred"] - z_score * pred_df["volatility_pred"]
                        )

                all_scenarios[scenario] = predictions

            # Cache results
            if self.cache is not None:
                self.cache.set(cache_key, all_scenarios)

            # Queue for GUI update
            self.prediction_queue.put(("success", all_scenarios, df))

        except Exception as e:
            logger.error(f"Prediction update failed: {e}", exc_info=True)
            self.prediction_queue.put(("error", str(e), None))

        finally:
            # Schedule GUI update check
            self.parent.after(100, self._check_prediction_queue)

    def _check_prediction_queue(self):
        """Check prediction queue and update GUI if results available."""
        try:
            while not self.prediction_queue.empty():
                status, data, df = self.prediction_queue.get_nowait()

                if status == "success":
                    self.current_predictions = data
                    self.df_history = df
                    self._update_visualization()
                    self._update_metrics_table()
                    self.last_update_time = time.time()
                    self._update_status("Predictions updated successfully")
                    self._update_last_update_label()
                elif status == "error":
                    error_msg = data
                    messagebox.showerror("Prediction Error", f"Failed to generate predictions:\n{error_msg}")
                    self._update_status(f"Error: {error_msg}")
        except queue.Empty:
            pass

    def _fetch_data_simple(self, sqlite_path: str, table: str) -> pd.DataFrame:
        """Simple fallback data fetching from SQLite."""
        import sqlite3

        conn = sqlite3.connect(sqlite_path)
        query = f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 2000"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Reverse to chronological order
        df = df.iloc[::-1].reset_index(drop=True)

        # Ensure required columns
        if 'close' not in df.columns:
            raise ValueError("DataFrame missing 'close' column")

        # Add features if not present (use fiboevo if available)
        try:
            from fiboevo import add_technical_features
            df = add_technical_features(df)
        except ImportError:
            logger.warning("fiboevo not available, using raw data")

        return df

    # ==========================================
    # Visualization Updates
    # ==========================================

    def _update_visualization(self):
        """Update the main prediction fan plot."""
        if self.ax is None or self.current_predictions is None or self.df_history is None:
            return

        try:
            # Clear previous plot
            self.ax.clear()

            # Get base scenario predictions (always present)
            predictions_base = self.current_predictions.get("base", {})

            if not predictions_base:
                self.ax.text(0.5, 0.5, "No predictions available",
                           ha='center', va='center', fontsize=12, color='gray',
                           transform=self.ax.transAxes)
                self.canvas.draw()
                return

            # Use custom visualization function if available, otherwise fallback
            if plot_prediction_fan_live is not None:
                # Advanced visualization with all features
                plot_prediction_fan_live(
                    ax=self.ax,
                    df=self.df_history,
                    predictions_by_scenario=self.current_predictions,
                    show_history=self.show_history_var.get(),
                    show_confidence=self.show_confidence_var.get(),
                    show_probability=self.show_probability_var.get(),
                    color_scheme=self.color_scheme_var.get()
                )
            else:
                # Simple fallback visualization
                self._plot_simple_fan(predictions_base)

            # Refresh canvas
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Failed to update visualization: {e}", exc_info=True)
            messagebox.showerror("Visualization Error", f"Failed to update plot:\n{e}")

    def _plot_simple_fan(self, predictions_by_horizon: Dict[int, pd.DataFrame]):
        """Simple fallback plotting function."""
        if not predictions_by_horizon or self.df_history is None:
            return

        # Plot historical data
        if self.show_history_var.get():
            hist_data = self.df_history.iloc[-200:]  # Last 200 candles
            if "timestamp" in hist_data.columns:
                x_hist = pd.to_datetime(hist_data["timestamp"])
            else:
                x_hist = hist_data.index

            self.ax.plot(x_hist, hist_data["close"], label="Historical Close",
                        color="black", linewidth=2, alpha=0.7, zorder=10)

        # Plot predictions for each horizon
        horizons_sorted = sorted(predictions_by_horizon.keys())
        n_horizons = len(horizons_sorted)

        # Color map
        if cm is not None:
            colors = cm.viridis(np.linspace(0.2, 0.9, n_horizons))
        else:
            colors = ["blue"] * n_horizons

        for i, h in enumerate(horizons_sorted):
            pred_df = predictions_by_horizon[h]

            if len(pred_df) == 0:
                continue

            color = colors[i]

            # Get x-coordinates (shifted by horizon)
            if "timestamp" in pred_df.columns:
                x_pred = pd.to_datetime(pred_df["timestamp"]) + pd.Timedelta(hours=h)
            else:
                x_pred = pred_df["index"] + h

            # Plot prediction line
            self.ax.plot(x_pred, pred_df["close_pred"],
                        label=f"h={h}h",
                        color=color,
                        linewidth=1.5,
                        alpha=0.8)

            # Plot confidence bands
            if self.show_confidence_var.get() and "upper_bound" in pred_df.columns:
                self.ax.fill_between(
                    x_pred,
                    pred_df["lower_bound"],
                    pred_df["upper_bound"],
                    alpha=0.15,
                    color=color
                )

        # Formatting
        self.ax.set_xlabel("Time", fontsize=11)
        self.ax.set_ylabel("Price (USD)", fontsize=11)
        self.ax.set_title("Multi-Horizon Prediction Fan", fontsize=13, fontweight="bold")
        self.ax.legend(loc="upper left", fontsize=9, ncol=2)
        self.ax.grid(True, alpha=0.3)

        # Format x-axis for datetime
        if "timestamp" in self.df_history.columns and mdates is not None:
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            self.fig.autofmt_xdate()

    def _update_metrics_table(self):
        """Update the metrics table with current predictions."""
        if self.metrics_tree is None or self.current_predictions is None:
            return

        try:
            # Clear existing rows
            for item in self.metrics_tree.get_children():
                self.metrics_tree.delete(item)

            # Get base scenario predictions
            predictions_base = self.current_predictions.get("base", {})

            if not predictions_base:
                return

            # Get current price
            current_price = None
            if self.df_history is not None and len(self.df_history) > 0:
                current_price = float(self.df_history.iloc[-1]["close"])
                self.current_price_label.config(text=f"${current_price:,.2f}")

            # Add rows for each horizon (use most recent prediction)
            for h in sorted(predictions_base.keys()):
                pred_df = predictions_base[h]

                if len(pred_df) == 0:
                    continue

                # Get last prediction (most recent)
                last_pred = pred_df.iloc[-1]

                pred_price = last_pred["close_pred"]

                # Calculate change
                if current_price is not None:
                    change_dollars = pred_price - current_price
                    change_pct = (change_dollars / current_price) * 100
                else:
                    change_dollars = 0
                    change_pct = 0

                # Signal
                if change_pct > 0.5:
                    signal = "↑"
                elif change_pct < -0.5:
                    signal = "↓"
                else:
                    signal = "→"

                # Target time (estimate)
                if "timestamp" in pred_df.columns:
                    base_time = pd.to_datetime(last_pred["timestamp"])
                    target_time = base_time + pd.Timedelta(hours=h)
                    target_time_str = target_time.strftime("%Y-%m-%d %H:%M")
                else:
                    target_time_str = f"+{h}h"

                # Confidence interval
                lower = last_pred.get("lower_bound", pred_price)
                upper = last_pred.get("upper_bound", pred_price)
                ci_str = f"${lower:,.0f} - ${upper:,.0f}"

                # Insert row
                self.metrics_tree.insert("", "end", values=(
                    f"{h}h",
                    target_time_str,
                    f"${pred_price:,.2f}",
                    f"${change_dollars:+,.2f}",
                    f"{change_pct:+.2f}%",
                    ci_str,
                    signal
                ))

        except Exception as e:
            logger.error(f"Failed to update metrics table: {e}", exc_info=True)

    # ==========================================
    # UI Helpers
    # ==========================================

    def _update_status(self, message: str):
        """Update the status label."""
        if self.status_label is not None:
            self.status_label.config(text=message)
            logger.info(f"Dashboard status: {message}")

    def _update_last_update_label(self):
        """Update the last update timestamp label."""
        if self.last_update_label is not None and self.last_update_time is not None:
            time_str = datetime.fromtimestamp(self.last_update_time).strftime("%Y-%m-%d %H:%M:%S")
            self.last_update_label.config(text=time_str, fg="green")

    def _select_all_horizons(self):
        """Select all horizon checkboxes."""
        for var in self.horizon_vars.values():
            var.set(True)

    def _clear_all_horizons(self):
        """Clear all horizon checkboxes."""
        for var in self.horizon_vars.values():
            var.set(False)

    def _clear_plot(self):
        """Clear the prediction plot."""
        if self.ax is not None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Plot cleared.\nClick 'Update Now' to generate predictions.",
                        ha='center', va='center', fontsize=12, color='gray',
                        transform=self.ax.transAxes)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price (USD)")
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()

        # Clear metrics table
        if self.metrics_tree is not None:
            for item in self.metrics_tree.get_children():
                self.metrics_tree.delete(item)

        # Reset state
        self.current_predictions = None
        self.df_history = None
        self._update_status("Plot cleared")

    def _export_predictions(self):
        """Export current predictions to CSV."""
        if self.current_predictions is None:
            messagebox.showwarning("No Data", "No predictions to export")
            return

        try:
            output_dir = filedialog.askdirectory(title="Select Output Directory")

            if not output_dir:
                return

            output_dir = Path(output_dir)

            # Export each scenario
            for scenario, predictions_by_horizon in self.current_predictions.items():
                for h, pred_df in predictions_by_horizon.items():
                    filename = f"predictions_{scenario}_h{h:02d}.csv"
                    filepath = output_dir / filename
                    pred_df.to_csv(filepath, index=False)
                    logger.info(f"Exported {filename}")

            messagebox.showinfo("Export Complete", f"Predictions exported to:\n{output_dir}")

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"Failed to export:\n{e}")

    def cleanup(self):
        """Cleanup resources when tab is destroyed."""
        self.stop_event.set()
        if self.update_thread is not None and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        logger.info("PredictionDashboardTab cleaned up")


# ==========================================
# Standalone Test
# ==========================================

def main():
    """Standalone test of the dashboard."""
    root = Tk()
    root.title("Prediction Dashboard Test")
    root.geometry("1400x900")

    # Create dummy app instance
    class DummyApp:
        def __init__(self):
            self.sqlite_path = StringVar(value="data_manager/exports/Binance_BTCUSDT_1h.db")
            self.table = StringVar(value="ohlcv")

    dummy_app = DummyApp()

    # Create notebook
    nb = ttk.Notebook(root)
    nb.pack(fill=BOTH, expand=True)

    # Create dashboard tab
    tab = Frame(nb)
    nb.add(tab, text="Predictions Dashboard")

    dashboard = PredictionDashboardTab(
        parent_frame=tab,
        app_instance=dummy_app,
        config_path="config/dashboard_config.json"
    )

    root.mainloop()


if __name__ == "__main__":
    main()
