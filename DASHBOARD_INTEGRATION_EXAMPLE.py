#!/usr/bin/env python3
"""
DASHBOARD_INTEGRATION_EXAMPLE.py

Example showing how to integrate the Prediction Dashboard into TradeApp.

This file demonstrates:
1. Import statements to add
2. Method to add in _build_ui()
3. Complete integration example

Author: Claude (Anthropic)
Date: 2025-10-30
"""

# ==========================================
# STEP 1: Add Import at Top of TradeApp.py
# ==========================================

# Add after existing imports (around line 115 in TradeApp.py):
"""
try:
    from prediction_dashboard_tab import PredictionDashboardTab
except ImportError:
    PredictionDashboardTab = None
    logging.getLogger(__name__).warning("PredictionDashboardTab not available")
"""


# ==========================================
# STEP 2: Add Method to TradeApp Class
# ==========================================

def _build_predictions_dashboard_tab(self):
    """
    Build the Predictions Dashboard tab.

    This method should be called from _build_ui() after other tabs are created.

    Add this method to the TradeApp class in TradeApp.py.
    """
    if PredictionDashboardTab is None:
        # Fallback if module not available
        tab = Frame(self.nb)
        self.nb.add(tab, text="Predictions Dashboard")

        error_frame = Frame(tab)
        error_frame.pack(expand=True, fill=BOTH)

        Label(error_frame,
              text="Prediction Dashboard module not available",
              fg="red",
              font=("Arial", 12, "bold")).pack(pady=20)

        Label(error_frame,
              text="Ensure prediction_dashboard_tab.py is in the project directory",
              font=("Arial", 10)).pack(pady=5)

        logger.warning("PredictionDashboardTab not available")
        return

    try:
        # Create tab frame
        tab = Frame(self.nb)
        self.nb.add(tab, text="Predictions Dashboard")

        # Initialize dashboard
        self.predictions_dashboard = PredictionDashboardTab(
            parent_frame=tab,
            app_instance=self,
            config_path="config/dashboard_config.json"
        )

        logger.info("Predictions Dashboard tab initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Predictions Dashboard: {e}", exc_info=True)

        # Show error in tab
        error_frame = Frame(tab)
        error_frame.pack(expand=True, fill=BOTH)

        Label(error_frame,
              text=f"Error initializing dashboard:",
              fg="red",
              font=("Arial", 12, "bold")).pack(pady=20)

        Label(error_frame,
              text=str(e),
              font=("Arial", 10),
              wraplength=600).pack(pady=10)

        Button(error_frame,
               text="Retry",
               command=lambda: self._build_predictions_dashboard_tab()).pack(pady=10)


# ==========================================
# STEP 3: Modify _build_ui() in TradeApp
# ==========================================

# In TradeApp._build_ui() method, add the dashboard tab call:
"""
def _build_ui(self):
    top = Frame(self.root)
    top.pack(side=TOP, fill=X, padx=6, pady=6)

    # ... existing top controls ...

    # Notebook
    self.nb = ttk.Notebook(self.root)
    self.nb.pack(side=TOP, fill=BOTH, expand=True, padx=6, pady=6)

    # Existing tabs
    self._build_preview_tab()
    self._build_train_tab()
    self._build_backtest_tab()
    self._build_status_tab()
    self._build_audit_tab()

    # NEW: Add Predictions Dashboard tab
    self._build_predictions_dashboard_tab()  # <-- ADD THIS LINE

    # ... rest of existing code ...
"""


# ==========================================
# COMPLETE INTEGRATION CODE SNIPPET
# ==========================================

"""
# ---------------------------------------------------------------------------
# Complete integration code to add to TradeApp.py
# ---------------------------------------------------------------------------

# 1. At top of file (after existing imports around line 115):

try:
    from prediction_dashboard_tab import PredictionDashboardTab
except ImportError:
    PredictionDashboardTab = None
    logging.getLogger(__name__).warning("PredictionDashboardTab not available")


# 2. Add this method to TradeApp class (after _build_audit_tab or similar):

    def _build_predictions_dashboard_tab(self):
        '''Build the Predictions Dashboard tab.'''
        if PredictionDashboardTab is None:
            tab = Frame(self.nb)
            self.nb.add(tab, text="Predictions Dashboard")
            Label(tab, text="Prediction Dashboard module not available",
                  fg="red", font=("Arial", 12)).pack(pady=50)
            logger.warning("PredictionDashboardTab not available")
            return

        try:
            tab = Frame(self.nb)
            self.nb.add(tab, text="Predictions Dashboard")

            self.predictions_dashboard = PredictionDashboardTab(
                parent_frame=tab,
                app_instance=self,
                config_path="config/dashboard_config.json"
            )

            logger.info("Predictions Dashboard tab initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Predictions Dashboard: {e}", exc_info=True)
            Label(tab, text=f"Error initializing dashboard:\\n{e}",
                  fg="red", font=("Arial", 10)).pack(pady=50)


# 3. In _build_ui() method (around line 1658), after existing tab builders:

    def _build_ui(self):
        # ... existing code ...

        # Existing tabs
        self._build_preview_tab()
        self._build_train_tab()
        self._build_backtest_tab()
        self._build_status_tab()
        self._build_audit_tab()

        # NEW: Add Predictions Dashboard tab
        self._build_predictions_dashboard_tab()  # <-- ADD THIS LINE

        # ... rest of existing code ...


# 4. Optional: Add cleanup in __del__ or cleanup method:

    def cleanup(self):
        '''Cleanup resources before closing.'''
        # ... existing cleanup code ...

        # Cleanup dashboard
        if hasattr(self, 'predictions_dashboard') and self.predictions_dashboard is not None:
            try:
                self.predictions_dashboard.cleanup()
            except Exception as e:
                logger.error(f"Dashboard cleanup failed: {e}")

# ---------------------------------------------------------------------------
"""


# ==========================================
# MINIMAL STANDALONE TEST
# ==========================================

def test_dashboard_standalone():
    """
    Test the dashboard as a standalone application.

    Run this to verify dashboard works before integrating into TradeApp.
    """
    from tkinter import Tk, Frame, StringVar
    from tkinter import ttk
    import logging

    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("Prediction Dashboard Standalone Test")
    print("="*60)

    # Create root window
    root = Tk()
    root.title("Prediction Dashboard Test")
    root.geometry("1400x900")

    # Create dummy app instance (mimics TradeApp)
    class DummyApp:
        def __init__(self):
            self.sqlite_path = StringVar(
                value="data_manager/exports/Binance_BTCUSDT_1h.db"
            )
            self.table = StringVar(value="ohlcv")
            self.symbol = StringVar(value="BTCUSDT")
            self.timeframe = StringVar(value="1h")

    dummy_app = DummyApp()

    # Create notebook
    nb = ttk.Notebook(root)
    nb.pack(fill='both', expand=True)

    # Create dashboard tab
    try:
        from prediction_dashboard_tab import PredictionDashboardTab

        tab = Frame(nb)
        nb.add(tab, text="Predictions Dashboard")

        dashboard = PredictionDashboardTab(
            parent_frame=tab,
            app_instance=dummy_app,
            config_path="config/dashboard_config.json"
        )

        print("\n✓ Dashboard initialized successfully!")
        print("\nInstructions:")
        print("1. Click 'Load Model from Artifacts'")
        print("2. Select horizons (e.g., check 1h, 5h, 10h)")
        print("3. Click 'Update Now'")
        print("4. Wait for predictions to generate")
        print("5. View the prediction fan and metrics table")
        print("\nClose window to exit test.\n")

    except ImportError as e:
        print(f"\n✗ Failed to import PredictionDashboardTab: {e}")
        print("\nEnsure these files exist in the project directory:")
        print("  - prediction_dashboard_tab.py")
        print("  - dashboard_utils.py")
        print("  - dashboard_visualizations.py")
        print("  - config/dashboard_config.json")
        return False

    except Exception as e:
        print(f"\n✗ Failed to initialize dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run GUI event loop
    root.mainloop()
    return True


# ==========================================
# VERIFICATION CHECKLIST
# ==========================================

def print_integration_checklist():
    """Print checklist for integration verification."""

    checklist = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║         PREDICTION DASHBOARD INTEGRATION CHECKLIST            ║
    ╚═══════════════════════════════════════════════════════════════╝

    BEFORE INTEGRATION:
    ✓ [ ] All files present:
        - prediction_dashboard_tab.py
        - dashboard_utils.py
        - dashboard_visualizations.py
        - config/dashboard_config.json
        - multi_horizon_fan_inference.py (existing)
        - multi_horizon_inference.py (existing)

    ✓ [ ] Dependencies installed:
        pip install torch numpy pandas matplotlib scipy scikit-learn joblib

    ✓ [ ] Model artifacts exist:
        - artifacts/model_best.pt
        - artifacts/meta.json
        - artifacts/scaler.pkl

    ✓ [ ] Database exists:
        - data_manager/exports/Binance_BTCUSDT_1h.db

    INTEGRATION STEPS:
    ✓ [ ] Added import statement to TradeApp.py (line ~115)
    ✓ [ ] Added _build_predictions_dashboard_tab() method to TradeApp class
    ✓ [ ] Called _build_predictions_dashboard_tab() in _build_ui() method
    ✓ [ ] (Optional) Added cleanup() call for dashboard

    TESTING:
    ✓ [ ] Run standalone test: python DASHBOARD_INTEGRATION_EXAMPLE.py
    ✓ [ ] Launch TradeApp: python TradeApp.py
    ✓ [ ] Navigate to "Predictions Dashboard" tab
    ✓ [ ] Load model from artifacts
    ✓ [ ] Select horizons (1h, 5h, 10h)
    ✓ [ ] Click "Update Now"
    ✓ [ ] Verify predictions display correctly
    ✓ [ ] Check metrics table populated
    ✓ [ ] Test auto-refresh (enable and wait 5 min)
    ✓ [ ] Test export (click "Export Predictions...")

    TROUBLESHOOTING:
    - If imports fail: Check file paths and Python path
    - If model fails: Verify artifacts/ directory structure
    - If predictions fail: Check database path in main app
    - If plot empty: Click "Update Now" button
    - If errors: Check logs/ directory for details

    For detailed help, see PREDICTION_DASHBOARD_GUIDE.md
    """

    print(checklist)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import sys

    print("\n" + "="*60)
    print("PREDICTION DASHBOARD INTEGRATION HELPER")
    print("="*60 + "\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run standalone test
        print("Running standalone test...\n")
        success = test_dashboard_standalone()
        sys.exit(0 if success else 1)

    elif len(sys.argv) > 1 and sys.argv[1] == "--checklist":
        # Show checklist
        print_integration_checklist()

    else:
        # Show usage
        print("Usage:")
        print("  python DASHBOARD_INTEGRATION_EXAMPLE.py --test")
        print("      Run standalone test of dashboard")
        print()
        print("  python DASHBOARD_INTEGRATION_EXAMPLE.py --checklist")
        print("      Show integration checklist")
        print()
        print("  python DASHBOARD_INTEGRATION_EXAMPLE.py")
        print("      Show this help message")
        print()
        print("For full integration instructions, see:")
        print("  PREDICTION_DASHBOARD_GUIDE.md")
        print()
        print_integration_checklist()
