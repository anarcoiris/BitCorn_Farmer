#!/usr/bin/env python3
"""
TradeApp_enhancements.py

Enhancements to TradeApp.py:
1. Add ttk theme support with multiple theme options
2. Add feature inspection functionality to audit tab
3. Improve audit tab with feature correlation analysis
4. Add model comparison tools

To apply these enhancements, either:
- Import this module and call apply_enhancements(app_instance)
- Or manually merge the functions into TradeApp.py
"""

import json
import sqlite3
from pathlib import Path
from tkinter import ttk, messagebox
import threading

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None


def setup_ttk_theme(root):
    """
    Setup ttk theme for the application.

    Tries modern themes in order of preference:
    1. 'azure' (custom theme if available)
    2. 'clam' (clean modern look)
    3. 'alt' (alternative)
    4. 'default' (fallback)
    """
    style = ttk.Style(root)

    # Get available themes
    available_themes = style.theme_names()
    print(f"Available themes: {available_themes}")

    # Theme preference order
    preferred_themes = ['azure', 'alt', 'clam', 'default']

    selected_theme = None
    for theme in preferred_themes:
        if theme in available_themes:
            selected_theme = theme
            break

    if selected_theme:
        try:
            style.theme_use(selected_theme)
            print(f"Using theme: {selected_theme}")
        except Exception as e:
            print(f"Error applying theme {selected_theme}: {e}")

    # Custom style improvements
    try:
        # Notebook tabs
        style.configure('TNotebook.Tab', padding=[12, 6])
        style.map('TNotebook.Tab',
                  background=[('selected', '#0078d7'), ('!selected', '#f0f0f0')],
                  foreground=[('selected', 'white'), ('!selected', 'black')])

        # Buttons
        style.configure('TButton', padding=6)
        style.configure('Primary.TButton',
                       background='#0078d7',
                       foreground='white',
                       padding=8)

        # Frames
        style.configure('Card.TFrame', background='white', relief='raised', borderwidth=1)

        # Labels
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 11, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Error.TLabel', foreground='red')

    except Exception as e:
        print(f"Error customizing styles: {e}")

    return style


def add_feature_inspection_to_audit(app):
    """
    Add feature inspection functionality to the audit tab.

    This adds:
    - Button to run feature inspection test
    - Display of feature correlations
    - Comparison between old and new feature sets
    - Export feature dataframe to SQLite
    """

    # Find audit tab frame
    if not hasattr(app, 'nb') or not hasattr(app, 'audit_text'):
        print("ERROR: Audit tab not found in app")
        return

    # Add feature inspection section above the audit buttons
    # We'll insert it into the existing audit tab
    try:
        # Get the audit tab (it should be one of the notebook tabs)
        audit_tab = None
        for i in range(app.nb.index('end')):
            if app.nb.tab(i, 'text') == 'Audit':
                audit_tab = app.nb.nametowidget(app.nb.tabs()[i])
                break

        if not audit_tab:
            print("ERROR: Could not find Audit tab widget")
            return

        # Create feature inspection frame at the top
        from tkinter import Frame, Label, Button, LEFT

        feat_frame = Frame(audit_tab, relief='raised', borderwidth=2)
        feat_frame.pack(fill='x', padx=6, pady=(6, 0), before=audit_tab.winfo_children()[0])

        Label(feat_frame, text="Feature Inspection", font=("Arial", 11, "bold")).pack(anchor="w", padx=6, pady=4)

        btn_frame = Frame(feat_frame)
        btn_frame.pack(fill='x', padx=6, pady=4)

        Button(btn_frame, text="Inspect Features", command=lambda: _run_feature_inspection(app)).pack(side=LEFT, padx=4)
        Button(btn_frame, text="Export Features DB", command=lambda: _export_features_db(app)).pack(side=LEFT, padx=4)
        Button(btn_frame, text="Compare Models", command=lambda: _compare_models(app)).pack(side=LEFT, padx=4)

        print("Feature inspection added to audit tab")

    except Exception as e:
        print(f"ERROR adding feature inspection: {e}")
        import traceback
        traceback.print_exc()


def _run_feature_inspection(app):
    """Run feature inspection and display results in audit tab."""

    def _job():
        try:
            app._append_audit_log("\n" + "="*60)
            app._append_audit_log("FEATURE INSPECTION")
            app._append_audit_log("="*60)

            # Check if we have feature dataframe
            if not hasattr(app, 'df_features') or app.df_features is None:
                app._append_audit_log("[ERROR] No feature dataframe loaded")
                app._append_audit_log("Please load data first from the Data tab")
                return

            df_features = app.df_features.copy()
            app._append_audit_log(f"\n[INFO] Analyzing {len(df_features)} rows, {len(df_features.columns)} columns")

            # Load model metadata
            model_paths = ["artifacts/meta.json", "artifacts_v2/meta.json"]
            models_info = {}

            for model_path in model_paths:
                if Path(model_path).exists():
                    with open(model_path, 'r') as f:
                        meta = json.load(f)
                    model_name = "artifacts" if "artifacts/" in model_path else "artifacts_v2"
                    models_info[model_name] = meta
                    app._append_audit_log(f"\n[INFO] Loaded {model_name}: {len(meta['feature_cols'])} features, horizon={meta['horizon']}")

            if not models_info:
                app._append_audit_log("[WARN] No model metadata found")
                return

            # Analyze price-level features
            price_level_patterns = ['log_close', 'sma_', 'ema_', 'bb_m', 'bb_up', 'bb_dn', 'fib_r_', 'fibext_']

            for model_name, meta in models_info.items():
                price_features = [
                    f for f in meta['feature_cols']
                    if any(p in f for p in price_level_patterns)
                ]

                app._append_audit_log(f"\n[ANALYSIS] {model_name} price-level features: {len(price_features)}/{len(meta['feature_cols'])}")

                if price_features:
                    for feat in price_features[:5]:
                        app._append_audit_log(f"  - {feat}")
                    if len(price_features) > 5:
                        app._append_audit_log(f"  ... and {len(price_features) - 5} more")
                else:
                    app._append_audit_log("  [GOOD] No price-level features (price-invariant)")

            # Compute correlations with close price
            if 'close' in df_features.columns and np is not None:
                app._append_audit_log("\n[ANALYSIS] Computing correlations with close price...")

                close_vals = df_features['close'].values
                high_corr_features = []

                for model_name, meta in models_info.items():
                    model_high_corr = []

                    for feat in meta['feature_cols']:
                        if feat in df_features.columns and feat != 'close':
                            try:
                                valid = (~pd.isna(df_features[feat])) & (~pd.isna(close_vals))
                                if valid.sum() > 20:
                                    corr = np.corrcoef(df_features[feat].values[valid], close_vals[valid])[0, 1]
                                    if np.isfinite(corr) and abs(corr) > 0.90:
                                        model_high_corr.append((feat, float(corr)))
                            except Exception:
                                pass

                    if model_high_corr:
                        app._append_audit_log(f"\n[WARN] {model_name} features with |corr| > 0.90: {len(model_high_corr)}")
                        for feat, corr in sorted(model_high_corr, key=lambda x: abs(x[1]), reverse=True)[:5]:
                            app._append_audit_log(f"  {feat}: {corr:.4f}")
                    else:
                        app._append_audit_log(f"\n[GOOD] {model_name} has no high correlations with close")

            app._append_audit_log("\n" + "="*60)
            app._append_audit_log("[INFO] Feature inspection complete")
            app._append_audit_log("="*60)

        except Exception as e:
            app._append_audit_log(f"\n[ERROR] Feature inspection failed: {e}")
            import traceback
            app._append_audit_log(traceback.format_exc())

    # Run in background thread
    t = threading.Thread(target=_job, daemon=True)
    t.start()


def _export_features_db(app):
    """Export current feature dataframe to SQLite database."""

    if not hasattr(app, 'df_features') or app.df_features is None:
        messagebox.showwarning("No Data", "No feature dataframe to export")
        return

    try:
        output_path = "features_export.db"
        conn = sqlite3.connect(output_path)

        app.df_features.to_sql("features", conn, if_exists="replace", index=False)

        # Also export scaled if available
        if hasattr(app, 'df_scaled') and app.df_scaled is not None:
            app.df_scaled.to_sql("features_scaled", conn, if_exists="replace", index=False)

        conn.close()

        app._append_audit_log(f"\n[INFO] Features exported to {output_path}")
        app._append_audit_log(f"  - features table: {len(app.df_features)} rows")
        if hasattr(app, 'df_scaled') and app.df_scaled is not None:
            app._append_audit_log(f"  - features_scaled table: {len(app.df_scaled)} rows")

        messagebox.showinfo("Export Complete", f"Features saved to {output_path}")

    except Exception as e:
        app._append_audit_log(f"\n[ERROR] Export failed: {e}")
        messagebox.showerror("Export Failed", str(e))


def _compare_models(app):
    """Compare artifacts/ and artifacts_v2/ models."""

    def _job():
        try:
            app._append_audit_log("\n" + "="*60)
            app._append_audit_log("MODEL COMPARISON")
            app._append_audit_log("="*60)

            # Load both models
            models = {}
            for path in ["artifacts/meta.json", "artifacts_v2/meta.json"]:
                if Path(path).exists():
                    with open(path, 'r') as f:
                        meta = json.load(f)
                    name = "OLD" if "artifacts/" in path else "NEW"
                    models[name] = meta

            if len(models) < 2:
                app._append_audit_log("[WARN] Need both artifacts/ and artifacts_v2/ models for comparison")
                return

            old = models["OLD"]
            new = models["NEW"]

            app._append_audit_log(f"\nOLD model (artifacts/):")
            app._append_audit_log(f"  Features: {len(old['feature_cols'])}")
            app._append_audit_log(f"  Sequence length: {old['seq_len']}")
            app._append_audit_log(f"  Horizon: {old['horizon']}")
            app._append_audit_log(f"  Hidden size: {old['hidden']}")

            app._append_audit_log(f"\nNEW model (artifacts_v2/):")
            app._append_audit_log(f"  Features: {len(new['feature_cols'])}")
            app._append_audit_log(f"  Sequence length: {new['seq_len']}")
            app._append_audit_log(f"  Horizon: {new['horizon']}")
            app._append_audit_log(f"  Hidden size: {new['hidden']}")
            if 'best_val_dir_acc' in new:
                app._append_audit_log(f"  Val Dir Accuracy: {new['best_val_dir_acc']:.2f}%")

            # Feature comparison
            old_set = set(old['feature_cols'])
            new_set = set(new['feature_cols'])

            removed = old_set - new_set
            added = new_set - old_set
            common = old_set & new_set

            app._append_audit_log(f"\nFeature Changes:")
            app._append_audit_log(f"  Common features: {len(common)}")
            app._append_audit_log(f"  Removed from old: {len(removed)}")
            app._append_audit_log(f"  Added in new: {len(added)}")

            if removed:
                app._append_audit_log(f"\nRemoved features ({len(removed)}):")
                for feat in sorted(removed)[:10]:
                    app._append_audit_log(f"  - {feat}")
                if len(removed) > 10:
                    app._append_audit_log(f"  ... and {len(removed) - 10} more")

            if added:
                app._append_audit_log(f"\nAdded features ({len(added)}):")
                for feat in sorted(added):
                    app._append_audit_log(f"  + {feat}")

            app._append_audit_log("\n" + "="*60)
            app._append_audit_log("[INFO] Model comparison complete")
            app._append_audit_log("="*60)

        except Exception as e:
            app._append_audit_log(f"\n[ERROR] Model comparison failed: {e}")
            import traceback
            app._append_audit_log(traceback.format_exc())

    # Run in background
    t = threading.Thread(target=_job, daemon=True)
    t.start()


def apply_enhancements(app):
    """
    Apply all enhancements to an existing TradingAppExtended instance.

    Usage:
        from TradeApp_enhancements import apply_enhancements
        app = TradingAppExtended(root)
        apply_enhancements(app)
    """
    print("Applying TradeApp enhancements...")

    # 1. Setup ttk theme
    try:
        setup_ttk_theme(app.root)
        print("  [OK] TTK theme configured")
    except Exception as e:
        print(f"  [WARN] Theme setup failed: {e}")

    # 2. Add feature inspection to audit tab
    try:
        add_feature_inspection_to_audit(app)
        print("  [OK] Feature inspection added to audit tab")
    except Exception as e:
        print(f"  [WARN] Feature inspection addition failed: {e}")

    print("Enhancements applied successfully!")


if __name__ == "__main__":
    print(__doc__)
    print("\nThis module provides enhancements to TradeApp.py")
    print("To use, import and call apply_enhancements(app_instance)")
