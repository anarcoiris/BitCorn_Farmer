#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_multi_horizon_integration.py

Quick integration test for multi-horizon prediction dashboard.
Tests:
1. Import all required modules
2. Load model artifacts
3. Generate single-point multi-horizon predictions
4. Verify predictions structure
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Multi-Horizon Integration Test")
print("=" * 70)

# Test 1: Import modules
print("\n[1/4] Testing imports...")
try:
    from multi_horizon_fan_inference import predict_single_point_multi_horizon
    from dashboard_visualizations_simple import plot_prediction_fan_live_simple
    print("[OK] Modules imported successfully")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Load artifacts
print("\n[2/4] Loading model artifacts...")
try:
    import json
    import torch
    import joblib

    artifacts_dir = parent_dir / "artifacts"
    meta_path = artifacts_dir / "meta.json"
    model_path = artifacts_dir / "model_best.pt"
    scaler_path = artifacts_dir / "scaler.pkl"

    if not meta_path.exists():
        print(f"[FAIL] Meta file not found: {meta_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"[FAIL] Model file not found: {model_path}")
        sys.exit(1)

    if not scaler_path.exists():
        print(f"[FAIL] Scaler file not found: {scaler_path}")
        sys.exit(1)

    # Load meta
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load model
    import fiboevo
    hidden = meta.get("hidden", 64)
    n_features = len(meta.get("feature_cols", []))

    model = fiboevo.LSTM2Head(input_size=n_features, hidden_size=hidden)
    checkpoint = torch.load(model_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"[OK] Loaded model: {n_features} features, hidden={hidden}")
    print(f"[OK] Loaded scaler and meta")

except Exception as e:
    print(f"[FAIL] Artifact loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load recent data and generate predictions
print("\n[3/4] Generating multi-horizon predictions...")
try:
    import sqlite3
    import pandas as pd
    from core.feature_registry import FEATURE_REGISTRY, detect_system_from_meta

    # Try multiple database files
    db_candidates = [
        "Binance_BTCUSDT_1h.db",
        "marketdata_base.db",
        "marketdata_replica.db"
    ]

    db_path = None
    for db_file in db_candidates:
        candidate = parent_dir / "data_manager" / "exports" / db_file
        if candidate.exists():
            db_path = str(candidate)
            print(f"     Trying database: {db_file}")
            break

    if db_path is None:
        print(f"[FAIL] No database file found. Tried: {db_candidates}")
        sys.exit(1)

    # Detect available tables in database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    available_tables = [row[0] for row in cursor.fetchall()]
    print(f"     Available tables: {available_tables}")

    # Prefer ohlcv, fallback to aggtrade
    if "ohlcv" in available_tables:
        table = "ohlcv"
    elif "aggtrade" in available_tables:
        table = "aggtrade"
    else:
        print(f"[FAIL] No suitable table found. Available: {available_tables}")
        conn.close()
        sys.exit(1)

    print(f"     Using table: {table}")

    # Get timeframe from meta.json
    timeframe = meta.get("timeframe", "1h")  # Default to 1h if not in meta
    print(f"     Model timeframe: {timeframe}")

    # Load data (no timeframe filter for aggtrade which may not have that column)
    if "timeframe" in pd.read_sql_query(f"PRAGMA table_info({table})", conn)["name"].values:
        query = f"""
            SELECT * FROM {table}
            WHERE symbol = 'BTCUSDT' AND timeframe = ?
            ORDER BY ts DESC
            LIMIT 1000
        """
        df = pd.read_sql_query(query, conn, params=[timeframe])
    else:
        # aggtrade table - no timeframe column
        query = f"""
            SELECT * FROM {table}
            WHERE symbol = 'BTCUSDT'
            ORDER BY ts DESC
            LIMIT 1000
        """
        df = pd.read_sql_query(query, conn)

    conn.close()

    if df.empty:
        print(f"[FAIL] No data found in {table} for BTCUSDT")
        sys.exit(1)

    # Reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)

    # Auto-detect feature system from meta.json
    feature_system = detect_system_from_meta(meta)
    if feature_system is None:
        print("[WARN] Could not auto-detect feature system, defaulting to v2")
        feature_system = "v2"

    print(f"     Using feature system: {feature_system}")

    # Compute features using feature registry
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values if "high" in df.columns else None
    low = df["low"].astype(float).values if "low" in df.columns else None
    volume = df["volume"].astype(float).values if "volume" in df.columns else None

    df_feats = FEATURE_REGISTRY.compute_features(
        close, high=high, low=low, volume=volume,
        system_name=feature_system,
        dropna_after=True
    )

    # Attach metadata
    for col in ("timestamp", "ts", "close"):
        if col in df.columns and col not in df_feats.columns:
            df_feats[col] = df[col].values[-len(df_feats):]

    df_feats = df_feats.reset_index(drop=True)

    # Generate predictions
    horizons = [1, 3, 5, 10, 15, 20, 30]
    device = torch.device("cpu")

    predictions = predict_single_point_multi_horizon(
        df=df_feats,
        model=model,
        meta=meta,
        scaler=scaler,
        device=device,
        horizons=horizons,
        method="scaling"
    )

    current_price = float(df_feats.iloc[-1]["close"])
    print(f"[OK] Generated predictions for horizons: {horizons}")
    print(f"     Current price: ${current_price:,.2f}")

except Exception as e:
    print(f"[FAIL] Prediction generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify predictions structure
print("\n[4/4] Validating predictions structure...")
try:
    required_keys = ["horizon", "price", "log_return", "volatility",
                     "ci_lower_95", "ci_upper_95", "change_usd", "change_pct"]

    for h in horizons:
        if h not in predictions:
            print(f"[FAIL] Missing horizon {h} in predictions")
            sys.exit(1)

        pred = predictions[h]
        missing = [k for k in required_keys if k not in pred]
        if missing:
            print(f"[FAIL] Horizon {h} missing keys: {missing}")
            sys.exit(1)

    print("[OK] All predictions have correct structure")

    # Display summary
    print("\n" + "=" * 70)
    print("Predictions Summary:")
    print("=" * 70)
    print(f"{'Horizon':<10} {'Price':<12} {'Change %':<12} {'95% CI':<25}")
    print("-" * 70)

    for h in horizons:
        pred = predictions[h]
        price = pred["price"]
        change_pct = pred["change_pct"]
        ci_lower = pred["ci_lower_95"]
        ci_upper = pred["ci_upper_95"]

        direction = "UP" if change_pct > 0 else "DN" if change_pct < 0 else "=="
        print(f"h={h:<8} ${price:<11,.2f} {direction} {change_pct:<10.2f}% "
              f"[${ci_lower:,.0f}, ${ci_upper:,.0f}]")

    print("=" * 70)
    print("\n[SUCCESS] All tests passed!")
    print("\nIntegration test successful. The multi-horizon prediction system is ready.")
    print("\nNext steps:")
    print("1. Start TradeApp: py -3.10 TradeApp.py")
    print("2. Go to Status tab")
    print("3. Start daemon")
    print("4. Enable 'Multi-Horizon Mode' checkbox")
    print("5. Watch the prediction fan update every 5 seconds")

except Exception as e:
    print(f"[FAIL] Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
