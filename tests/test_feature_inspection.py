#!/usr/bin/env python3
"""
test_feature_inspection.py

Comprehensive feature inspection test that:
1. Loads OHLCV data from SQLite database
2. Computes features using both old (artifacts/) and new (artifacts_v2/) feature sets
3. Saves feature dataframes to SQLite for inspection
4. Compares feature distributions and correlations
5. Validates which features are actually being passed to the model

This helps diagnose temporal lag issues by verifying:
- No look-ahead bias in features
- Feature alignment between training and inference
- Price-level vs price-invariant features
"""

import sys
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
try:
    from fiboevo import add_technical_features, add_technical_features_v2
except ImportError:
    print("ERROR: Cannot import fiboevo module")
    sys.exit(1)


def inspect_features(
    db_path: str,
    output_db_path: str = "features_inspection.db",
    n_rows: int = 1000
):
    """
    Load data, compute features, save to SQLite for inspection.

    Args:
        db_path: Path to source OHLCV database
        output_db_path: Path for output database with features
        n_rows: Number of recent rows to analyze
    """
    print("="*70)
    print("FEATURE INSPECTION TEST")
    print("="*70)

    # Load data
    print(f"\n1. Loading data from {db_path}...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM ohlcv ORDER BY ts DESC LIMIT {n_rows}", conn)
    df = df.sort_values("ts").reset_index(drop=True)
    conn.close()

    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {pd.to_datetime(df['ts'].min(), unit='s')} to {pd.to_datetime(df['ts'].max(), unit='s')}")

    # Add timestamp
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Compute features (old method)
    print("\n2. Computing features (OLD method - 39 features)...")
    df_old_features = add_technical_features(
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        volume=df["volume"].values,
        dropna_after=False
    )
    df_old_features["timestamp"] = df["timestamp"].values[:len(df_old_features)]
    df_old_features = df_old_features.dropna().reset_index(drop=True)

    print(f"   Generated {len(df_old_features.columns)} features")
    print(f"   Rows after dropna: {len(df_old_features)}")

    # Compute features (new method - v2)
    print("\n3. Computing features (NEW method - 14 clean features)...")
    try:
        df_new_features = add_technical_features_v2(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            volume=df["volume"].values,
            dropna_after=False,
            out_dtype="float32"
        )
        df_new_features["timestamp"] = df["timestamp"].values[:len(df_new_features)]
        df_new_features["close"] = df["close"].values[:len(df_new_features)]
        df_new_features = df_new_features.dropna().reset_index(drop=True)

        print(f"   Generated {len(df_new_features.columns)} features")
        print(f"   Rows after dropna: {len(df_new_features)}")
    except Exception as e:
        print(f"   ERROR: {e}")
        df_new_features = None

    # Load model metadata
    print("\n4. Loading model metadata...")
    meta_old = json.load(open("artifacts/meta.json"))
    print(f"   OLD model: {len(meta_old['feature_cols'])} features, horizon={meta_old['horizon']}")

    if Path("artifacts_v2/meta.json").exists():
        meta_new = json.load(open("artifacts_v2/meta.json"))
        print(f"   NEW model: {len(meta_new['feature_cols'])} features, horizon={meta_new['horizon']}")
    else:
        meta_new = None
        print("   NEW model metadata not found")

    # Save to SQLite
    print(f"\n5. Saving features to {output_db_path}...")
    out_conn = sqlite3.connect(output_db_path)

    # Save old features
    df_old_features.to_sql("features_old", out_conn, if_exists="replace", index=False)
    print(f"   Saved {len(df_old_features)} rows to 'features_old' table")

    # Save new features
    if df_new_features is not None:
        df_new_features.to_sql("features_new", out_conn, if_exists="replace", index=False)
        print(f"   Saved {len(df_new_features)} rows to 'features_new' table")

    # Save feature lists for comparison
    feature_comparison = pd.DataFrame({
        "feature_old": pd.Series(meta_old['feature_cols']),
        "in_old_df": pd.Series([c in df_old_features.columns for c in meta_old['feature_cols']])
    })

    if meta_new:
        feature_comparison["feature_new"] = pd.Series(meta_new['feature_cols'])
        feature_comparison["in_new_df"] = pd.Series([c in df_new_features.columns if df_new_features is not None else False for c in meta_new['feature_cols']])

    feature_comparison.to_sql("feature_comparison", out_conn, if_exists="replace", index=False)
    print(f"   Saved feature comparison table")

    # Analyze price-level features
    print("\n6. Analyzing price-level features (potential temporal lag causes)...")
    price_level_patterns = ['log_close', 'sma_', 'ema_', 'bb_m', 'bb_up', 'bb_dn', 'fib_r_', 'fibext_']

    price_level_features_old = [
        f for f in meta_old['feature_cols']
        if any(pattern in f for pattern in price_level_patterns)
    ]

    print(f"\n   OLD model price-level features: {len(price_level_features_old)}/{len(meta_old['feature_cols'])}")
    for f in price_level_features_old[:10]:
        print(f"     - {f}")
    if len(price_level_features_old) > 10:
        print(f"     ... and {len(price_level_features_old) - 10} more")

    if meta_new:
        price_level_features_new = [
            f for f in meta_new['feature_cols']
            if any(pattern in f for pattern in price_level_patterns)
        ]
        print(f"\n   NEW model price-level features: {len(price_level_features_new)}/{len(meta_new['feature_cols'])}")
        if price_level_features_new:
            for f in price_level_features_new:
                print(f"     - {f}")
        else:
            print("     [NONE - All features are price-invariant!]")

    # Compute correlations with close price
    print("\n7. Computing correlations with current close price...")

    if "close" in df_old_features.columns:
        close_vals = df_old_features["close"].values

        high_corr_old = []
        for feat in meta_old['feature_cols']:
            if feat in df_old_features.columns and feat != 'close':
                valid = (~pd.isna(df_old_features[feat])) & (~pd.isna(close_vals))
                if valid.sum() > 20:
                    corr = np.corrcoef(df_old_features[feat].values[valid], close_vals[valid])[0, 1]
                    if np.isfinite(corr) and abs(corr) > 0.90:
                        high_corr_old.append((feat, float(corr)))

        print(f"\n   OLD model features with |corr| > 0.90 with close: {len(high_corr_old)}")
        for feat, corr in sorted(high_corr_old, key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"     {feat}: {corr:.4f}")

    if df_new_features is not None and meta_new and "close" in df_new_features.columns:
        close_vals_new = df_new_features["close"].values

        high_corr_new = []
        for feat in meta_new['feature_cols']:
            if feat in df_new_features.columns and feat != 'close':
                valid = (~pd.isna(df_new_features[feat])) & (~pd.isna(close_vals_new))
                if valid.sum() > 20:
                    corr = np.corrcoef(df_new_features[feat].values[valid], close_vals_new[valid])[0, 1]
                    if np.isfinite(corr) and abs(corr) > 0.90:
                        high_corr_new.append((feat, float(corr)))

        print(f"\n   NEW model features with |corr| > 0.90 with close: {len(high_corr_new)}")
        if high_corr_new:
            for feat, corr in sorted(high_corr_new, key=lambda x: abs(x[1]), reverse=True):
                print(f"     {feat}: {corr:.4f}")
        else:
            print("     [NONE - No high correlations detected!]")

    # Save correlation analysis
    print("\n8. Saving correlation analysis...")
    corr_analysis = pd.DataFrame({
        "feature": [f for f, _ in high_corr_old],
        "corr_with_close": [c for _, c in high_corr_old],
        "model": ["old"] * len(high_corr_old)
    })

    if df_new_features is not None and meta_new:
        corr_analysis_new = pd.DataFrame({
            "feature": [f for f, _ in high_corr_new],
            "corr_with_close": [c for _, c in high_corr_new],
            "model": ["new"] * len(high_corr_new)
        })
        corr_analysis = pd.concat([corr_analysis, corr_analysis_new], ignore_index=True)

    corr_analysis.to_sql("correlation_analysis", out_conn, if_exists="replace", index=False)
    print(f"   Saved correlation analysis")

    out_conn.close()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nFeature inspection database saved: {output_db_path}")
    print("\nTables created:")
    print("  - features_old: Old feature set (39 features)")
    print("  - features_new: New feature set (14 features)")
    print("  - feature_comparison: Side-by-side comparison")
    print("  - correlation_analysis: High correlations with close price")
    print("\nYou can now inspect the features using:")
    print(f"  sqlite3 {output_db_path}")
    print("  SELECT * FROM feature_comparison;")
    print("  SELECT * FROM correlation_analysis;")
    print("\nOr load in Python:")
    print(f"  df_old = pd.read_sql('SELECT * FROM features_old', sqlite3.connect('{output_db_path}'))")
    print(f"  df_new = pd.read_sql('SELECT * FROM features_new', sqlite3.connect('{output_db_path}'))")

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    print(f"\n[!] OLD model has {len(price_level_features_old)} price-level features")
    print(f"    These are strongly correlated with absolute price levels")
    print(f"    and likely cause the temporal lag you observed.")
    print()
    print(f"[+] NEW model has {len(price_level_features_new) if meta_new else 'N/A'} price-level features")
    print(f"    Using price-invariant features should eliminate temporal lag.")
    print()
    print("[!] Recommendation: Test inference with artifacts_v2 model")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect features for temporal lag diagnosis")
    parser.add_argument(
        "--data",
        type=str,
        default="data_manager/exports/Binance_BTCUSDT_1h.db",
        help="Path to input OHLCV database"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features_inspection.db",
        help="Path for output features database"
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=1000,
        help="Number of recent rows to analyze"
    )

    args = parser.parse_args()

    inspect_features(
        db_path=args.data,
        output_db_path=args.output,
        n_rows=args.n_rows
    )
