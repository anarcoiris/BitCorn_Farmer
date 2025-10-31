#!/usr/bin/env python3
"""
Analyze temporal lag issue in predictions.

This script investigates:
1. Whether predictions are being projected to correct future timestamps
2. Whether the lag is in visualization or in model predictions
3. What features are used and if they contain look-ahead bias
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_predictions():
    """Analyze temporal alignment of predictions."""

    print("="*70)
    print("TEMPORAL LAG ANALYSIS")
    print("="*70)

    # Load predictions
    df = pd.read_csv("future_predictions.csv")
    print(f"\nLoaded {len(df)} predictions")
    print(f"Types: {df['prediction_type'].value_counts().to_dict()}")

    # Focus on jump predictions with actual values
    recent = df[df['prediction_type'] == 'jump_historical'].copy()
    valid = recent.dropna(subset=['prediction_error'])

    print(f"\nJump predictions with actuals: {len(valid)}")

    if len(valid) > 0:
        print("\n" + "-"*70)
        print("TEMPORAL ALIGNMENT CHECK")
        print("-"*70)
        print("\nFirst 10 predictions:")
        print("Format: t -> prediction@(t+horizon) vs actual@(t+horizon)")
        print()

        for i in range(min(10, len(valid))):
            row = valid.iloc[i]
            t_base = pd.to_datetime(row['timestamp'])
            t_pred = t_base + pd.Timedelta(hours=10)  # horizon=10

            print(f"{i+1}. Base time: {t_base}")
            print(f"   Prediction target: {t_pred}")
            print(f"   Predicted: ${row['close_pred']:.2f}")
            print(f"   Actual:    ${row['close_actual_future']:.2f}")
            print(f"   Error:     ${row['prediction_error']:.2f} ({row['prediction_error_pct']:.2f}%)")
            print()

        # Check directional accuracy
        recent['pred_direction'] = (recent['close_pred'] > recent['close_current']).astype(int)
        recent['actual_direction'] = (recent['close_actual_future'] > recent['close_current']).astype(int)
        valid_dir = recent.dropna(subset=['pred_direction', 'actual_direction'])

        if len(valid_dir) > 0:
            dir_acc = (valid_dir['pred_direction'] == valid_dir['actual_direction']).mean() * 100
            print(f"\nDirectional Accuracy: {dir_acc:.2f}%")

            # Analyze errors
            mae = valid['prediction_error'].abs().mean()
            rmse = np.sqrt((valid['prediction_error'] ** 2).mean())
            mape = valid['prediction_error_pct'].abs().mean()

            print(f"MAE: ${mae:.2f}")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAPE: {mape:.2f}%")

    # Analyze feature composition
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)

    import json

    # Current model features
    with open("artifacts/meta.json", "r") as f:
        meta_old = json.load(f)

    print(f"\nCurrent model (artifacts/):")
    print(f"  Features: {len(meta_old['feature_cols'])}")
    print(f"  Horizon: {meta_old['horizon']}")
    print(f"  Seq len: {meta_old['seq_len']}")

    # Check for problematic features
    problematic = []
    feature_cols = meta_old['feature_cols']

    # Features that leak price level
    price_leak_features = [f for f in feature_cols if any(x in f for x in ['log_close', 'sma_', 'ema_', 'bb_m', 'bb_up', 'bb_dn', 'fib_r_', 'fibext_'])]
    if price_leak_features:
        problematic.extend(price_leak_features)
        print(f"\n[!] PRICE LEVEL FEATURES (may cause lag): {len(price_leak_features)}")
        for f in price_leak_features[:10]:
            print(f"    - {f}")
        if len(price_leak_features) > 10:
            print(f"    ... and {len(price_leak_features) - 10} more")

    # Duplicate information
    print(f"\n[!] POTENTIAL DUPLICATES:")
    print(f"  - log_ret_1 vs ret_1: Both represent 1-period returns")
    print(f"  - log_ret_5 vs ret_5: Both represent 5-period returns")
    print(f"  - sma_X vs ema_X: Highly correlated moving averages")
    print(f"  - fib_r_X vs dist_fib_r_X: Redundant Fibonacci info")

    # Compare with v2
    if Path("artifacts_v2/meta.json").exists():
        with open("artifacts_v2/meta.json", "r") as f:
            meta_v2 = json.load(f)

        print(f"\n\nCleaned model (artifacts_v2/):")
        print(f"  Features: {len(meta_v2['feature_cols'])} (removed {len(meta_old['feature_cols']) - len(meta_v2['feature_cols'])})")
        print(f"  Horizon: {meta_v2['horizon']}")
        print(f"  Val Dir Acc: {meta_v2.get('best_val_dir_acc', 'N/A')}")

        print(f"\n  Cleaned features:")
        for f in meta_v2['feature_cols']:
            print(f"    - {f}")

        # Removed features
        removed = set(meta_old['feature_cols']) - set(meta_v2['feature_cols'])
        print(f"\n  Removed features ({len(removed)}):")
        for f in sorted(removed)[:15]:
            print(f"    - {f}")
        if len(removed) > 15:
            print(f"    ... and {len(removed) - 15} more")

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    print("\n1. [!] Current model uses {0} features including price-level features".format(len(meta_old['feature_cols'])))
    print("   These features (log_close, sma_X, ema_X, fib_r_X) are strongly")
    print("   correlated with absolute price and may cause temporal lag.")
    print()
    print("2. [!] Duplicate/redundant features present:")
    print("   - Both log_ret and ret versions")
    print("   - Multiple MA types (SMA + EMA)")
    print("   - Both Fib levels and distances")
    print()
    print("3. [+] Cleaned model (artifacts_v2) available with 14 features")
    print("   Uses only price-invariant features (returns, ratios, indicators)")
    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nThe temporal lag is likely caused by:")
    print("  * Price-level features (log_close, MAs, Fib levels) creating")
    print("    strong correlation with historical prices")
    print("  * Model learning to predict 'price + small adjustment' rather")
    print("    than true future dynamics")
    print()
    print("Solution:")
    print("  1. Test with artifacts_v2 model (14 clean features)")
    print("  2. If still lagging, retrain from scratch")
    print("  3. Use only price-invariant features (returns, ratios, volatility)")
    print()

if __name__ == "__main__":
    analyze_predictions()
