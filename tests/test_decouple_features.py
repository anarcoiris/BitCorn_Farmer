#!/usr/bin/env python3
"""
test_decouple_features.py

Test script to verify feature decoupling reduces correlation with close.

Tests:
1. Generate synthetic OHLCV data
2. Create features with decouple_from_close=False (baseline)
3. Create features with decouple_from_close=True (decoupled)
4. Compare correlations with close
5. Verify decoupled features have lower correlation

Usage:
    python test_decouple_features.py
"""

import sys
import numpy as np
import pandas as pd

def test_decouple_features():
    """Test that decoupling reduces correlation with close"""
    try:
        import fiboevo as fibo
    except ImportError:
        print("❌ FAIL: Cannot import fiboevo module")
        return False

    print("="*60)
    print("FEATURE DECOUPLING TEST")
    print("="*60)
    print()

    # Generate synthetic OHLCV data
    np.random.seed(42)
    n = 500
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)  # Random walk
    noise = np.abs(np.random.randn(n) * 0.5)
    high = close + noise
    low = close - noise
    volume = np.random.lognormal(10, 1, n)

    print(f"✓ Generated synthetic OHLCV data: {n} rows")
    print()

    # Test 1: Features WITHOUT decoupling (baseline)
    print("--- Test 1: Baseline features (decouple_from_close=False) ---")
    df_baseline = fibo.add_technical_features(
        close, high=high, low=low, volume=volume,
        decouple_from_close=False
    )

    # Select numeric features (exclude core OHLC and metadata)
    exclude = {"timestamp", "open", "close", "symbol", "timeframe", "exchange"}
    numeric_cols = df_baseline.select_dtypes(include=[np.number]).columns.tolist()
    baseline_features = [c for c in numeric_cols if c not in exclude and c in df_baseline.columns]

    # Compute correlations with close
    baseline_close = df_baseline["close"].values
    baseline_corrs = {}
    for col in baseline_features:
        valid = (~pd.isna(df_baseline[col])) & (~pd.isna(baseline_close))
        if valid.sum() > 20:
            corr = np.corrcoef(df_baseline[col].values[valid], baseline_close[valid])[0, 1]
            if np.isfinite(corr):
                baseline_corrs[col] = abs(corr)

    # Find high correlation features (>0.90)
    high_corr_baseline = {k: v for k, v in baseline_corrs.items() if v > 0.90}
    print(f"  Total features: {len(baseline_features)}")
    print(f"  High correlation (>0.90): {len(high_corr_baseline)}")
    if high_corr_baseline:
        print(f"  High corr features: {list(high_corr_baseline.keys())[:10]}...")  # Show first 10
    print()

    # Test 2: Features WITH decoupling
    print("--- Test 2: Decoupled features (decouple_from_close=True) ---")
    df_decoupled = fibo.add_technical_features(
        close, high=high, low=low, volume=volume,
        decouple_from_close=True
    )

    # Select numeric features
    decoupled_numeric = df_decoupled.select_dtypes(include=[np.number]).columns.tolist()
    decoupled_features = [c for c in decoupled_numeric if c not in exclude and c in df_decoupled.columns]

    # Compute correlations with close
    decoupled_close = df_decoupled["close"].values
    decoupled_corrs = {}
    for col in decoupled_features:
        valid = (~pd.isna(df_decoupled[col])) & (~pd.isna(decoupled_close))
        if valid.sum() > 20:
            corr = np.corrcoef(df_decoupled[col].values[valid], decoupled_close[valid])[0, 1]
            if np.isfinite(corr):
                decoupled_corrs[col] = abs(corr)

    # Find high correlation features (>0.90)
    high_corr_decoupled = {k: v for k, v in decoupled_corrs.items() if v > 0.90}
    print(f"  Total features: {len(decoupled_features)}")
    print(f"  High correlation (>0.90): {len(high_corr_decoupled)}")
    if high_corr_decoupled:
        print(f"  High corr features: {list(high_corr_decoupled.keys())[:10]}...")
    print()

    # Test 3: Verify specific features were decoupled
    print("--- Test 3: Verify specific feature transformations ---")
    checks = []

    # Check 1: high/low removed, new features created
    if "high" not in df_decoupled.columns and "low" not in df_decoupled.columns:
        if "hl_spread_norm" in df_decoupled.columns or "close_hl_position" in df_decoupled.columns:
            print("  ✓ high/low replaced with normalized ratios")
            checks.append(True)
        else:
            print("  ✗ high/low removed but replacement features missing")
            checks.append(False)
    else:
        print("  ✗ high/low still present in decoupled features")
        checks.append(False)

    # Check 2: Absolute MAs removed
    has_absolute_ma = any("sma_" in col or "ema_" in col for col in decoupled_features
                          if not col.endswith("_dist") and not col.endswith("_ratio"))
    if not has_absolute_ma:
        print("  ✓ Absolute moving averages removed")
        checks.append(True)
    else:
        print("  ✗ Some absolute moving averages still present")
        checks.append(False)

    # Check 3: Distance features created
    has_dist_features = any("_dist" in col for col in decoupled_features)
    if has_dist_features:
        print("  ✓ Normalized distance features created")
        checks.append(True)
    else:
        print("  ✗ No normalized distance features found")
        checks.append(False)

    # Check 4: BB position created
    if "bb_position" in decoupled_features:
        print("  ✓ Bollinger Band position feature created")
        checks.append(True)
    else:
        print("  ✗ Bollinger Band position feature missing")
        checks.append(False)

    print()

    # Test 4: Compare average correlations
    print("--- Test 4: Compare correlation statistics ---")
    avg_corr_baseline = np.mean(list(baseline_corrs.values())) if baseline_corrs else 0
    avg_corr_decoupled = np.mean(list(decoupled_corrs.values())) if decoupled_corrs else 0

    print(f"  Baseline avg abs correlation: {avg_corr_baseline:.4f}")
    print(f"  Decoupled avg abs correlation: {avg_corr_decoupled:.4f}")
    print(f"  Reduction: {((avg_corr_baseline - avg_corr_decoupled) / avg_corr_baseline * 100):.1f}%")
    print()

    # Success criteria
    success = True
    print("="*60)
    print("TEST RESULTS")
    print("="*60)

    # Criterion 1: Decoupling reduces high-correlation features
    if len(high_corr_decoupled) < len(high_corr_baseline):
        print(f"✓ PASS: High correlation features reduced from {len(high_corr_baseline)} to {len(high_corr_decoupled)}")
    else:
        print(f"✗ FAIL: High correlation features not reduced ({len(high_corr_baseline)} → {len(high_corr_decoupled)})")
        success = False

    # Criterion 2: Average correlation reduced
    if avg_corr_decoupled < avg_corr_baseline:
        print(f"✓ PASS: Average correlation reduced ({avg_corr_baseline:.4f} → {avg_corr_decoupled:.4f})")
    else:
        print(f"✗ FAIL: Average correlation not reduced ({avg_corr_baseline:.4f} → {avg_corr_decoupled:.4f})")
        success = False

    # Criterion 3: Feature transformations applied
    if all(checks):
        print(f"✓ PASS: All feature transformations applied correctly ({sum(checks)}/{len(checks)})")
    else:
        print(f"✗ FAIL: Some feature transformations missing ({sum(checks)}/{len(checks)})")
        success = False

    print()
    if success:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now use --decouple-features flag in prepare_dataset.py")
        print("Example:")
        print("  python prepare_dataset.py --sqlite data.db --symbol BTCUSDT --timeframe 30m --decouple-features")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the implementation of decouple_from_close logic.")
        return False


if __name__ == "__main__":
    success = test_decouple_features()
    sys.exit(0 if success else 1)
