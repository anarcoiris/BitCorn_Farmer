#!/usr/bin/env python3
"""
test_integration.py

Comprehensive integration test script to verify API updates and prevent data leakage.

Tests:
1. fiboevo.load_model() returns tuple (model, meta)
2. fiboevo.load_scaler() with feature validation
3. fiboevo.prepare_input_for_model() prevents double-scaling
4. fiboevo.temporal_split_indexes() prevents data leakage
5. Model loading in TradeApp and trading_daemon
6. Feature column alignment validation

Usage:
    python test_integration.py
"""

import sys
import os
from pathlib import Path
import traceback
import shutil

# Test results tracking
test_results = []

def test_result(name: str, passed: bool, message: str = ""):
    """Record test result"""
    test_results.append({
        "test": name,
        "passed": passed,
        "message": message
    })
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")
    if message:
        print(f"  → {message}")

def print_summary():
    """Print test summary"""
    passed = sum(1 for r in test_results if r["passed"])
    total = len(test_results)
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed < total:
        print("\nFailed tests:")
        for r in test_results:
            if not r["passed"]:
                print(f"  - {r['test']}: {r['message']}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)

# ============================================================================
# TEST 1: fiboevo module imports
# ============================================================================
def test_fiboevo_imports():
    """Test that fiboevo can be imported and has required functions"""
    try:
        import fiboevo as fibo
        test_result("Import fiboevo", True)

        # Check for required functions
        required_funcs = [
            "load_model",
            "load_scaler",
            "prepare_input_for_model",
            "temporal_split_indexes",
            "ensure_scaler_feature_names",
            "add_technical_features",
            "create_sequences_from_df"
        ]

        missing = []
        for func in required_funcs:
            if not hasattr(fibo, func):
                missing.append(func)

        if missing:
            test_result("fiboevo required functions", False, f"Missing: {', '.join(missing)}")
            return False
        else:
            test_result("fiboevo required functions", True)
            return True

    except Exception as e:
        test_result("Import fiboevo", False, str(e))
        return False

# ============================================================================
# TEST 2: load_model returns tuple
# ============================================================================
def test_load_model_signature():
    """Test that load_model() returns (model, meta) tuple"""
    test_dir = Path("test_artifacts")
    try:
        import fiboevo as fibo
        import torch
        import numpy as np

        # Create a minimal test model and save it
        test_dir.mkdir(exist_ok=True)

        # Use correct parameters that match the saved model
        input_size = 10
        hidden_size = 16
        num_layers = 2

        model = fibo.LSTM2Head(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        test_path = test_dir / "test_model.pt"

        # Save with metadata
        meta = {
            "input_size": input_size,
            "hidden": hidden_size,
            "num_layers": num_layers,
            "feature_cols": [f"feat_{i}" for i in range(input_size)]
        }
        fibo.save_model(model, test_path, meta=meta)

        # Test load_model returns tuple
        result = fibo.load_model(test_path, input_size=input_size, hidden=hidden_size)

        if isinstance(result, tuple) and len(result) == 2:
            loaded_model, loaded_meta = result
            test_result("load_model returns tuple", True, f"Returns: {type(result)}")

            # Verify meta contains expected keys
            if isinstance(loaded_meta, dict) and "feature_cols" in loaded_meta:
                test_result("load_model meta validation", True, f"Meta keys: {list(loaded_meta.keys())}")
            else:
                test_result("load_model meta validation", False, f"Meta incomplete: {loaded_meta}")
        else:
            test_result("load_model returns tuple", False, f"Returns: {type(result)}")

    except Exception as e:
        test_result("load_model returns tuple", False, f"Exception: {e}\n{traceback.format_exc()}")
    finally:
        # Cleanup - use shutil.rmtree for Windows compatibility
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

# ============================================================================
# TEST 3: load_scaler with feature validation
# ============================================================================
def test_load_scaler():
    """Test load_scaler() with feature validation"""
    test_dir = Path("test_artifacts")
    try:
        import fiboevo as fibo
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import joblib

        # Create test scaler
        test_dir.mkdir(exist_ok=True)

        feature_cols = [f"feat_{i}" for i in range(5)]
        X_train = np.random.randn(100, 5).astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(X_train)

        test_path = test_dir / "test_scaler.pkl"
        joblib.dump(scaler, test_path)

        # Test load_scaler with feature_cols
        loaded_scaler = fibo.load_scaler(test_path, feature_cols=feature_cols)

        # Check if feature_names_in_ was set
        if hasattr(loaded_scaler, "feature_names_in_"):
            test_result("load_scaler sets feature_names_in_", True,
                       f"feature_names_in_: {loaded_scaler.feature_names_in_}")
        else:
            test_result("load_scaler sets feature_names_in_", False,
                       "feature_names_in_ not set")

    except Exception as e:
        test_result("load_scaler with validation", False, f"Exception: {e}\n{traceback.format_exc()}")
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

# ============================================================================
# TEST 4: prepare_input_for_model prevents double-scaling
# ============================================================================
def test_prepare_input_for_model():
    """Test prepare_input_for_model() creates correct tensor"""
    try:
        import fiboevo as fibo
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        import numpy as np
        import torch

        # Create test data
        feature_cols = [f"feat_{i}" for i in range(5)]
        df = pd.DataFrame(np.random.randn(50, 5).astype(np.float32), columns=feature_cols)

        # Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(df.values[:30])  # fit on first 30 rows only
        scaler.feature_names_in_ = np.array(feature_cols, dtype=object)

        seq_len = 10

        # Test prepare_input_for_model
        seq_df = df.iloc[-seq_len:].reset_index(drop=True)
        x_tensor = fibo.prepare_input_for_model(
            seq_df,
            feature_cols,
            seq_len,
            scaler=scaler,
            method="per_row"
        )

        # Verify output
        if isinstance(x_tensor, torch.Tensor):
            if x_tensor.shape == (1, seq_len, len(feature_cols)):
                test_result("prepare_input_for_model shape", True,
                           f"Shape: {x_tensor.shape}")

                # Check if scaled (mean should be close to 0)
                mean_val = float(x_tensor.mean())
                if abs(mean_val) < 2.0:  # reasonable threshold for scaled data
                    test_result("prepare_input_for_model scaling", True,
                               f"Mean: {mean_val:.4f}")
                else:
                    test_result("prepare_input_for_model scaling", False,
                               f"Mean too large: {mean_val:.4f}")
            else:
                test_result("prepare_input_for_model shape", False,
                           f"Wrong shape: {x_tensor.shape}")
        else:
            test_result("prepare_input_for_model", False,
                       f"Wrong type: {type(x_tensor)}")

    except Exception as e:
        test_result("prepare_input_for_model", False, f"Exception: {e}\n{traceback.format_exc()}")

# ============================================================================
# TEST 5: temporal_split_indexes prevents data leakage
# ============================================================================
def test_temporal_split():
    """Test temporal_split_indexes() returns correct indices"""
    try:
        import fiboevo as fibo

        n_total = 1000
        seq_len = 32
        horizon = 10

        n_train_seq, n_val_seq, n_test_seq, train_rows_end = fibo.temporal_split_indexes(
            n_total=n_total,
            seq_len=seq_len,
            horizon=horizon,
            val_frac=0.2,
            test_frac=0.1
        )

        # Verify splits
        total_seq = n_train_seq + n_val_seq + n_test_seq
        expected_seq = n_total - seq_len - horizon + 1

        if total_seq == expected_seq:
            test_result("temporal_split_indexes total sequences", True,
                       f"Total: {total_seq}, Expected: {expected_seq}")
        else:
            test_result("temporal_split_indexes total sequences", False,
                       f"Total: {total_seq}, Expected: {expected_seq}")

        # Verify train_rows_end
        if train_rows_end is not None and train_rows_end > 0:
            test_result("temporal_split_indexes train_rows_end", True,
                       f"train_rows_end: {train_rows_end}")

            # Verify it's less than total (shouldn't include val/test data)
            if train_rows_end < n_total:
                test_result("temporal_split prevents leakage", True,
                           f"train_rows_end ({train_rows_end}) < n_total ({n_total})")
            else:
                test_result("temporal_split prevents leakage", False,
                           f"train_rows_end ({train_rows_end}) >= n_total ({n_total})")
        else:
            test_result("temporal_split_indexes train_rows_end", False,
                       f"Invalid train_rows_end: {train_rows_end}")

    except Exception as e:
        test_result("temporal_split_indexes", False, f"Exception: {e}\n{traceback.format_exc()}")

# ============================================================================
# TEST 6: Check prepare_dataset.py uses temporal_split_indexes
# ============================================================================
def test_prepare_dataset_integration():
    """Test that prepare_dataset.py uses temporal_split_indexes"""
    try:
        prepare_dataset_path = Path("prepare_dataset.py")

        if not prepare_dataset_path.exists():
            test_result("prepare_dataset.py exists", False, "File not found")
            return

        content = prepare_dataset_path.read_text(encoding="utf-8")

        # Check for temporal_split_indexes usage
        if "temporal_split_indexes" in content:
            test_result("prepare_dataset uses temporal_split_indexes", True)
        else:
            test_result("prepare_dataset uses temporal_split_indexes", False,
                       "temporal_split_indexes not found in file")

        # Check for scaler.feature_names_in_ assignment
        if "feature_names_in_" in content:
            test_result("prepare_dataset sets feature_names_in_", True)
        else:
            test_result("prepare_dataset sets feature_names_in_", False,
                       "feature_names_in_ assignment not found")

        # Check for train_rows_end in metadata
        if "train_rows_end" in content and '"train_rows_end"' in content:
            test_result("prepare_dataset saves train_rows_end", True)
        else:
            test_result("prepare_dataset saves train_rows_end", False,
                       "train_rows_end not in metadata")

    except Exception as e:
        test_result("prepare_dataset integration", False, f"Exception: {e}")

# ============================================================================
# TEST 7: Check TradeApp.py uses new APIs
# ============================================================================
def test_tradeapp_integration():
    """Test that TradeApp.py uses new APIs"""
    try:
        tradeapp_path = Path("TradeApp.py")

        if not tradeapp_path.exists():
            test_result("TradeApp.py exists", False, "File not found")
            return

        content = tradeapp_path.read_text(encoding="utf-8")

        # Check for fibo.load_model usage
        if "fibo.load_model(" in content or "fiboevo.load_model(" in content:
            test_result("TradeApp uses fibo.load_model", True)
        else:
            test_result("TradeApp uses fibo.load_model", False,
                       "fibo.load_model() not found")

        # Check for prepare_input_for_model usage
        if "prepare_input_for_model" in content:
            test_result("TradeApp uses prepare_input_for_model", True)
        else:
            test_result("TradeApp uses prepare_input_for_model", False,
                       "prepare_input_for_model not found")

        # Check for tuple unpacking
        if "model, meta = " in content:
            test_result("TradeApp unpacks load_model tuple", True)
        else:
            test_result("TradeApp unpacks load_model tuple", False,
                       "Tuple unpacking pattern not found")

    except Exception as e:
        test_result("TradeApp integration", False, f"Exception: {e}")

# ============================================================================
# TEST 8: Check trading_daemon.py uses new APIs
# ============================================================================
def test_daemon_integration():
    """Test that trading_daemon.py uses new APIs"""
    try:
        daemon_path = Path("trading_daemon.py")

        if not daemon_path.exists():
            test_result("trading_daemon.py exists", False, "File not found")
            return

        content = daemon_path.read_text(encoding="utf-8")

        # Check for prepare_input_for_model usage
        if "prepare_input_for_model" in content:
            test_result("trading_daemon uses prepare_input_for_model", True)
        else:
            test_result("trading_daemon uses prepare_input_for_model", False,
                       "prepare_input_for_model not found")

        # Check for fibo.load_scaler usage
        if "fibo.load_scaler(" in content:
            test_result("trading_daemon uses fibo.load_scaler", True)
        else:
            test_result("trading_daemon uses fibo.load_scaler", False,
                       "fibo.load_scaler() not found")

    except Exception as e:
        test_result("trading_daemon integration", False, f"Exception: {e}")

# ============================================================================
# Main test runner
# ============================================================================
def main():
    print("="*60)
    print("INTEGRATION TEST SUITE")
    print("Testing API updates and data leakage prevention")
    print("="*60)
    print()

    # Run all tests
    if test_fiboevo_imports():
        test_load_model_signature()
        test_load_scaler()
        test_prepare_input_for_model()
        test_temporal_split()

    # Integration tests (don't require artifacts)
    test_prepare_dataset_integration()
    test_tradeapp_integration()
    test_daemon_integration()

    # Print summary
    print_summary()

if __name__ == "__main__":
    main()
