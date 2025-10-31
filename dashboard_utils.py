#!/usr/bin/env python3
"""
dashboard_utils.py

Utility functions for the prediction dashboard.

Provides:
- Data fetching from SQLite with feature engineering
- Prediction caching for performance
- Asynchronous prediction runner for non-blocking GUI
- Thread-safe queue management

Author: Claude (Anthropic)
Date: 2025-10-30
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_latest_data(
    sqlite_path: str,
    table: str = "ohlcv",
    min_rows: int = 1000,
    add_features: bool = True
) -> pd.DataFrame:
    """
    Fetch latest data from SQLite database with optional feature engineering.

    Args:
        sqlite_path: Path to SQLite database
        table: Table name (default: ohlcv)
        min_rows: Minimum number of rows to fetch
        add_features: Whether to compute technical features

    Returns:
        DataFrame with OHLCV data and features (if requested)

    Raises:
        FileNotFoundError: If database doesn't exist
        ValueError: If table is empty or missing required columns
    """
    db_path = Path(sqlite_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {sqlite_path}")

    try:
        conn = sqlite3.connect(sqlite_path)

        # Fetch latest rows
        query = f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT {min_rows}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) == 0:
            raise ValueError(f"Table {table} is empty")

        # Reverse to chronological order
        df = df.iloc[::-1].reset_index(drop=True)

        logger.info(f"Fetched {len(df)} rows from {sqlite_path}")

        # Validate required columns
        required_cols = ['close', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add technical features if requested
        if add_features:
            try:
                from fiboevo import add_technical_features
                df = add_technical_features(df)
                logger.info(f"Added technical features, now have {len(df.columns)} columns")
            except ImportError:
                logger.warning("fiboevo not available, skipping feature engineering")
            except Exception as e:
                logger.error(f"Feature engineering failed: {e}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch data from {sqlite_path}: {e}")
        raise


def check_data_freshness(
    sqlite_path: str,
    table: str = "ohlcv",
    max_age_minutes: int = 60
) -> Tuple[bool, Optional[datetime]]:
    """
    Check if data in database is fresh (recently updated).

    Args:
        sqlite_path: Path to SQLite database
        table: Table name
        max_age_minutes: Maximum acceptable age in minutes

    Returns:
        Tuple of (is_fresh, last_timestamp)
        - is_fresh: True if data is fresh, False otherwise
        - last_timestamp: Timestamp of most recent data point
    """
    try:
        conn = sqlite3.connect(sqlite_path)
        query = f"SELECT MAX(timestamp) FROM {table}"
        cursor = conn.execute(query)
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            # Parse timestamp (handle different formats)
            last_ts_str = result[0]

            try:
                # Try ISO format
                last_ts = pd.to_datetime(last_ts_str)
            except:
                # Try Unix timestamp
                try:
                    last_ts = pd.to_datetime(float(last_ts_str), unit='s')
                except:
                    logger.warning(f"Could not parse timestamp: {last_ts_str}")
                    return False, None

            # Check freshness
            age = datetime.now() - last_ts.to_pydatetime()
            is_fresh = age < timedelta(minutes=max_age_minutes)

            logger.info(f"Data age: {age}, fresh: {is_fresh}")
            return is_fresh, last_ts.to_pydatetime()

        return False, None

    except Exception as e:
        logger.error(f"Failed to check data freshness: {e}")
        return False, None


class PredictionCache:
    """
    Simple in-memory cache for predictions.

    Caches predictions based on a key tuple of (horizons, data_length, last_price).
    This avoids recomputing predictions when the underlying data hasn't changed.

    Attributes:
        max_age_seconds: Maximum age for cached entries (default: 300s = 5min)
        max_size: Maximum number of cache entries (default: 10)
    """

    def __init__(self, max_age_seconds: int = 300, max_size: int = 10):
        """Initialize prediction cache."""
        self.cache: Dict[Any, Tuple[float, Any]] = {}
        self.max_age_seconds = max_age_seconds
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value from cache if it exists and is not expired.

        Args:
            key: Cache key (should be hashable)

        Returns:
            Cached value if available and fresh, None otherwise
        """
        with self.lock:
            if key not in self.cache:
                return None

            timestamp, value = self.cache[key]

            # Check if expired
            age = time.time() - timestamp

            if age > self.max_age_seconds:
                logger.info(f"Cache entry expired (age: {age:.1f}s)")
                del self.cache[key]
                return None

            logger.info(f"Cache hit (age: {age:.1f}s)")
            return value

    def set(self, key: Any, value: Any):
        """
        Store value in cache with current timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Enforce size limit
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.items(), key=lambda x: x[1][0])[0]
                del self.cache[oldest_key]
                logger.info(f"Cache full, removed oldest entry")

            self.cache[key] = (time.time(), value)
            logger.info(f"Cached new entry (total: {len(self.cache)})")

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "max_age_seconds": self.max_age_seconds
            }


class AsyncPredictionRunner:
    """
    Asynchronous prediction runner for non-blocking GUI updates.

    Runs predictions in a background thread and provides results via callback.

    Usage:
        runner = AsyncPredictionRunner(callback=update_gui)
        runner.run_prediction(df, model, meta, scaler, device, horizons)
    """

    def __init__(self, callback: Optional[callable] = None):
        """
        Initialize async prediction runner.

        Args:
            callback: Function to call with results (success, data)
        """
        self.callback = callback
        self.thread: Optional[threading.Thread] = None
        self.is_running = False

    def run_prediction(
        self,
        df: pd.DataFrame,
        model: Any,
        meta: Dict[str, Any],
        scaler: Any,
        device: Any,
        horizons: list,
        **kwargs
    ):
        """
        Start async prediction generation.

        Args:
            df: Historical data
            model: LSTM model
            meta: Model metadata
            scaler: StandardScaler
            device: torch device
            horizons: List of horizons to predict
            **kwargs: Additional arguments for predict_multiple_horizons
        """
        if self.is_running:
            logger.warning("Prediction already running, ignoring new request")
            return

        self.is_running = True

        self.thread = threading.Thread(
            target=self._run_prediction_thread,
            args=(df, model, meta, scaler, device, horizons),
            kwargs=kwargs,
            daemon=True
        )
        self.thread.start()

    def _run_prediction_thread(
        self,
        df: pd.DataFrame,
        model: Any,
        meta: Dict[str, Any],
        scaler: Any,
        device: Any,
        horizons: list,
        **kwargs
    ):
        """Internal prediction thread function."""
        try:
            from multi_horizon_fan_inference import predict_multiple_horizons

            logger.info("Starting async prediction generation...")

            predictions = predict_multiple_horizons(
                df=df,
                model=model,
                meta=meta,
                scaler=scaler,
                device=device,
                horizons=horizons,
                **kwargs
            )

            logger.info("Async prediction completed successfully")

            if self.callback is not None:
                self.callback("success", predictions)

        except Exception as e:
            logger.error(f"Async prediction failed: {e}", exc_info=True)

            if self.callback is not None:
                self.callback("error", str(e))

        finally:
            self.is_running = False

    def is_busy(self) -> bool:
        """Check if prediction is currently running."""
        return self.is_running


def validate_data_for_prediction(
    df: pd.DataFrame,
    meta: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate that DataFrame has all required features for model inference.

    Args:
        df: DataFrame to validate
        meta: Model metadata with feature_cols

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if data is valid, False otherwise
        - error_message: Description of validation error (None if valid)
    """
    try:
        required_features = meta.get("feature_cols", [])
        seq_len = meta.get("seq_len", 32)

        # Check minimum length
        if len(df) < seq_len + 10:
            return False, f"DataFrame too short: need at least {seq_len + 10} rows, got {len(df)}"

        # Check for required columns
        missing_features = [f for f in required_features if f not in df.columns]

        if missing_features:
            return False, f"Missing features: {missing_features[:5]}..."  # Show first 5

        # Check for NaN values in recent data
        recent_data = df.iloc[-seq_len:]
        nan_cols = recent_data[required_features].isna().sum()
        nan_cols = nan_cols[nan_cols > 0]

        if len(nan_cols) > 0:
            return False, f"NaN values in recent data: {dict(nan_cols)}"

        # Check for timestamp
        if "timestamp" not in df.columns:
            logger.warning("No timestamp column, predictions will use indices")

        return True, None

    except Exception as e:
        return False, f"Validation error: {e}"


def estimate_prediction_time(
    n_predictions: int,
    n_horizons: int,
    device_type: str = "cpu"
) -> float:
    """
    Estimate time required to generate predictions.

    Args:
        n_predictions: Number of predictions per horizon
        n_horizons: Number of horizons
        device_type: Device type ("cpu" or "cuda")

    Returns:
        Estimated time in seconds
    """
    # Rough estimates based on typical LSTM inference speed
    # These are conservative estimates for planning purposes

    base_time_per_pred = 0.005 if device_type == "cuda" else 0.02  # seconds

    total_predictions = n_predictions * n_horizons
    estimated_time = total_predictions * base_time_per_pred

    # Add overhead for data preparation
    overhead = 0.5

    return estimated_time + overhead


def format_prediction_summary(
    predictions_by_horizon: Dict[int, pd.DataFrame],
    current_price: float
) -> str:
    """
    Format prediction summary as human-readable text.

    Args:
        predictions_by_horizon: Dict mapping horizon -> predictions DataFrame
        current_price: Current price

    Returns:
        Formatted summary string
    """
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("PREDICTION SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Current Price: ${current_price:,.2f}")
    summary_lines.append("")

    for h in sorted(predictions_by_horizon.keys()):
        pred_df = predictions_by_horizon[h]

        if len(pred_df) == 0:
            continue

        # Get last prediction
        last_pred = pred_df.iloc[-1]
        pred_price = last_pred["close_pred"]

        change_dollars = pred_price - current_price
        change_pct = (change_dollars / current_price) * 100

        # Confidence interval
        lower = last_pred.get("lower_bound", pred_price)
        upper = last_pred.get("upper_bound", pred_price)

        summary_lines.append(f"Horizon: {h}h")
        summary_lines.append(f"  Predicted Price: ${pred_price:,.2f}")
        summary_lines.append(f"  Change: ${change_dollars:+,.2f} ({change_pct:+.2f}%)")
        summary_lines.append(f"  95% CI: ${lower:,.2f} - ${upper:,.2f}")
        summary_lines.append("")

    summary_lines.append("=" * 60)

    return "\n".join(summary_lines)


def compute_horizon_statistics(
    predictions_by_horizon: Dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute summary statistics for each horizon.

    Args:
        predictions_by_horizon: Dict mapping horizon -> predictions DataFrame

    Returns:
        DataFrame with statistics for each horizon
    """
    stats_data = []

    for h in sorted(predictions_by_horizon.keys()):
        pred_df = predictions_by_horizon[h]

        if len(pred_df) == 0:
            continue

        valid = pred_df.dropna(subset=["prediction_error"]) if "prediction_error" in pred_df.columns else pred_df

        stats = {
            "horizon": h,
            "n_predictions": len(pred_df),
            "mean_predicted_price": pred_df["close_pred"].mean(),
            "std_predicted_price": pred_df["close_pred"].std(),
            "mean_volatility": pred_df["volatility_pred"].mean(),
            "min_prediction": pred_df["close_pred"].min(),
            "max_prediction": pred_df["close_pred"].max()
        }

        if len(valid) > 0 and "prediction_error" in valid.columns:
            stats.update({
                "mae": valid["prediction_error"].abs().mean(),
                "rmse": np.sqrt((valid["prediction_error"] ** 2).mean()),
                "mape": valid["prediction_error_pct"].abs().mean() if "prediction_error_pct" in valid.columns else np.nan
            })

        stats_data.append(stats)

    return pd.DataFrame(stats_data)


# ==========================================
# Data Quality Checks
# ==========================================

def check_for_gaps(df: pd.DataFrame, max_gap_minutes: int = 120) -> Tuple[bool, list]:
    """
    Check for gaps in timestamp sequence.

    Args:
        df: DataFrame with timestamp column
        max_gap_minutes: Maximum acceptable gap in minutes

    Returns:
        Tuple of (has_gaps, gap_indices)
        - has_gaps: True if gaps found
        - gap_indices: List of indices where gaps occur
    """
    if "timestamp" not in df.columns:
        logger.warning("No timestamp column, cannot check for gaps")
        return False, []

    try:
        timestamps = pd.to_datetime(df["timestamp"])
        time_diffs = timestamps.diff()

        # Convert to minutes
        time_diffs_minutes = time_diffs.dt.total_seconds() / 60

        # Find gaps
        gap_mask = time_diffs_minutes > max_gap_minutes
        gap_indices = gap_mask[gap_mask].index.tolist()

        if len(gap_indices) > 0:
            logger.warning(f"Found {len(gap_indices)} gaps in data (>{max_gap_minutes} min)")
            return True, gap_indices

        return False, []

    except Exception as e:
        logger.error(f"Gap check failed: {e}")
        return False, []


def check_for_stale_features(df: pd.DataFrame, feature_cols: list) -> Tuple[bool, list]:
    """
    Check for features with constant values (potential staleness indicator).

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names

    Returns:
        Tuple of (has_stale, stale_features)
        - has_stale: True if stale features found
        - stale_features: List of feature names with constant values
    """
    stale_features = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Check last 10 rows
        recent = df[col].iloc[-10:]

        if recent.nunique() == 1:
            stale_features.append(col)
            logger.warning(f"Feature {col} has constant value in recent data")

    return len(stale_features) > 0, stale_features


# ==========================================
# Logging Utilities
# ==========================================

class PredictionLogger:
    """
    Logger for prediction events with structured output.

    Logs predictions to file with timestamps for audit trail.
    """

    def __init__(self, log_dir: str = "logs/predictions"):
        """Initialize prediction logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create daily log file
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"predictions_{today}.log"

    def log_prediction_event(
        self,
        event_type: str,
        horizons: list,
        current_price: float,
        predictions_summary: Optional[Dict[int, float]] = None,
        error: Optional[str] = None
    ):
        """
        Log a prediction event.

        Args:
            event_type: Event type ("success", "error", "cached")
            horizons: List of horizons predicted
            current_price: Current market price
            predictions_summary: Dict mapping horizon -> predicted price
            error: Error message if applicable
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "horizons": horizons,
            "current_price": current_price,
            "predictions": predictions_summary,
            "error": error
        }

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write prediction log: {e}")


# ==========================================
# Test Functions
# ==========================================

def test_fetch_data():
    """Test data fetching."""
    try:
        df = fetch_latest_data(
            "data_manager/exports/Binance_BTCUSDT_1h.db",
            table="ohlcv",
            min_rows=1000
        )
        print(f"Fetched {len(df)} rows with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Last close: ${df['close'].iloc[-1]:,.2f}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def test_cache():
    """Test prediction cache."""
    cache = PredictionCache(max_age_seconds=5)

    key1 = ("test", 1, 2)
    value1 = {"data": "example"}

    # Set and get
    cache.set(key1, value1)
    result = cache.get(key1)
    print(f"Cache get: {result}")

    # Wait for expiration
    import time
    print("Waiting for cache expiration...")
    time.sleep(6)

    result = cache.get(key1)
    print(f"After expiration: {result}")

    return True


if __name__ == "__main__":
    # Run tests
    print("Testing data fetch...")
    test_fetch_data()

    print("\nTesting cache...")
    test_cache()
