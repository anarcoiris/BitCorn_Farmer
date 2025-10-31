#!/usr/bin/env python3
"""
feature_registry.py

Central registry for feature engineering systems.
Allows dynamic switching between different feature sets (v1, v2, custom).

Author: Claude (Anthropic)
Date: 2025-10-31
"""

from typing import Callable, Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineeringRegistry:
    """
    Central registry for feature engineering systems.
    Allows dynamic switching between different feature sets.
    """

    def __init__(self):
        self._systems: Dict[str, Dict[str, Any]] = {}
        self._active_system: str = "v2"  # Default to clean features

    def register_system(
        self,
        name: str,
        function: Callable,
        description: str,
        expected_features: List[str],
        version: str = "1.0"
    ):
        """
        Register a feature engineering system.

        Args:
            name: Unique identifier (e.g., "v1", "v2", "custom_1")
            function: Feature engineering function
                     signature: (close, high, low, volume, dropna_after) -> DataFrame
            description: Human-readable description
            expected_features: List of feature column names this system produces
            version: Version string
        """
        self._systems[name] = {
            "function": function,
            "description": description,
            "expected_features": expected_features,
            "version": version
        }
        logger.debug(f"Registered feature system: {name} ({len(expected_features)} features)")

    def get_system(self, name: str) -> Callable:
        """Get feature engineering function by name."""
        if name not in self._systems:
            available = list(self._systems.keys())
            raise ValueError(
                f"System '{name}' not found. Available systems: {available}"
            )
        return self._systems[name]["function"]

    def get_system_info(self, name: str) -> Dict[str, Any]:
        """Get full information about a system."""
        if name not in self._systems:
            raise ValueError(f"System '{name}' not found")
        return self._systems[name]

    def get_active_system(self) -> str:
        """Get currently active system name."""
        return self._active_system

    def set_active_system(self, name: str):
        """Set active feature engineering system."""
        if name not in self._systems:
            raise ValueError(f"System '{name}' not found")
        self._active_system = name
        logger.info(f"Switched to feature system: {name}")

    def list_systems(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered systems with summary info.

        Returns:
            Dict with system names as keys, containing:
            - description: str
            - n_features: int
            - version: str
        """
        return {
            name: {
                "description": info["description"],
                "n_features": len(info["expected_features"]),
                "version": info["version"]
            }
            for name, info in self._systems.items()
        }

    def compute_features(
        self,
        close: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        system_name: Optional[str] = None,
        dropna_after: bool = True
    ) -> pd.DataFrame:
        """
        Compute features using specified (or active) system.

        Args:
            close: Close price array
            high: High price array (optional)
            low: Low price array (optional)
            volume: Volume array (optional)
            system_name: System to use (defaults to active)
            dropna_after: Drop NaN rows after computation

        Returns:
            DataFrame with computed features
        """
        system = system_name or self._active_system
        func = self.get_system(system)

        logger.debug(f"Computing features using system: {system}")

        # Call the feature engineering function
        df_features = func(
            close,
            high=high,
            low=low,
            volume=volume,
            dropna_after=dropna_after
        )

        return df_features


# Global registry instance
FEATURE_REGISTRY = FeatureEngineeringRegistry()


def register_builtin_systems():
    """
    Register built-in feature engineering systems (v1 and v2).
    Called automatically on import.
    """
    try:
        import fiboevo
    except ImportError:
        logger.warning("fiboevo module not found, cannot register built-in systems")
        return

    # Register v1 (39 features with price levels)
    if hasattr(fiboevo, 'add_technical_features'):
        FEATURE_REGISTRY.register_system(
            name="v1",
            function=fiboevo.add_technical_features,
            description="Original 39 features (Fibonacci, MA, RSI, ATR, TD Sequential) - includes price levels",
            expected_features=[
                'log_close', 'log_ret_1', 'log_ret_5', 'sma_5', 'sma_20', 'sma_50',
                'ema_5', 'ema_20', 'ema_50', 'ma_diff_20', 'bb_m', 'bb_std', 'bb_up',
                'bb_dn', 'bb_width', 'rsi_14', 'atr_14', 'raw_vol_10', 'raw_vol_30',
                'fib_r_236', 'dist_fib_r_236', 'fib_r_382', 'dist_fib_r_382', 'fib_r_500',
                'dist_fib_r_500', 'fib_r_618', 'dist_fib_r_618', 'fib_r_786', 'dist_fib_r_786',
                'fibext_1272', 'dist_fibext_1272', 'fibext_1618', 'dist_fibext_1618',
                'fibext_2000', 'dist_fibext_2000', 'td_buy_setup', 'td_sell_setup', 'ret_1', 'ret_5'
            ],
            version="1.0"
        )
        logger.info("Registered feature system: v1 (39 features)")

    # Register v2 (14 clean features, no price-level leakage)
    if hasattr(fiboevo, 'add_technical_features_v2'):
        FEATURE_REGISTRY.register_system(
            name="v2",
            function=fiboevo.add_technical_features_v2,
            description="Clean 14 features, no price-level leakage (log returns, ratios, indicators)",
            expected_features=[
                'log_ret_1', 'log_ret_5', 'sma_ratio_5', 'sma_ratio_20', 'sma_ratio_50',
                'ema_ratio_5', 'ema_ratio_20', 'ema_ratio_50', 'bb_position', 'bb_width',
                'rsi_14', 'atr_pct', 'raw_vol_10', 'raw_vol_30'
                # Note: v2 may or may not include td_buy_setup, td_sell_setup depending on config
            ],
            version="2.0"
        )
        logger.info("Registered feature system: v2 (14 features, clean)")


# Auto-register built-in systems on import
register_builtin_systems()


def validate_feature_compatibility(
    model_features: List[str],
    system_features: List[str]
) -> tuple[bool, List[str]]:
    """
    Check if feature system matches model expectations.

    Args:
        model_features: Features required by model (from meta.json)
        system_features: Features provided by system

    Returns:
        Tuple of (is_compatible, missing_features)
        - is_compatible: True if all required features are present
        - missing_features: List of features required but not provided
    """
    missing = set(model_features) - set(system_features)

    if missing:
        logger.warning(
            f"Feature system missing required features: {sorted(missing)}\n"
            f"Model requires: {len(model_features)} features\n"
            f"System provides: {len(system_features)} features"
        )
        return False, sorted(missing)

    return True, []


def detect_system_from_meta(meta: Dict[str, Any]) -> str:
    """
    Automatically detect which feature system to use based on meta.json.

    Args:
        meta: Model metadata dictionary (with "feature_cols" key)

    Returns:
        System name ("v1", "v2", or None if cannot detect)
    """
    if "feature_cols" not in meta:
        logger.warning("meta.json does not contain 'feature_cols'")
        return None

    feature_cols = meta["feature_cols"]
    n_features = len(feature_cols)

    # Heuristics to detect system version
    if n_features >= 35:  # v1 has 39 features
        # Check for v1-specific features
        v1_markers = ["fib_r_236", "fibext_1272", "bb_m", "log_close"]
        if any(marker in feature_cols for marker in v1_markers):
            logger.info(f"Detected feature system: v1 ({n_features} features)")
            return "v1"

    elif 10 <= n_features <= 20:  # v2 has ~14 features
        # Check for v2-specific features
        v2_markers = ["sma_ratio_5", "ema_ratio_5", "bb_position", "atr_pct"]
        if any(marker in feature_cols for marker in v2_markers):
            logger.info(f"Detected feature system: v2 ({n_features} features)")
            return "v2"

    logger.warning(f"Could not auto-detect feature system ({n_features} features)")
    return None
