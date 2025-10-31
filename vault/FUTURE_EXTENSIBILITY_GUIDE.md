# Future Extensibility Guide
## BitCorn Farmer - Multi-Horizon Prediction System

**Date:** 2025-10-31
**Author:** Claude (Anthropic)
**Purpose:** Architecture roadmap for planned enhancements

---

## Table of Contents

1. [Current System Overview](#current-system-overview)
2. [Feature Engineering System Selector](#feature-engineering-system-selector)
3. [GUI-Based Feature Editor](#gui-based-feature-editor)
4. [Val/Train Split Configuration](#valtrain-split-configuration)
5. [Rolling Window Configuration](#rolling-window-configuration)
6. [Multiple Prediction Fans](#multiple-prediction-fans)
7. [Implementation Priorities](#implementation-priorities)

---

## Current System Overview

### Existing Components

**Feature Engineering:**
- `add_technical_features()` (v1) - 39 features with price levels (current model uses this)
- `add_technical_features_v2()` (v2) - 15-18 clean features, no price leakage

**Model Architecture:**
- LSTM2Head dual-output (log_return + volatility)
- Single horizon prediction with scaling to multi-horizon

**GUI Integration:**
- Status tab with live multi-horizon prediction fan
- WebSocket data ingestion
- TradingDaemon with 5-second inference loop

---

## 1. Feature Engineering System Selector

### Objective
Allow users to toggle between different feature engineering systems (v1, v2, future versions) through GUI.

### Architecture Design

#### A. Backend: Feature Engineering Registry

Create `feature_engineering_registry.py`:

```python
# feature_engineering_registry.py

from typing import Callable, Dict, List, Any
import pandas as pd
import numpy as np

class FeatureEngineeringRegistry:
    """
    Central registry for feature engineering systems.
    Allows dynamic switching between different feature sets.
    """

    def __init__(self):
        self._systems: Dict[str, Dict[str, Any]] = {}
        self._active_system: str = "v1"

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
            function: Feature engineering function (signature: close, high, low, volume -> DataFrame)
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

    def get_system(self, name: str) -> Callable:
        """Get feature engineering function by name."""
        if name not in self._systems:
            raise ValueError(f"System '{name}' not found. Available: {list(self._systems.keys())}")
        return self._systems[name]["function"]

    def get_active_system(self) -> str:
        """Get currently active system name."""
        return self._active_system

    def set_active_system(self, name: str):
        """Set active feature engineering system."""
        if name not in self._systems:
            raise ValueError(f"System '{name}' not found")
        self._active_system = name

    def list_systems(self) -> Dict[str, Dict[str, Any]]:
        """List all registered systems."""
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
        high: np.ndarray = None,
        low: np.ndarray = None,
        volume: np.ndarray = None,
        system_name: str = None
    ) -> pd.DataFrame:
        """
        Compute features using specified (or active) system.

        Args:
            close, high, low, volume: Price/volume arrays
            system_name: System to use (defaults to active)

        Returns:
            DataFrame with computed features
        """
        system = system_name or self._active_system
        func = self.get_system(system)
        return func(close, high=high, low=low, volume=volume, dropna_after=True)


# Global registry instance
FEATURE_REGISTRY = FeatureEngineeringRegistry()


def register_builtin_systems():
    """Register built-in feature engineering systems."""
    import fiboevo

    # Register v1 (39 features)
    FEATURE_REGISTRY.register_system(
        name="v1",
        function=fiboevo.add_technical_features,
        description="Original 39 features with Fibonacci, MA, RSI, ATR, TD Sequential",
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

    # Register v2 (15-18 clean features)
    FEATURE_REGISTRY.register_system(
        name="v2",
        function=fiboevo.add_technical_features_v2,
        description="Clean 15-18 features, no price-level leakage",
        expected_features=[
            'log_ret_1', 'log_ret_5', 'sma_ratio_5', 'sma_ratio_20', 'sma_ratio_50',
            'ema_ratio_5', 'ema_ratio_20', 'ema_ratio_50', 'bb_position', 'bb_width',
            'rsi_14', 'atr_pct', 'raw_vol_10', 'raw_vol_30', 'ret_1', 'ret_5',
            'td_buy_setup', 'td_sell_setup'  # Optional
        ],
        version="2.0"
    )

# Auto-register on import
register_builtin_systems()
```

#### B. GUI Integration

**Add to TradeApp.py:**

```python
# In __init__:
self.feature_system_var = StringVar(value="v1")  # Active feature system
self.feature_system_options = ["v1", "v2"]  # Will be populated dynamically

# In _build_config_tab() or new Settings tab:
Label(frm, text="Feature System:").grid(row=X, column=0, sticky=W)
OptionMenu(frm, self.feature_system_var, *self.feature_system_options).grid(row=X, column=1)
Button(frm, text="Reload Feature Systems", command=self._reload_feature_systems).grid(row=X, column=2)

def _reload_feature_systems(self):
    from feature_engineering_registry import FEATURE_REGISTRY
    systems = FEATURE_REGISTRY.list_systems()
    self.feature_system_options = list(systems.keys())
    # Update OptionMenu widget
    # ... (recreate menu or use ttk.Combobox for dynamic options)
```

**Modify TradingDaemon.iteration_once() and iteration_once_multi_horizon():**

```python
# Replace:
df_feats = fibo.add_technical_features(close, high=high, low=low, volume=vol, dropna_after=True)

# With:
from feature_engineering_registry import FEATURE_REGISTRY
active_system = self.feature_system  # New attribute set by GUI
df_feats = FEATURE_REGISTRY.compute_features(close, high=high, low=low, volume=vol, system_name=active_system)
```

#### C. Model Compatibility Check

**Add validation:**

```python
def validate_feature_compatibility(model_features: List[str], system_features: List[str]) -> bool:
    """Check if feature system matches model expectations."""
    missing = set(model_features) - set(system_features)
    if missing:
        raise ValueError(f"Feature system missing required features: {missing}")
    return True
```

---

## 2. GUI-Based Feature Editor

### Objective
Allow users to create custom feature engineering pipelines through GUI.

### Architecture Design

#### A. Feature Builder Components

Create modular feature building blocks:

```python
# feature_builder.py

from typing import List, Dict, Any
import pandas as pd
import numpy as np

class FeatureBlock:
    """Base class for feature building blocks."""

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute feature from DataFrame. Override in subclasses."""
        raise NotImplementedError


class LogReturnBlock(FeatureBlock):
    """Log return feature."""

    def __init__(self, period: int = 1):
        super().__init__("log_return", {"period": period})

    def compute(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        return np.log(close / close.shift(self.params["period"]))


class SMABlock(FeatureBlock):
    """Simple moving average."""

    def __init__(self, window: int = 20, ratio: bool = False):
        super().__init__("sma", {"window": window, "ratio": ratio})

    def compute(self, df: pd.DataFrame) -> pd.Series:
        sma = df["close"].rolling(window=self.params["window"]).mean()
        if self.params["ratio"]:
            return df["close"] / sma
        return sma


class RSIBlock(FeatureBlock):
    """Relative strength index."""

    def __init__(self, period: int = 14):
        super().__init__("rsi", {"period": period})

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # RSI calculation
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.params["period"]).mean()
        avg_loss = loss.rolling(window=self.params["period"]).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# Registry of available blocks
FEATURE_BLOCKS = {
    "Log Return": LogReturnBlock,
    "SMA": SMABlock,
    "EMA": lambda period: ...,  # Implement
    "RSI": RSIBlock,
    "ATR": lambda period: ...,  # Implement
    "Bollinger Bands": lambda window, std: ...,  # Implement
}
```

#### B. GUI Feature Editor

**New window in TradeApp.py:**

```python
def _open_feature_editor(self):
    """Open feature engineering editor window."""
    top = Toplevel(self.root)
    top.title("Feature Engineering Editor")
    top.geometry("800x600")

    # Left: Available blocks
    frm_blocks = Frame(top, width=200)
    frm_blocks.pack(side=LEFT, fill=Y, padx=5, pady=5)
    Label(frm_blocks, text="Available Blocks", font=("Arial", 12, "bold")).pack()

    for block_name in FEATURE_BLOCKS.keys():
        Button(frm_blocks, text=block_name,
               command=lambda bn=block_name: self._add_feature_block(bn)).pack(fill=X, pady=2)

    # Center: Pipeline builder (drag-and-drop or list)
    frm_pipeline = Frame(top)
    frm_pipeline.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)
    Label(frm_pipeline, text="Feature Pipeline", font=("Arial", 12, "bold")).pack()

    self.feature_pipeline_tree = ttk.Treeview(frm_pipeline, columns=("Block", "Params"), height=20)
    self.feature_pipeline_tree.heading("Block", text="Feature Block")
    self.feature_pipeline_tree.heading("Params", text="Parameters")
    self.feature_pipeline_tree.pack(fill=BOTH, expand=True)

    # Right: Parameters editor
    frm_params = Frame(top, width=200)
    frm_params.pack(side=RIGHT, fill=Y, padx=5, pady=5)
    Label(frm_params, text="Block Parameters", font=("Arial", 12, "bold")).pack()

    # ... (parameter editing widgets)

    # Bottom: Actions
    frm_actions = Frame(top)
    frm_actions.pack(side=BOTTOM, fill=X, padx=5, pady=5)
    Button(frm_actions, text="Test Pipeline", command=self._test_feature_pipeline).pack(side=LEFT)
    Button(frm_actions, text="Save as Custom System", command=self._save_custom_feature_system).pack(side=LEFT)
    Button(frm_actions, text="Export to Code", command=self._export_feature_code).pack(side=LEFT)
```

---

## 3. Val/Train Split Configuration

### Objective
Allow users to configure train/validation split parameters through GUI.

### Architecture Design

#### A. Split Configuration Class

```python
# split_config.py

from dataclasses import dataclass
from typing import Literal

@dataclass
class SplitConfig:
    """Configuration for train/validation split."""

    method: Literal["random", "sequential", "rolling", "walk_forward"] = "sequential"
    val_fraction: float = 0.1
    test_fraction: float = 0.0
    shuffle: bool = False
    random_seed: int = 42

    # Rolling window parameters
    n_splits: int = 5
    train_window: int = None  # None = expanding window
    val_window: int = None

    # Walk-forward parameters
    retrain_frequency: int = 100  # Retrain every N samples

    def validate(self):
        """Validate configuration parameters."""
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val + test fractions must be < 1")


def create_split(
    data_length: int,
    config: SplitConfig
) -> Dict[str, np.ndarray]:
    """
    Create train/val/test split indices based on config.

    Returns:
        Dict with keys: "train_idx", "val_idx", "test_idx"
    """
    # Implementation for different split methods
    ...
```

#### B. GUI Integration

**Add to Train tab:**

```python
# In _build_train_tab():

frm_split = LabelFrame(tab, text="Train/Val Split Configuration")
frm_split.pack(fill=X, padx=6, pady=6)

# Split method
Label(frm_split, text="Split method:").grid(row=0, column=0, sticky=W)
self.split_method_var = StringVar(value="sequential")
OptionMenu(frm_split, self.split_method_var, "random", "sequential", "rolling", "walk_forward").grid(row=0, column=1)

# Val fraction
Label(frm_split, text="Validation fraction:").grid(row=1, column=0, sticky=W)
self.val_frac = DoubleVar(value=0.1)
Entry(frm_split, textvariable=self.val_frac, width=8).grid(row=1, column=1)

# Rolling window parameters (conditional visibility)
self.frm_rolling = Frame(frm_split)
self.frm_rolling.grid(row=2, column=0, columnspan=3, sticky=W)

Label(self.frm_rolling, text="N splits:").grid(row=0, column=0)
self.n_splits_var = IntVar(value=5)
Entry(self.frm_rolling, textvariable=self.n_splits_var, width=8).grid(row=0, column=1)

Label(self.frm_rolling, text="Train window:").grid(row=1, column=0)
self.train_window_var = IntVar(value=0)  # 0 = expanding
Entry(self.frm_rolling, textvariable=self.train_window_var, width=8).grid(row=1, column=1)

# Show/hide rolling params based on method
def _on_split_method_change(*args):
    if self.split_method_var.get() in ["rolling", "walk_forward"]:
        self.frm_rolling.grid()
    else:
        self.frm_rolling.grid_remove()

self.split_method_var.trace("w", _on_split_method_change)
_on_split_method_change()  # Initial state
```

---

## 4. Rolling Window Configuration

### Objective
Configure sequence length, prediction horizon, and rolling window parameters.

### Architecture Design

#### A. Window Configuration

```python
# window_config.py

@dataclass
class WindowConfig:
    """Configuration for rolling windows and sequences."""

    seq_len: int = 32  # Lookback window
    horizon: int = 10  # Prediction horizon
    stride: int = 1  # Step size for rolling window

    # Multi-horizon settings
    multi_horizon_enabled: bool = False
    horizons: List[int] = None  # [1, 3, 5, 10, 15, 20, 30]

    # Overlap handling
    allow_overlap: bool = True
    min_gap: int = 0  # Minimum gap between sequences

    def validate(self):
        if self.seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
```

#### B. GUI Integration

**Add to Train/Config tab:**

```python
frm_window = LabelFrame(tab, text="Sequence & Horizon Configuration")
frm_window.pack(fill=X, padx=6, pady=6)

Label(frm_window, text="Sequence length:").grid(row=0, column=0, sticky=W)
self.seq_len = IntVar(value=32)
Entry(frm_window, textvariable=self.seq_len, width=8).grid(row=0, column=1)

Label(frm_window, text="Prediction horizon:").grid(row=1, column=0, sticky=W)
self.horizon = IntVar(value=10)
Entry(frm_window, textvariable=self.horizon, width=8).grid(row=1, column=1)

Label(frm_window, text="Rolling stride:").grid(row=2, column=0, sticky=W)
self.stride_var = IntVar(value=1)
Entry(frm_window, textvariable=self.stride_var, width=8).grid(row=2, column=1)

# Multi-horizon configuration
Checkbutton(frm_window, text="Enable multi-horizon",
            variable=self.multi_horizon_enabled_var).grid(row=3, column=0, columnspan=2)

Label(frm_window, text="Horizons (comma-separated):").grid(row=4, column=0, sticky=W)
self.horizons_var = StringVar(value="1,3,5,10,15,20,30")
Entry(frm_window, textvariable=self.horizons_var, width=20).grid(row=4, column=1)

# Validation button
Button(frm_window, text="Validate Config",
       command=self._validate_window_config).grid(row=5, column=0, columnspan=2)
```

---

## 5. Multiple Prediction Fans

### Objective
Display multiple prediction fans simultaneously (different models, confidence levels, scenarios).

### Architecture Design

#### A. Multi-Fan Data Structure

```python
# multi_fan_predictions.py

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PredictionFan:
    """Single prediction fan with metadata."""

    name: str
    predictions: Dict[int, Dict]  # {horizon: prediction_data}
    model_id: str
    timestamp: str
    color: str = "viridis"
    alpha: float = 1.0
    enabled: bool = True

    # Confidence level override
    confidence_level: float = 0.95  # Default 95%


class MultiF FanManager:
    """Manage multiple prediction fans."""

    def __init__(self):
        self.fans: Dict[str, PredictionFan] = {}

    def add_fan(self, fan: PredictionFan):
        """Add a prediction fan."""
        self.fans[fan.name] = fan

    def remove_fan(self, name: str):
        """Remove a prediction fan."""
        if name in self.fans:
            del self.fans[name]

    def get_active_fans(self) -> List[PredictionFan]:
        """Get list of enabled fans."""
        return [fan for fan in self.fans.values() if fan.enabled]

    def toggle_fan(self, name: str):
        """Toggle fan visibility."""
        if name in self.fans:
            self.fans[name].enabled = not self.fans[name].enabled
```

#### B. Visualization Enhancement

**Extend dashboard_visualizations_simple.py:**

```python
def plot_multi_fan_comparison(
    ax,
    df_history: pd.DataFrame,
    fans: List[PredictionFan],
    n_history: int = 100
):
    """
    Plot multiple prediction fans for comparison.

    Args:
        ax: matplotlib axis
        df_history: Historical OHLCV data
        fans: List of PredictionFan objects
        n_history: Number of historical candles
    """
    # Plot historical (once)
    df_hist = df_history.tail(n_history).copy()
    hist_prices = df_hist["close"].values
    hist_times = pd.to_datetime(df_hist["timestamp"])

    ax.plot(hist_times, hist_prices, color="black", linewidth=2.5,
            label="Historical", zorder=10)

    # Current price marker
    current_price = hist_prices[-1]
    current_time = hist_times.iloc[-1]
    ax.scatter([current_time], [current_price], color="red", s=120,
              marker="o", edgecolors="white", linewidths=2, zorder=15)

    # Plot each fan
    for fan in fans:
        if not fan.enabled:
            continue

        cmap = plt.get_cmap(fan.color)
        horizons = sorted(fan.predictions.keys())
        colors = [cmap(i / max(1, len(horizons) - 1)) for i in range(len(horizons))]

        for i, h in enumerate(horizons):
            pred = fan.predictions[h]
            future_time = current_time + pd.Timedelta(hours=h)
            pred_price = pred["price"]

            # Draw prediction line
            ax.plot([current_time, future_time], [current_price, pred_price],
                    color=colors[i], linewidth=1.5, alpha=fan.alpha,
                    label=f"{fan.name} h={h}", zorder=5)

            # Confidence band (lighter)
            if "ci_lower_95" in pred and "ci_upper_95" in pred:
                ax.fill_between([current_time, future_time],
                               [current_price, pred["ci_lower_95"]],
                               [current_price, pred["ci_upper_95"]],
                               color=colors[i], alpha=fan.alpha * 0.1, zorder=1)

    ax.legend(loc="best", fontsize=7, ncol=3)
    # ... (rest of formatting)
```

#### C. GUI Integration

**Add to Status tab:**

```python
# Fan management panel
frm_fan_mgmt = LabelFrame(tab, text="Prediction Fans")
frm_fan_mgmt.pack(fill=X, padx=6, pady=6)

# List of fans with checkboxes
self.fans_tree = ttk.Treeview(frm_fan_mgmt, columns=("Name", "Model", "Color"), height=4)
for col in ("Name", "Model", "Color"):
    self.fans_tree.heading(col, text=col)
self.fans_tree.pack(fill=X)

Button(frm_fan_mgmt, text="Add Fan", command=self._add_prediction_fan).pack(side=LEFT)
Button(frm_fan_mgmt, text="Remove Fan", command=self._remove_prediction_fan).pack(side=LEFT)
Button(frm_fan_mgmt, text="Configure", command=self._configure_fan).pack(side=LEFT)
```

---

## Implementation Priorities

### Phase 1: Foundation (Immediate)
1. ✅ Multi-horizon prediction system (COMPLETED)
2. ✅ Status tab integration (COMPLETED)
3. Feature engineering registry system
4. Val/train split configuration

### Phase 2: Enhanced Configuration (Near-term)
1. GUI controls for split configuration
2. Rolling window parameter editor
3. Feature system selector in GUI
4. Model compatibility validation

### Phase 3: Advanced Features (Medium-term)
1. GUI-based feature editor (basic blocks)
2. Multiple prediction fans display
3. Fan comparison tools
4. Export/import custom configurations

### Phase 4: Production Features (Long-term)
1. Advanced feature editor (custom code blocks)
2. Automated hyperparameter optimization
3. Model ensemble support
4. A/B testing framework for feature systems

---

## Code Organization

### Recommended File Structure

```
BitCorn_Farmer/
├── core/
│   ├── feature_engineering_registry.py
│   ├── feature_builder.py
│   ├── split_config.py
│   ├── window_config.py
│   └── multi_fan_predictions.py
│
├── gui/
│   ├── feature_editor_window.py
│   ├── split_config_window.py
│   └── fan_management_window.py
│
├── models/
│   ├── model_v1/  (39 features)
│   ├── model_v2/  (clean features)
│   └── custom_models/
│
└── configs/
    ├── feature_systems/
    │   ├── v1.json
    │   ├── v2.json
    │   └── custom_*.json
    └── training_configs/
        └── default.json
```

---

## Configuration File Format

### Feature System Config (JSON)

```json
{
  "name": "v2_custom",
  "version": "2.1",
  "description": "Custom feature set based on v2",
  "blocks": [
    {
      "type": "LogReturn",
      "params": {"period": 1},
      "output_name": "log_ret_1"
    },
    {
      "type": "SMA",
      "params": {"window": 20, "ratio": true},
      "output_name": "sma_ratio_20"
    },
    {
      "type": "RSI",
      "params": {"period": 14},
      "output_name": "rsi_14"
    }
  ],
  "expected_features": ["log_ret_1", "sma_ratio_20", "rsi_14", "..."]
}
```

### Training Config (JSON)

```json
{
  "split": {
    "method": "rolling",
    "val_fraction": 0.1,
    "n_splits": 5,
    "train_window": null
  },
  "window": {
    "seq_len": 32,
    "horizon": 10,
    "stride": 1,
    "multi_horizon": {
      "enabled": true,
      "horizons": [1, 3, 5, 10, 15, 20, 30]
    }
  },
  "training": {
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.001,
    "hidden": 64
  }
}
```

---

## Next Steps for User

1. **Review this document** and prioritize features based on immediate needs

2. **Start with Feature Registry** (easiest first step):
   - Implement `feature_engineering_registry.py`
   - Add selector to GUI
   - Test toggling between v1 and v2

3. **Add Split Configuration** (moderate complexity):
   - Implement `split_config.py`
   - Add GUI controls to Train tab
   - Test different split strategies

4. **Implement Feature Editor** (most complex):
   - Start with basic blocks (LogReturn, SMA, RSI)
   - Build GUI for pipeline editing
   - Add validation and testing tools

5. **Extend to Multi-Fan Display**:
   - Implement `MultiF FanManager`
   - Update visualization functions
   - Add fan management UI

---

## Compatibility Notes

- All enhancements designed to be **backward-compatible**
- Existing models will continue to work with v1 features
- New feature systems can coexist with old ones
- Configuration files optional (defaults to current behavior)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-31
**Status:** Ready for implementation
