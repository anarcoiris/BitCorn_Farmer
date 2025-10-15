#!/usr/bin/env python3
"""
Diagnostic script to analyze why the LSTM model is producing "naive" predictions
that closely follow the current price instead of predicting actual future changes.

This script will:
1. Load the trained model and analyze its predictions
2. Check for data leakage in features
3. Compute persistence forecast baseline
4. Visualize prediction distributions and errors
5. Provide recommendations for retraining
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# Import from existing modules
from multi_horizon_inference import load_model_and_artifacts, predict_multi_horizon_jump
from fiboevo import add_technical_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_feature_leakage(df: pd.DataFrame, feature_cols: list) -> Dict[str, Any]:
    """
    Check for potential data leakage in features.
    Features that directly contain future information or close price absolute values
    can cause the model to learn trivial patterns.
    """
    logger.info("\n" + "="*70)
    logger.info("FEATURE LEAKAGE ANALYSIS")
    logger.info("="*70)

    issues = []
    warnings_list = []

    # Check for absolute price features
    absolute_price_features = [f for f in feature_cols if any(x in f for x in ['log_close', 'close', 'price'])]
    if 'log_close' in absolute_price_features:
        issues.append({
            'feature': 'log_close',
            'severity': 'HIGH',
            'reason': 'Absolute log_close contains information about P_t directly. Model can use this to predict P_t instead of learning ΔP.'
        })

    # Check correlation between features and close price
    if 'close' in df.columns:
        correlations = {}
        for feat in feature_cols:
            if feat in df.columns:
                corr = df[feat].corr(df['close'])
                correlations[feat] = corr
                if abs(corr) > 0.95 and feat != 'close':
                    warnings_list.append({
                        'feature': feat,
                        'severity': 'MEDIUM',
                        'reason': f'Very high correlation with close price: {corr:.4f}'
                    })

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        logger.info(f"\nTop 10 features by correlation with close price:")
        for feat, corr in sorted_corr[:10]:
            logger.info(f"  {feat:20s}: {corr:+.4f}")

    logger.info(f"\n{len(issues)} CRITICAL issues found:")
    for issue in issues:
        logger.info(f"  [{issue['severity']}] {issue['feature']}: {issue['reason']}")

    logger.info(f"\n{len(warnings_list)} warnings:")
    for warning in warnings_list:
        logger.info(f"  [{warning['severity']}] {warning['feature']}: {warning['reason']}")

    return {
        'issues': issues,
        'warnings': warnings_list,
        'correlations': correlations if 'close' in df.columns else {}
    }


def compute_persistence_baseline(predictions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute metrics for a "persistence forecast" baseline.
    Persistence forecast: assume price doesn't change: P_{t+h} = P_t

    This is the simplest possible forecast and our model should beat it significantly.
    """
    logger.info("\n" + "="*70)
    logger.info("PERSISTENCE FORECAST BASELINE")
    logger.info("="*70)

    valid = predictions_df.dropna(subset=['close_actual_future', 'close_current'])

    # Persistence forecast error: assume no change
    persistence_error = valid['close_actual_future'] - valid['close_current']
    persistence_mae = persistence_error.abs().mean()
    persistence_rmse = np.sqrt((persistence_error ** 2).mean())
    persistence_mape = (persistence_error.abs() / valid['close_current']).mean() * 100

    # Model error
    model_error = valid['prediction_error']
    model_mae = model_error.abs().mean()
    model_rmse = np.sqrt((model_error ** 2).mean())
    model_mape = (model_error.abs() / valid['close_current']).mean() * 100

    # Directional accuracy for persistence (always predicts no change = wrong direction)
    actual_direction = (valid['close_actual_future'] > valid['close_current']).astype(int)
    persistence_direction = 0  # no change
    persistence_dir_acc = (actual_direction == persistence_direction).mean() * 100

    # Model directional accuracy
    model_direction = (valid['close_pred'] > valid['close_current']).astype(int)
    model_dir_acc = (actual_direction == model_direction).mean() * 100

    logger.info(f"\nPersistence Forecast (P_{{t+12}} = P_t):")
    logger.info(f"  MAE:  ${persistence_mae:.2f}")
    logger.info(f"  RMSE: ${persistence_rmse:.2f}")
    logger.info(f"  MAPE: {persistence_mape:.2f}%")
    logger.info(f"  Directional Accuracy: {persistence_dir_acc:.1f}%")

    logger.info(f"\nModel Performance:")
    logger.info(f"  MAE:  ${model_mae:.2f}")
    logger.info(f"  RMSE: ${model_rmse:.2f}")
    logger.info(f"  MAPE: {model_mape:.2f}%")
    logger.info(f"  Directional Accuracy: {model_dir_acc:.1f}%")

    improvement_mae = ((persistence_mae - model_mae) / persistence_mae) * 100
    improvement_dir = model_dir_acc - persistence_dir_acc

    logger.info(f"\nImprovement over Persistence:")
    logger.info(f"  MAE improvement: {improvement_mae:+.1f}%")
    logger.info(f"  Directional improvement: {improvement_dir:+.1f} percentage points")

    if improvement_mae < 5:
        logger.warning("⚠️  Model is barely better than persistence forecast!")
    if model_dir_acc < 55:
        logger.warning("⚠️  Directional accuracy is barely better than random (50%)!")

    return {
        'persistence_mae': persistence_mae,
        'persistence_rmse': persistence_rmse,
        'persistence_mape': persistence_mape,
        'persistence_dir_acc': persistence_dir_acc,
        'model_mae': model_mae,
        'model_rmse': model_rmse,
        'model_mape': model_mape,
        'model_dir_acc': model_dir_acc,
        'improvement_mae_pct': improvement_mae,
        'improvement_dir_points': improvement_dir
    }


def analyze_prediction_distribution(predictions_df: pd.DataFrame, save_path: str = "diagnostic_plots.png"):
    """
    Create diagnostic plots to visualize model behavior.
    """
    logger.info("\n" + "="*70)
    logger.info("PREDICTION DISTRIBUTION ANALYSIS")
    logger.info("="*70)

    valid = predictions_df.dropna(subset=['close_actual_future', 'close_pred', 'close_current'])

    # Compute returns
    actual_return = (valid['close_actual_future'] - valid['close_current']) / valid['close_current']
    predicted_return = (valid['close_pred'] - valid['close_current']) / valid['close_current']

    logger.info(f"\nReturn Statistics:")
    logger.info(f"  Actual returns:    mean={actual_return.mean():.4f}, std={actual_return.std():.4f}")
    logger.info(f"  Predicted returns: mean={predicted_return.mean():.4f}, std={predicted_return.std():.4f}")

    # Check if predictions are just scaled-down versions of actuals
    scaling_factor = predicted_return.std() / actual_return.std()
    logger.info(f"  Prediction scale factor: {scaling_factor:.4f}")

    if scaling_factor < 0.3:
        logger.warning("⚠️  Predictions have much lower variance than actuals!")
        logger.warning("    This suggests the model is 'hedging' by predicting small changes.")

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Scatter - Predicted vs Actual Returns
    ax1 = axes[0, 0]
    ax1.scatter(actual_return, predicted_return, alpha=0.5, s=20)
    ax1.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual Return (12h ahead)')
    ax1.set_ylabel('Predicted Return (12h ahead)')
    ax1.set_title('Predicted vs Actual Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add correlation
    corr = np.corrcoef(actual_return, predicted_return)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Distribution of Prediction Errors
    ax2 = axes[0, 1]
    errors_pct = valid['prediction_error_pct']
    ax2.hist(errors_pct, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(errors_pct.mean(), color='red', linestyle='--', label=f'Mean: {errors_pct.mean():.2f}%')
    ax2.axvline(errors_pct.median(), color='green', linestyle='--', label=f'Median: {errors_pct.median():.2f}%')
    ax2.set_xlabel('Prediction Error (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Return distributions
    ax3 = axes[1, 0]
    ax3.hist(actual_return, bins=50, alpha=0.5, label='Actual', edgecolor='black')
    ax3.hist(predicted_return, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Return (fraction)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Prediction bias over time
    ax4 = axes[1, 1]
    window_size = 50
    rolling_bias = errors_pct.rolling(window=window_size, center=True).mean()
    ax4.plot(rolling_bias, alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1)
    ax4.set_xlabel('Prediction Index')
    ax4.set_ylabel(f'Rolling Mean Error (%) [window={window_size}]')
    ax4.set_title('Prediction Bias Over Time')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"\nDiagnostic plots saved to: {save_path}")

    return {
        'actual_return_mean': actual_return.mean(),
        'actual_return_std': actual_return.std(),
        'predicted_return_mean': predicted_return.mean(),
        'predicted_return_std': predicted_return.std(),
        'scaling_factor': scaling_factor,
        'correlation': corr
    }


def generate_recommendations(leakage_analysis: Dict, baseline_metrics: Dict, distribution_metrics: Dict) -> str:
    """
    Generate actionable recommendations based on diagnostics.
    """
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS")
    logger.info("="*70)

    recommendations = []

    # Check for data leakage
    if len(leakage_analysis['issues']) > 0:
        recommendations.append({
            'priority': 'CRITICAL',
            'action': 'Remove log_close from features',
            'reason': 'log_close creates data leakage - model can memorize P_t and predict P_t + small noise',
            'implementation': 'Retrain with features excluding log_close. Keep only returns/differences.'
        })

    # Check model performance vs baseline
    if baseline_metrics['improvement_mae_pct'] < 10:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Model is barely better than persistence forecast',
            'reason': f"Only {baseline_metrics['improvement_mae_pct']:.1f}% improvement in MAE",
            'implementation': 'Increase model capacity, adjust learning rate, or change architecture'
        })

    # Check prediction variance
    if distribution_metrics['scaling_factor'] < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Model predictions have low variance',
            'reason': f"Pred/Actual std ratio = {distribution_metrics['scaling_factor']:.2f}",
            'implementation': 'Add directional accuracy loss term, reduce regularization, or use different loss function'
        })

    # Check directional accuracy
    if baseline_metrics['model_dir_acc'] < 55:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Poor directional accuracy',
            'reason': f"Only {baseline_metrics['model_dir_acc']:.1f}% (barely better than 50% random)",
            'implementation': 'Add directional loss component or focal loss for imbalanced directions'
        })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n{i}. [{rec['priority']}] {rec['action']}")
        logger.info(f"   Reason: {rec['reason']}")
        logger.info(f"   Implementation: {rec['implementation']}")

    if len(recommendations) == 0:
        logger.info("\n✓ No critical issues found. Model seems reasonable.")
    else:
        logger.info(f"\n⚠️  Found {len(recommendations)} issues requiring attention.")

    # Specific retraining recommendations
    logger.info("\n" + "-"*70)
    logger.info("RETRAINING STRATEGY:")
    logger.info("-"*70)
    logger.info("""
1. **Remove Data Leakage:**
   - Exclude 'log_close' from feature_cols
   - Verify no other absolute price features remain

2. **Adjust Loss Function:**
   - Add directional accuracy penalty
   - Consider Huber loss or focal loss
   - Weight by volatility (penalize more in volatile periods)

3. **Increase Model Capacity:**
   - Try hidden_size=128 or 256 (currently 92)
   - Add more layers (num_layers=3 or 4)
   - Consider attention mechanism

4. **Multi-Horizon Architecture:**
   - Implement h separate output heads: log_ret_1, log_ret_2, ..., log_ret_12
   - This gives denser predictions and better gradient flow

5. **Training Improvements:**
   - Longer training (more epochs)
   - Learning rate scheduling (cosine annealing)
   - Data augmentation (train on multiple symbols/timeframes)

6. **Validation:**
   - Ensure train/val split is temporal (not random)
   - Track directional accuracy during training
   - Use early stopping based on directional accuracy, not just MSE
""")

    return recommendations


def main():
    """
    Run full diagnostic analysis.
    """
    logger.info("="*70)
    logger.info("LSTM MODEL DIAGNOSTIC TOOL")
    logger.info("="*70)

    # Paths
    MODEL_PATH = "artifacts/model_best.pt"
    META_PATH = "artifacts/meta.json"
    SCALER_PATH = "artifacts/scaler.pkl"
    DATA_PATH = "data_manager/exports/Binance_BTCUSDT_1h_fix.db"
    PREDICTIONS_CSV = "predictions_output.csv"

    # Step 1: Load predictions (if they exist)
    predictions_df = None
    if Path(PREDICTIONS_CSV).exists():
        logger.info(f"\nLoading existing predictions from {PREDICTIONS_CSV}...")
        predictions_df = pd.read_csv(PREDICTIONS_CSV)
        logger.info(f"Loaded {len(predictions_df)} predictions")
    else:
        logger.info(f"\nPredictions file not found. Please run example_multi_horizon.py first.")
        return

    # Step 2: Load data for feature analysis
    logger.info(f"\nLoading data from {DATA_PATH}...")
    import sqlite3
    conn = sqlite3.connect(DATA_PATH)
    df = pd.read_sql_query("SELECT * FROM ohlcv ORDER BY ts", conn)
    conn.close()

    # Add timestamp
    if "timestamp" not in df.columns and "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Compute features
    logger.info("Computing technical features...")
    timestamps = df["timestamp"].copy()
    df_features = add_technical_features(
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        volume=df["volume"].values,
        dropna_after=False
    )
    df_features["timestamp"] = timestamps
    df_features = df_features.dropna().reset_index(drop=True)

    # Load meta to get feature list
    import json
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta['feature_cols']

    # Step 3: Run diagnostics
    leakage_analysis = analyze_feature_leakage(df_features, feature_cols)
    baseline_metrics = compute_persistence_baseline(predictions_df)
    distribution_metrics = analyze_prediction_distribution(predictions_df, save_path="model_diagnostics.png")

    # Step 4: Generate recommendations
    recommendations = generate_recommendations(leakage_analysis, baseline_metrics, distribution_metrics)

    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*70)
    logger.info(f"\nResults saved to: model_diagnostics.png")
    logger.info("Review recommendations above and decide on retraining strategy.")


if __name__ == "__main__":
    main()
