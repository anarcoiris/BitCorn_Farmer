#!/usr/bin/env python3
"""
retrain_clean_features.py

Retrain LSTM model with CLEAN features (no price-level leakage).

Key improvements:
1. Uses add_technical_features_v2() - removes log_close, absolute MAs/EMAs, absolute Fibonacci
2. Uses combined_loss() - adds directional accuracy and variance matching penalties
3. Increased model capacity - hidden=128, layers=3 (from 92, 2)
4. Tracks additional metrics - directional accuracy, variance ratio, correlation

Target metrics (vs old model):
- MAE improvement >15% vs persistence (currently 3%)
- Directional accuracy >60% (currently 54.8%)
- Correlation >0.5 (currently 0.226)
- Variance ratio >0.5 (currently 0.15)

Usage:
    python retrain_clean_features.py --epochs 100 --lr 0.001 --batch-size 256
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import joblib

from fiboevo import (
    LSTM2Head,
    add_technical_features_v2,
    create_sequences_from_df,
    combined_loss,
    seed_everything
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(db_path: str) -> pd.DataFrame:
    """
    Load data from SQLite and compute clean features using v2.
    """
    logger.info(f"Loading data from {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM ohlcv ORDER BY ts", conn)
    conn.close()

    logger.info(f"Loaded {len(df)} rows")

    # Add timestamp
    if "timestamp" not in df.columns and "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Compute CLEAN features using v2
    logger.info("Computing clean technical features (v2)...")
    df_features = add_technical_features_v2(
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        volume=df.get("volume", None),
        fib_lookback=50,
        dropna_after=False,
        out_dtype="float32"
    )

    # Re-add required columns
    df_features["close"] = df["close"].values[:len(df_features)]
    df_features["timestamp"] = df["timestamp"].values[:len(df_features)]

    # Drop NaN rows
    before_len = len(df_features)
    df_features = df_features.dropna().reset_index(drop=True)
    after_len = len(df_features)

    logger.info(f"Dropped {before_len - after_len} NaN rows")
    logger.info(f"Final dataset: {len(df_features)} rows")
    logger.info(f"Features: {[c for c in df_features.columns if c not in ['close', 'timestamp']]}")

    return df_features


def build_dataset_v2(
    df: pd.DataFrame,
    seq_len: int = 64,
    horizon: int = 12,
    val_frac: float = 0.2
):
    """
    Build training and validation datasets with temporal split.
    """
    # Get feature columns (exclude close and timestamp)
    feature_cols = [c for c in df.columns if c not in ['close', 'timestamp', 'high', 'low']]

    logger.info(f"\nDataset Configuration:")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Sequence length: {seq_len}")
    logger.info(f"  Horizon: {horizon}")
    logger.info(f"  Validation fraction: {val_frac}")

    # Create sequences
    X_all, y_all, v_all = create_sequences_from_df(
        df, feature_cols, seq_len=seq_len, horizon=horizon
    )

    logger.info(f"  Total sequences: {len(X_all)}")

    # Temporal split (no shuffling!)
    n_sequences = len(X_all)
    n_val = max(1, int(round(val_frac * n_sequences)))
    n_train = n_sequences - n_val

    train_end_idx = seq_len + n_train - 1

    # Fit scaler on training data only
    scaler = StandardScaler()
    train_rows = df[list(feature_cols)].iloc[:train_end_idx + 1].values
    scaler.fit(train_rows.astype(np.float64))
    scaler.feature_names_in_ = np.array(list(feature_cols), dtype=object)

    logger.info(f"  Scaler fitted on {len(train_rows)} rows")

    # Scale all features
    df_scaled = df.copy()
    df_scaled[list(feature_cols)] = scaler.transform(
        df[list(feature_cols)].values.astype(np.float64)
    ).astype(np.float32)

    # Re-create sequences with scaled data
    X_all, y_all, v_all = create_sequences_from_df(
        df_scaled, feature_cols, seq_len=seq_len, horizon=horizon
    )

    # Split
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    v_train = v_all[:n_train]

    X_val = X_all[n_train:]
    y_val = y_all[n_train:]
    v_val = v_all[n_train:]

    logger.info(f"  Train sequences: {len(X_train)}")
    logger.info(f"  Val sequences: {len(X_val)}")

    return (X_train, y_train, v_train), (X_val, y_val, v_val), scaler, feature_cols


def compute_metrics(pred_ret, target_ret, prefix=""):
    """
    Compute evaluation metrics: MSE, directional accuracy, variance ratio, correlation.
    """
    pred_ret_np = pred_ret.detach().cpu().numpy().flatten()
    target_ret_np = target_ret.detach().cpu().numpy().flatten()

    # MSE
    mse = np.mean((pred_ret_np - target_ret_np) ** 2)

    # Directional accuracy
    pred_dir = np.sign(pred_ret_np)
    target_dir = np.sign(target_ret_np)
    dir_acc = (pred_dir == target_dir).mean() * 100

    # Variance ratio
    var_ratio = pred_ret_np.std() / (target_ret_np.std() + 1e-12)

    # Correlation
    corr = np.corrcoef(pred_ret_np, target_ret_np)[0, 1]

    metrics = {
        f"{prefix}mse": mse,
        f"{prefix}dir_acc": dir_acc,
        f"{prefix}var_ratio": var_ratio,
        f"{prefix}corr": corr
    }

    return metrics


def train_epoch_v2(model, loader, optimizer, device, use_combined_loss=True):
    """
    Train for one epoch using combined_loss or standard MSE.
    """
    model.train()
    total_loss = 0.0
    all_pred_ret = []
    all_target_ret = []
    n = 0

    for batch in loader:
        xb, yret, yvol = batch
        xb = xb.to(device)
        yret = yret.to(device)
        yvol = yvol.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_ret, pred_vol = model(xb)

        if use_combined_loss:
            loss = combined_loss(
                pred_ret, pred_vol, yret, yvol,
                alpha_vol=0.5, alpha_dir=0.3, alpha_var=0.2
            )
        else:
            # Standard MSE loss
            mse = nn.MSELoss()
            loss = mse(pred_ret, yret) + 0.5 * mse(pred_vol, yvol)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        # Collect for metrics
        all_pred_ret.append(pred_ret.detach())
        all_target_ret.append(yret.detach())

    avg_loss = total_loss / max(1, n)

    # Compute metrics
    all_pred_ret = torch.cat(all_pred_ret)
    all_target_ret = torch.cat(all_target_ret)
    metrics = compute_metrics(all_pred_ret, all_target_ret, prefix="train_")

    return avg_loss, metrics


def eval_epoch_v2(model, loader, device):
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0.0
    all_pred_ret = []
    all_target_ret = []
    n = 0

    mse = nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            xb, yret, yvol = batch
            xb = xb.to(device)
            yret = yret.to(device)
            yvol = yvol.to(device)

            pred_ret, pred_vol = model(xb)
            loss = mse(pred_ret, yret) + 0.5 * mse(pred_vol, yvol)

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

            all_pred_ret.append(pred_ret)
            all_target_ret.append(yret)

    avg_loss = total_loss / max(1, n)

    # Compute metrics
    all_pred_ret = torch.cat(all_pred_ret)
    all_target_ret = torch.cat(all_target_ret)
    metrics = compute_metrics(all_pred_ret, all_target_ret, prefix="val_")

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Retrain LSTM with clean features")
    parser.add_argument("--data", type=str, default="data_manager/exports/Binance_BTCUSDT_1h_fix.db")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size (increased from 92)")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers (increased from 2)")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-combined-loss", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="artifacts_v2")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("RETRAINING WITH CLEAN FEATURES (v2)")
    logger.info("="*70)
    logger.info(f"\nConfiguration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Set seed
    seed_everything(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load and prepare data
    df = load_and_prepare_data(args.data)

    # Build datasets
    (X_train, y_train, v_train), (X_val, y_val, v_val), scaler, feature_cols = build_dataset_v2(
        df, seq_len=args.seq_len, horizon=args.horizon, val_frac=args.val_frac
    )

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(v_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
        torch.from_numpy(v_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")

    model = LSTM2Head(
        input_size=len(feature_cols),
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    logger.info(f"\nModel architecture:")
    logger.info(f"  Input size: {len(feature_cols)}")
    logger.info(f"  Hidden size: {args.hidden}")
    logger.info(f"  Num layers: {args.layers}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info("="*70)

    best_val_loss = float('inf')
    best_val_dir_acc = 0.0
    patience_counter = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_dir_acc': [], 'val_dir_acc': [],
        'train_var_ratio': [], 'val_var_ratio': [],
        'train_corr': [], 'val_corr': []
    }

    for epoch in trange(args.epochs, desc="Epochs"):
        # Train
        train_loss, train_metrics = train_epoch_v2(
            model, train_loader, optimizer, device,
            use_combined_loss=args.use_combined_loss
        )

        # Evaluate
        val_loss, val_metrics = eval_epoch_v2(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dir_acc'].append(train_metrics['train_dir_acc'])
        history['val_dir_acc'].append(val_metrics['val_dir_acc'])
        history['train_var_ratio'].append(train_metrics['train_var_ratio'])
        history['val_var_ratio'].append(val_metrics['val_var_ratio'])
        history['train_corr'].append(train_metrics['train_corr'])
        history['val_corr'].append(val_metrics['val_corr'])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
            logger.info(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            logger.info(f"  Train Dir Acc: {train_metrics['train_dir_acc']:.2f}% | Val Dir Acc: {val_metrics['val_dir_acc']:.2f}%")
            logger.info(f"  Train Var Ratio: {train_metrics['train_var_ratio']:.4f} | Val Var Ratio: {val_metrics['val_var_ratio']:.4f}")
            logger.info(f"  Train Corr: {train_metrics['train_corr']:.4f} | Val Corr: {val_metrics['val_corr']:.4f}")

        # Save best model (based on directional accuracy)
        if val_metrics['val_dir_acc'] > best_val_dir_acc:
            best_val_dir_acc = val_metrics['val_dir_acc']
            best_val_loss = val_loss
            patience_counter = 0

            # Save model
            torch.save(model.state_dict(), output_dir / "model_best.pt")
            logger.info(f"  → Best model saved (dir_acc={best_val_dir_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
            break

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best validation directional accuracy: {best_val_dir_acc:.2f}%")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    # Save metadata
    meta = {
        "feature_cols": feature_cols,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "hidden": args.hidden,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "best_val_dir_acc": best_val_dir_acc,
        "best_val_loss": best_val_loss
    }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save scaler
    joblib.dump(scaler, output_dir / "scaler.pkl")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    logger.info(f"\nArtifacts saved to: {output_dir}")
    logger.info(f"  - model_best.pt")
    logger.info(f"  - meta.json")
    logger.info(f"  - scaler.pkl")
    logger.info(f"  - training_history.csv")

    logger.info("\nNext steps:")
    logger.info("  1. Run inference with clean features: python example_multi_horizon.py --use-v2")
    logger.info("  2. Compare metrics with old model")
    logger.info("  3. Validate improvement >15% MAE, >60% dir_acc, >0.5 corr")


if __name__ == "__main__":
    main()
