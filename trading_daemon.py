#!/usr/bin/env python3
"""
trading_daemon.py

Daemon que corre en background para:
 - leer datos recientes (sqlite)
 - crear features (usando fiboevo si está presente)
 - construir secuencias coherentes con el entrenamiento
 - aplicar scaler/model para predecir
 - tomar decisiones y simular/colocar órdenes (paper vs live via ccxt)
 - registrar ledger y logs

Diseñado para integrarse con trading_gui_extended.py:
 - constructor con parámetros coherentes con la GUI
 - métodos .start_loop() y .stop()
 - puede recibir una queue de logs para integrarse con la UI
 - rutas de artifacts compatibles (./artifacts/)
"""

from __future__ import annotations
import threading
import time
import logging
import json
import math
import queue
from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple
import sqlite3
import csv
import traceback
import argparse

import numpy as np
import pandas as pd

# Optional external packages
try:
    import torch
except Exception:
    torch = None

try:
    import joblib
except Exception:
    joblib = None

try:
    import ccxt
except Exception:
    ccxt = None

# Try to import fiboevo (features, sequence helpers, model class)
try:
    import fiboevo as fibo
except Exception:
    fibo = None

# Multi-horizon prediction support (for dashboard integration)
try:
    from multi_horizon_fan_inference import predict_single_point_multi_horizon
except ImportError:
    predict_single_point_multi_horizon = None

# Feature engineering registry
try:
    from core.feature_registry import FEATURE_REGISTRY
except ImportError:
    FEATURE_REGISTRY = None

# import config_manager
try:
    from config_manager import load_config, config_info
    from fp_utils import normalize_symbol, normalize_timeframe, to_tensor_float32
except Exception:
    # fallback local behavior: these functions are optional
    pass


# Logger
logger = logging.getLogger("TradingDaemon")
if not logger.handlers:
    # basic config if not configured by app
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    # logger.addHandler(ch)
    logger.setLevel(logging.INFO)


class TradingDaemon:
    """
    Background trading daemon.

    Notes:
    - Accepts ledger_path parameter for compatibility with GUI.
    - Accepts **kwargs so future caller-added args don't raise TypeError.
    - Configuration precedence: explicit constructor args > daemon_cfg.json > defaults.
    """

    def __init__(
        self,
        sqlite_path: str,
        sqlite_table: str,
        symbol: str,
        timeframe: str,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        ledger_path: Optional[str] = None,
        exchange_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
        seq_len: int = 32,
        feature_cols: Optional[List[str]] = None,
        poll_interval: float = 5.0,
        log_queue: Optional[Any] = None,
        logger_obj: Optional[logging.Logger] = None,
        config_path: Optional[str] = None,       # ruta JSON opcional (artifacts/daemon_cfg.json por defecto)
        auto_load_artifacts: bool = True,        # cargar modelo al crear la instancia
        **kwargs,
    ):

        # UI logging queue & settings (en __init__)
        import queue as _queue_mod

        # If a log_queue was provided by the caller (e.g. GUI), use it as the UI queue.
        # Otherwise create an internal queue so daemon still logs to a UI pane if desired.
        if log_queue is not None:
            # caller passed a queue object -> use it as the ui log queue
            self.ui_log_queue = log_queue
            self._external_log_queue = True
        else:
            self.ui_log_queue = _queue_mod.Queue(maxsize=2000)
            self._external_log_queue = False

        # Backwards-compatible: keep self.log_queue attribute (the parameter) as well
        # (some code may still expect self.log_queue). If caller passed log_queue, self.log_queue==self.ui_log_queue.
        self.log_queue = log_queue

        self.log_flush_interval_ms = 200                     # intervalo de vaciado (ms)
        # opcional: expón variable para controlar el nivel de logs mostrados
        self.ui_log_level = logging.DEBUG


        # Locks
        self._config_lock = threading.Lock()
        self._artifact_lock = threading.Lock()

        # artifacts dir: crear primero para poder usar config default dentro de artifacts
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # config path: priorizar argumento, si no usar artifacts/daemon_cfg.json
        self.config_path = Path(config_path) if config_path else (self.artifacts_dir / "daemon_cfg.json")
        self.auto_load_artifacts = bool(auto_load_artifacts)

        # Try load config file (non-fatal)
        cfg: Dict[str, Any] = {}
        try:
            if self.config_path.exists():
                with self.config_path.open("r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                    logger.info("Loaded daemon config from %s", self.config_path)
        except Exception:
            logger.exception("Failed to read config file %s", self.config_path)

        # small helper: prefer explicit arg (arg_val) if not None, otherwise cfg[name], otherwise cfg_def
        def _val(name: str, arg_val, cfg_def=None):
            if arg_val is not None:
                return arg_val
            if name in cfg:
                return cfg.get(name)
            return cfg_def

        # Basic config (constructor args take precedence)
        self.sqlite_path = Path(_val("sqlite_path", sqlite_path))
        self.sqlite_table = _val("sqlite_table", sqlite_table)
        self.symbol = _val("symbol", symbol)
        self.timeframe = _val("timeframe", timeframe)
        self.model_path = Path(_val("model_path", model_path, str(self.artifacts_dir / "model_best.pt")))
        self.scaler_path = Path(_val("scaler_path", scaler_path, str(self.artifacts_dir / "scaler.pkl")))
        self.meta_path = Path(_val("meta_path", meta_path, str(self.artifacts_dir / "meta.json")))
        self.exchange_id = _val("exchange_id", exchange_id)
        # NOTE: prefer not to persist api_key/api_secret in disk; we still accept them in runtime args or cfg if provided
        self.api_key = _val("api_key", api_key)
        self.api_secret = _val("api_secret", api_secret)
        self.paper = bool(_val("paper", paper, True))
        self.seq_len = int(_val("seq_len", seq_len, 32))
        self.feature_cols = _val("feature_cols", feature_cols, None)
        self.poll_interval = float(_val("poll_interval", poll_interval, 5.0))

        self.log_queue = log_queue
        self.logger = logger_obj or logger

        # control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # model artifacts
        self.model = None
        self.model_meta: Dict[str, Any] = {}
        self.model_scaler = None

        # exchange client (create if we are in live mode and have ccxt)
        self.exchange = None
        if (not self.paper) and ccxt is not None and self.exchange_id:
            try:
                ex_cls = getattr(ccxt, self.exchange_id)
                self.exchange = ex_cls({"apiKey": self.api_key, "secret": self.api_secret})
                self.logger.info("Exchange client initialized: %s", self.exchange_id)
            except Exception as e:
                self.logger.exception("Failed to initialize exchange client: %s", e)
                self.exchange = None

        # ledger path: use provided ledger_path (from GUI) if any, otherwise artifacts/ledger.csv
        if ledger_path:
            try:
                self.ledger_path = Path(ledger_path)
            except Exception:
                self.ledger_path = self.artifacts_dir / "ledger.csv"
        else:
            # allow cfg to override ledger_path
            ledger_cfg = cfg.get("ledger_path") if cfg else None
            self.ledger_path = Path(ledger_cfg) if ledger_cfg else (self.artifacts_dir / "ledger.csv")

        # Ensure ledger dir exists
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Multi-horizon predictions queue (for GUI consumption)
        self.predictions_queue = _queue_mod.Queue(maxsize=10)

        # Control flag: enable/disable multi-horizon mode
        self.multi_horizon_mode = False  # Can be toggled from GUI
        self.multi_horizon_horizons = [1, 3, 5, 10, 15, 20, 30]  # Default horizons

        # Feature engineering system selection (v1=39 features, v2=14 clean)
        self.feature_system = "v2"  # Can be changed by GUI

        # Torch device for inference
        self.device = None  # Will be set in load_model_and_scaler

        self._enqueue_log(f"TradingDaemon initialized (symbol={self.symbol}, tf={self.timeframe}, paper={self.paper}, ledger={self.ledger_path}).")

        # auto load model/scaler if requested
        if self.auto_load_artifacts:
            # Nota: la carga es síncrona; si prefieres no bloquear UI, la GUI puede ejecutar esto en un hilo.
            try:
                self.load_model_and_scaler()
            except Exception:
                self.logger.exception("Auto-load artifacts failed")

    # ----------------------------
    # Logging helper
    # ----------------------------
    def _enqueue_log(self, msg: str, level: int = logging.INFO):
        """
        Enqueue message to UI log queue (if present) and emit to Python logger.
        - msg: text message
        - level: logging level (e.g. logging.INFO, logging.DEBUG)
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"

        # Truncate to avoid filling UI with giant messages
        max_len = 4000
        if isinstance(line, str) and len(line) > max_len:
            line = line[:max_len] + "...[truncated]"

        # Put into ui_log_queue if available (this is the queue the GUI reads)
        try:
            ui_q = getattr(self, "ui_log_queue", None)
            if ui_q is not None:
                try:
                    # use non-blocking put to avoid stalls
                    ui_q.put_nowait(line)
                except Exception:
                    try:
                        ui_q.put(line, block=False)
                    except Exception:
                        # best-effort: swallow to avoid blocking daemon
                        pass
        except Exception:
            # swallow queue errors to avoid crashing daemon
            pass

        # For backwards compatibility, also put into self.log_queue if it's different
        try:
            if getattr(self, "log_queue", None) and getattr(self, "log_queue") is not ui_q:
                try:
                    self.log_queue.put_nowait(line)
                except Exception:
                    try:
                        self.log_queue.put(line, block=False)
                    except Exception:
                        pass
        except Exception:
            pass

        # Emit to the python logger using requested level
        try:
            if self.logger:
                try:
                    self.logger.log(level, msg)
                except Exception:
                    # fallback to print if logger misbehaves
                    print(line)
            else:
                print(line)
        except Exception:
            # ultimate fallback
            try:
                print(line)
            except Exception:
                pass


    def attach_log_queue(self, q: Any, replace: bool = True):
        """
        Attach an external queue (from GUI) so daemon will push logs there.
        If replace=False and a queue already attached, the call is ignored.
        """
        if q is None:
            return
        if getattr(self, "ui_log_queue", None) is None or replace:
            self.ui_log_queue = q
            self._external_log_queue = True
            self._enqueue_log("Attached external UI log queue.", level=logging.DEBUG)


    # ----------------------------
    # Config helpers
    # ----------------------------
    def save_config(self, path: Optional[str] = None):
        """Escribe la configuración actual a JSON (por defecto artifacts/daemon_cfg.json).
        Por seguridad, no incluye api_key/api_secret a menos que explícitamente se añadan."""
        p = Path(path) if path else self.config_path
        try:
            cfg = {
                "sqlite_path": str(self.sqlite_path),
                "sqlite_table": self.sqlite_table,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "model_path": str(self.model_path),
                "scaler_path": str(self.scaler_path),
                "meta_path": str(self.meta_path),
                "ledger_path": str(self.ledger_path),
                "exchange_id": self.exchange_id,
                # "api_key": self.api_key,  # OMITIDOS por defecto por seguridad
                # "api_secret": self.api_secret,
                "paper": self.paper,
                "seq_len": self.seq_len,
                "feature_cols": self.feature_cols,
                "poll_interval": self.poll_interval,
            }
            with p.open("w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            self._enqueue_log(f"Saved daemon config to {p}")
        except Exception:
            self.logger.exception("Failed to save config")

    def update_from_dict(self, cfg: Dict[str, Any], save: bool = False, reload_artifacts: bool = False):
        """Aplicar un dict de configuración en caliente (thread-safe).
        Sólo actualiza claves conocidas para evitar sobrescribir internals."""
        with self._config_lock:
            path_keys = {"sqlite_path", "model_path", "scaler_path", "meta_path", "ledger_path"}
            allowed = {"sqlite_path", "sqlite_table", "symbol", "timeframe", "model_path", "scaler_path", "meta_path", "paper", "seq_len", "feature_cols", "poll_interval", "exchange_id", "api_key", "api_secret", "ledger_path"}
            for k in allowed:
                if k in cfg:
                    v = cfg[k]
                    if k in path_keys and v is not None:
                        setattr(self, k, Path(v))
                    else:
                        setattr(self, k, v)
            self._enqueue_log("Configuration updated from dict.")
            if save:
                try:
                    self.save_config()
                except Exception:
                    self.logger.exception("Failed to save config after update")
            if reload_artifacts:
                # reload artifacts synchronously
                try:
                    self.load_model_and_scaler()
                except Exception:
                    self.logger.exception("Failed to reload artifacts after config update")

    # ----------------------------
    # Control API: start/stop
    # ----------------------------
    def start_loop(self):
        """Start the daemon loop in a background thread."""
        if self._thread and self._thread.is_alive():
            self._enqueue_log("Daemon loop already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_loop, daemon=True)
        self._thread.start()
        self._enqueue_log("Daemon loop started.")

    def stop(self, wait: bool = True, timeout: float = 5.0):
        """Signal the loop to stop. Optionally wait for thread join."""
        self._stop_event.set()
        if self._thread and wait:
            self._thread.join(timeout=timeout)
        self._enqueue_log("Daemon stopped.")

    def run_loop(self):
        """Main loop: inference + trading (with multi-horizon support)."""
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                if self.multi_horizon_mode:
                    # Multi-horizon mode: generate fan predictions
                    predictions = self.iteration_once_multi_horizon()

                    # Extract native horizon for trading (optional)
                    native_h = self.model_meta.get("horizon", 10)
                    if native_h in predictions:
                        pred_data = predictions[native_h]
                        pred_log_ret = pred_data.get("log_return", 0)

                        # Trading decision (only on native horizon)
                        if self._should_trade(pred_log_ret):
                            # Note: For now, we skip trading in multi-horizon mode
                            # (or you can implement full trading logic here)
                            self._enqueue_log(
                                f"[Multi-horizon] Trading signal: {pred_log_ret:.4f} (not executing in MH mode)",
                                level=logging.INFO
                            )
                else:
                    # Standard single-horizon mode
                    self.iteration_once()

                backoff = 1.0  # reset on success
            except Exception as e:
                self._enqueue_log(f"Iteration failed: {e}")
                self.logger.exception("Iteration exception:")
                backoff = min(backoff * 2.0, 60.0)
            # Sleep but allow early exit
            slept = 0.0
            step = 0.1
            while slept < self.poll_interval:
                if self._stop_event.is_set():
                    break
                time.sleep(step)
                slept += step
            # optional backoff between iterations
            if backoff > 1.0:
                time.sleep(backoff)

    # ----------------------------
    # Model & scaler load/save helpers
    # ----------------------------
    def load_model_and_scaler(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None, meta_path: Optional[str] = None):
        """
        Load model (torch) and scaler (joblib) into memory.
        Handles various checkpoint layouts (state_dict, model object, nested dicts).
        Sets self.model, self.model_scaler, self.model_meta.

        This function is thread-safe via self._artifact_lock.
        """
        # Use the instance lock to prevent concurrent loads
        with self._artifact_lock:
            mp = Path(model_path) if model_path else Path(self.model_path)
            sp = Path(scaler_path) if scaler_path else Path(self.scaler_path)
            metap = Path(meta_path) if meta_path else Path(self.meta_path)

            self._enqueue_log(f"Attempting to load model from {mp}")
            # load meta if exists
            meta: Dict[str, Any] = {}
            if metap.exists():
                try:
                    meta = json.loads(metap.read_text(encoding="utf-8"))
                    self._enqueue_log(f"Loaded meta: {metap}")
                except Exception:
                    self._enqueue_log(f"Failed to read meta.json: {metap}")
                    self.logger.exception("meta read error")

            # guard: torch must exist to load state_dicts
            if torch is None:
                self._enqueue_log("PyTorch not available: cannot load model state_dict. Aborting model load.")
                return

            try:
                ckpt = torch.load(str(mp), map_location="cpu")
            except Exception as e:
                self._enqueue_log(f"Failed to torch.load model file {mp}: {e}")
                self.logger.exception("torch.load error")
                return

            state = ckpt
            ckpt_meta: Dict[str, Any] = {}
            if isinstance(ckpt, dict):
                # common containers
                if "state" in ckpt and isinstance(ckpt["state"], dict):
                    state = ckpt["state"]
                    ckpt_meta = ckpt.get("meta", {})
                elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                    state = ckpt["model_state_dict"]
                    ckpt_meta = ckpt.get("meta", {})
                elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                    state = ckpt["state_dict"]
                    ckpt_meta = ckpt.get("meta", {})
                else:
                    # maybe top-level state_dict or model object
                    if "meta" in ckpt and isinstance(ckpt["meta"], dict):
                        ckpt_meta = ckpt["meta"]

            # combine meta: file-provided ckpt_meta overridden by meta.json
            combined_meta = dict(ckpt_meta or {})
            combined_meta.update(meta or {})
            meta = combined_meta

            # Normalize keys (remove module. prefix from DataParallel)
            if isinstance(state, dict):
                normalized: Dict[str, Any] = {}
                for k, v in state.items():
                    nk = k[len("module.") :] if k.startswith("module.") else k
                    normalized[nk] = v
                state = normalized

            # Attempt to infer model architecture params
            inferred_input = None
            inferred_hidden = None
            inferred_layers = None
            if isinstance(state, dict):
                lstm_keys = [k for k in state.keys() if k.startswith("lstm.weight_ih_l")]
                if lstm_keys:
                    try:
                        nums = [int(k.split("l")[-1]) for k in lstm_keys if "lstm.weight_ih_l" in k]
                        inferred_layers = max(nums) + 1
                    except Exception:
                        inferred_layers = len(lstm_keys)
                if "lstm.weight_ih_l0" in state:
                    w = state["lstm.weight_ih_l0"]
                    shp = getattr(w, "shape", None)
                    if shp and len(shp) == 2:
                        inferred_input = int(shp[1])
                        inferred_hidden = int(shp[0] // 4)

            # Determine input_size and hidden from meta or inferred
            input_size = None
            if "input_size" in meta:
                try:
                    input_size = int(meta["input_size"])
                except Exception:
                    input_size = None
            if input_size is None and "feature_cols" in meta and isinstance(meta["feature_cols"], (list, tuple)):
                input_size = len(meta["feature_cols"])
            if input_size is None and inferred_input is not None:
                input_size = int(inferred_input)

            hidden = None
            if "hidden" in meta:
                try:
                    hidden = int(meta["hidden"])
                except Exception:
                    hidden = None
            if hidden is None and inferred_hidden is not None:
                hidden = int(inferred_hidden)
            if hidden is None:
                hidden = 64

            num_layers = None
            if "num_layers" in meta:
                try:
                    num_layers = int(meta["num_layers"])
                except Exception:
                    num_layers = None
            if num_layers is None and inferred_layers is not None:
                num_layers = int(inferred_layers)
            if num_layers is None:
                num_layers = 2

            # Build model if fibo.LSTM2Head is available
            if fibo is None or not hasattr(fibo, "LSTM2Head"):
                self._enqueue_log("fiboevo.LSTM2Head not available: cannot instantiate model class. Aborting model load.")
                return

            try:
                model = fibo.LSTM2Head(input_size=int(input_size or 1), hidden_size=int(hidden), num_layers=int(num_layers))
            except Exception as e:
                self._enqueue_log(f"Failed to instantiate LSTM2Head: {e}")
                self.logger.exception("Model instantiation error")
                return

            # Try loading state dict
            try:
                if isinstance(state, dict):
                    try:
                        model.load_state_dict(state)
                        self._enqueue_log("Loaded model state_dict (strict=True).")
                    except Exception:
                        try:
                            res = model.load_state_dict(state, strict=False)
                            self._enqueue_log("Loaded model state_dict (strict=False).")
                            missing = getattr(res, "missing_keys", None)
                            unexpected = getattr(res, "unexpected_keys", None)
                            if missing:
                                self._enqueue_log(f"Missing keys in state_dict: {missing}")
                            if unexpected:
                                self._enqueue_log(f"Unexpected keys in state_dict: {unexpected}")
                        except Exception:
                            self._enqueue_log("Failed to load state_dict into model.")
                            self.logger.exception("state_dict load failure")
                            # still continue; possibly checkpoint contains full model object
                            raise
                else:
                    # checkpoint not a dict: possibly serialized full model object
                    try:
                        model = state
                        self._enqueue_log("Checkpoint appears to be serialized model object.")
                    except Exception:
                        self._enqueue_log("Checkpoint format not recognized.")
                        return
            except Exception:
                self._enqueue_log("Model loading encountered errors; aborting.")
                return

            # Move to CPU by default (daemon inference likely CPU)
            try:
                device = torch.device("cpu")
                model.to(device)
                model.eval()
                self.device = device  # Store device for multi-horizon inference
                self._enqueue_log("Model prepared on CPU.")
            except Exception:
                self._enqueue_log("Warning: failed to move model to CPU (torch/device issue).")
                self.device = torch.device("cpu") if torch is not None else None

            # Load scaler if present
            scaler = None
            if sp.exists():
                if joblib is not None:
                    try:
                        scaler = joblib.load(str(sp))
                        self._enqueue_log(f"Loaded scaler from {sp}")
                    except Exception:
                        self._enqueue_log(f"Failed to load scaler {sp}")
                        self.logger.exception("scaler load error")
                else:
                    self._enqueue_log("joblib not installed: cannot load scaler.pkl")

            # assign only after full successful load
            self.model = model
            self.model_meta = meta or {}
            # Reset warning flag now that metadata is loaded
            self._logged_no_features = False

            # Load scaler with feature validation (prefer new API)
            if sp.exists():
                try:
                    if fibo and hasattr(fibo, 'load_scaler'):
                        # Use new API with feature validation
                        self.model_scaler = fibo.load_scaler(
                            sp,
                            feature_cols=meta.get("feature_cols")
                        )
                        self._enqueue_log(f"Loaded scaler with feature validation ({len(meta.get('feature_cols', []))} features)")
                    else:
                        # Fallback to joblib
                        scaler = joblib.load(str(sp))
                        if fibo and hasattr(fibo, 'ensure_scaler_feature_names'):
                            scaler = fibo.ensure_scaler_feature_names(scaler, meta)
                        self.model_scaler = scaler
                        self._enqueue_log(f"Loaded scaler (fallback method)")
                except Exception as e:
                    self._enqueue_log(f"Scaler load failed: {e}", level=logging.WARNING)
                    self.model_scaler = None
            else:
                self.model_scaler = None

    # ----------------------------
    # Core iteration: load data, features, predict, act
    # ----------------------------
    def iteration_once(self):
        """
        Single iteration:
         - load recent rows from sqlite
         - compute features (fibo.add_technical_features preferred)
         - build sequences
         - scale (if scaler)
         - predict using model (if present)
         - decide and act (paper or live)
        """
        # 1) load recent data for symbol/timeframe
        df = self._load_recent_rows(limit=1000)
        if df is None or df.empty:
            self._enqueue_log("No data available from sqlite.")
            return

        # 2) compute features
        if fibo is None or not hasattr(fibo, "add_technical_features"):
            self._enqueue_log("fiboevo.add_technical_features not available; skipping features/prediction.")
            return

        try:
            close = df["close"].astype(float).values
            high = df["high"].astype(float).values if "high" in df.columns else None
            low = df["low"].astype(float).values if "low" in df.columns else None
            vol = df["volume"].astype(float).values if "volume" in df.columns else None

            # Use feature registry system (v1 or v2)
            if FEATURE_REGISTRY is not None:
                feats = FEATURE_REGISTRY.compute_features(
                    close, high=high, low=low, volume=vol,
                    system_name=self.feature_system,
                    dropna_after=False  # Let daemon handle dropna later (line 720)
                )
            else:
                # Fallback to v1 if registry not available
                feats = fibo.add_technical_features(close, high=high, low=low, volume=vol, dropna_after=False)

            if not isinstance(feats, pd.DataFrame):
                feats = pd.DataFrame(np.asarray(feats))
            # attach relevant columns if missing
            for col in ("timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe"):
                if col in df.columns and col not in feats.columns:
                    feats[col] = df[col].values
            feats = feats.dropna().reset_index(drop=True)
        except Exception as e:
            self._enqueue_log(f"Feature creation failed: {e}")
            self.logger.exception("Feature error")
            return

        # 3) detect feature_cols (prefer meta)
        seq_len = int(self.seq_len)
        horizon = int(self.model_meta.get("horizon", 1) if self.model_meta else 1)

        # Preferred: use meta['feature_cols'] if present
        feature_cols = None
        if self.model_meta and isinstance(self.model_meta.get("feature_cols"), (list, tuple)):
            # keep only those still present in feats
            feature_cols = [c for c in self.model_meta["feature_cols"] if c in feats.columns]
            if len(feature_cols) != len(self.model_meta.get("feature_cols")):
                missing = [c for c in self.model_meta.get("feature_cols") if c not in feats.columns]
                self._enqueue_log(f"Warning: some feature_cols from model.meta are missing in current feats: {missing}")
                # fallback behaviour: continue with available ones (or choose to abort)
                # Option: abort if too many missing
                if len(feature_cols) < max(1, int(0.5 * len(self.model_meta.get("feature_cols")))):
                    self._enqueue_log("Too many feature_cols missing compared to model.meta -> aborting iteration.")
                    return
        else:
            # fallback: auto detect numeric columns excluding these
            exclude = {"timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "exchange"}
            numeric_cols = [c for c in feats.columns if c not in exclude and pd.api.types.is_numeric_dtype(feats[c])]
            feature_cols = numeric_cols

        if not feature_cols:
            self._enqueue_log("No feature columns available after detection. Aborting iteration.")
            return

        # 4) Prepare input using fiboevo.prepare_input_for_model (preferred)
        if torch is None or self.model is None:
            self._enqueue_log("Model or torch unavailable; skipping prediction.")
            return

        seq_len = int(self.seq_len)

        # Check we have enough data
        if len(feats) < seq_len:
            self._enqueue_log(f"Not enough data for prediction: {len(feats)} < {seq_len}")
            return

        try:
            # Use new API if available
            if hasattr(fibo, "prepare_input_for_model"):
                # Take last seq_len rows
                seq_df = feats.iloc[-seq_len:].reset_index(drop=True)

                # Validate we have required features
                missing = [c for c in feature_cols if c not in seq_df.columns]
                if missing:
                    self._enqueue_log(f"Missing features for prediction: {missing}", level=logging.ERROR)
                    return

                # Prepare input (handles scaling, validation, tensor creation)
                xb = fibo.prepare_input_for_model(
                    seq_df,
                    feature_cols,
                    seq_len,
                    scaler=self.model_scaler,
                    method="per_row"
                )

                self._enqueue_log(f"Prepared input tensor: shape={xb.shape}")

            else:
                # Fallback: manual sequence building and scaling
                self._enqueue_log("Using fallback sequence building (prepare_input_for_model not available)")
                horizon = int(self.model_meta.get("horizon", 1) if self.model_meta else 1)

                if hasattr(fibo, "create_sequences_from_df"):
                    X, y_ret, y_vol = fibo.create_sequences_from_df(feats, feature_cols, seq_len=seq_len, horizon=horizon)
                    X = np.asarray(X)
                else:
                    X, y = self._build_sequences_internal(feats, feature_cols, seq_len, horizon, dtype=np.float32)

                if X is None or X.shape[0] == 0:
                    self._enqueue_log("No sequences produced for prediction.")
                    return

                # Scale manually
                Xp = X
                if self.model_scaler is not None:
                    try:
                        flat = X.reshape(-1, X.shape[2])
                        flat_s = self.model_scaler.transform(flat)
                        Xp = flat_s.reshape(X.shape)
                    except Exception as e:
                        self._enqueue_log(f"Scaler transform failed: {e}")
                        Xp = X

                # Convert to tensor
                import torch as _torch
                xb = _torch.from_numpy(Xp[-1:]).float()

        except Exception as e:
            self._enqueue_log(f"Input preparation failed: {e}", level=logging.ERROR)
            self.logger.exception("prepare_input_for_model error")
            return

        # 5) Predict
        try:
            import torch as _torch
            # Ensure model on CPU and eval
            with self._artifact_lock:
                model_ref = self.model

            model_ref.eval()
            xb = xb.to("cpu")  # Ensure on CPU

            with _torch.no_grad():
                out = model_ref(xb)
                # model returns (ret, vol) tuple in fibo.LSTM2Head convention
                if isinstance(out, tuple) or isinstance(out, list):
                    pred = float(out[0].cpu().numpy().ravel()[0])
                else:
                    pred = float(out.cpu().numpy().ravel()[0])

            self._enqueue_log(f"Prediction computed: {pred:.6f}")

        except Exception as e:
            self._enqueue_log(f"Prediction error: {e}", level=logging.ERROR)
            self.logger.exception("Prediction exception")
            return

        # 7) decision and action
        try:
            if self._should_trade(pred):
                self._execute_trade(pred=pred, df=df, feats=feats, feature_cols=feature_cols)
            else:
                self._enqueue_log(f"No trade (signal {pred:.6f} within threshold).")
        except Exception as e:
            self._enqueue_log(f"Action failed: {e}")
            self.logger.exception("Action exception")

    def iteration_once_multi_horizon(self) -> Dict[int, Dict]:
        """
        Extended version of iteration_once() that generates multi-horizon predictions.

        Same data loading and feature engineering as iteration_once(),
        but uses multi_horizon_fan_inference for predictions.

        Returns:
            Dict of predictions keyed by horizon (or empty dict if error)
        """
        try:
            # 1. Load recent data
            df = self._load_recent_rows(limit=1000)
            if df is None or df.empty:
                self._enqueue_log("No data available for multi-horizon inference", level=logging.DEBUG)
                return {}

            # 2. Compute features
            if fibo is None or not hasattr(fibo, "add_technical_features"):
                self._enqueue_log("fiboevo not available for multi-horizon features", level=logging.DEBUG)
                return {}

            try:
                close = df["close"].astype(float).values
                high = df["high"].astype(float).values if "high" in df.columns else None
                low = df["low"].astype(float).values if "low" in df.columns else None
                vol = df["volume"].astype(float).values if "volume" in df.columns else None

                # Use feature registry system (v1 or v2)
                if FEATURE_REGISTRY is not None:
                    df_feats = FEATURE_REGISTRY.compute_features(
                        close, high=high, low=low, volume=vol,
                        system_name=self.feature_system,
                        dropna_after=True
                    )
                else:
                    # Fallback to v1 if registry not available
                    df_feats = fibo.add_technical_features(close, high=high, low=low, volume=vol, dropna_after=True)

                # Attach metadata columns
                for col in ("timestamp", "ts", "close"):
                    if col in df.columns and col not in df_feats.columns:
                        df_feats[col] = df[col].values[-len(df_feats):]

                df_feats = df_feats.reset_index(drop=True)
            except Exception as e:
                self._enqueue_log(f"Multi-horizon feature creation failed: {e}", level=logging.ERROR)
                return {}

            if len(df_feats) < self.seq_len:
                self._enqueue_log(f"Insufficient data after features: {len(df_feats)} < {self.seq_len}", level=logging.DEBUG)
                return {}

            # 3. Get feature columns
            feature_cols = self.model_meta.get("feature_cols", [])
            if not feature_cols:
                # Only log once to avoid spam (waiting for model/metadata to load)
                if not getattr(self, '_logged_no_features', False):
                    self._enqueue_log("Waiting for model metadata to load...", level=logging.DEBUG)
                    self._logged_no_features = True
                return {}

            # 4. Check if multi_horizon_fan_inference is available
            try:
                from multi_horizon_fan_inference import predict_single_point_multi_horizon
            except ImportError:
                self._enqueue_log("multi_horizon_fan_inference module not found", level=logging.ERROR)
                return {}

            # 5. Multi-horizon prediction
            with self._artifact_lock:
                if self.model is None or self.model_scaler is None:
                    self._enqueue_log("Model or scaler not loaded", level=logging.WARNING)
                    return {}

                try:
                    predictions = predict_single_point_multi_horizon(
                        df=df_feats,
                        model=self.model,
                        meta=self.model_meta,
                        scaler=self.model_scaler,
                        device=self.device if self.device is not None else torch.device("cpu"),
                        horizons=self.multi_horizon_horizons,
                        method="scaling"
                    )
                except Exception as e:
                    self._enqueue_log(f"Multi-horizon prediction failed: {e}", level=logging.ERROR)
                    return {}

            # 6. Push to queue for GUI
            try:
                self.predictions_queue.put_nowait(predictions)
            except queue.Full:
                # Queue full, discard oldest
                try:
                    self.predictions_queue.get_nowait()
                    self.predictions_queue.put_nowait(predictions)
                except Exception:
                    pass

            return predictions

        except Exception as e:
            self._enqueue_log(f"Multi-horizon inference error: {e}", level=logging.ERROR)
            import traceback
            self._enqueue_log(traceback.format_exc(), level=logging.DEBUG)
            return {}

    # ----------------------------
    # Trading logic helpers
    # ----------------------------
    def _should_trade(self, pred: float) -> bool:
        """
        Simple threshold-based decision. Make configurable as needed.
        """
        thr = float(self.model_meta.get("trade_threshold", 0.0005))
        return abs(pred) > thr

    def _execute_trade(self, pred: float, df: pd.DataFrame, feats: pd.DataFrame, feature_cols: List[str]):
        """
        Execute or simulate trade.
        In paper mode: write ledger entry.
        In live mode: attempt ccxt order and write ledger with order response.
        """
        ts = time.time()
        price = float(df["close"].astype(float).iloc[-1])
        side = "buy" if pred > 0 else "sell"
        # Position sizing: implement your logic here; placeholder uses fixed fraction
        equity = float(self.model_meta.get("equity", 10000.0))
        pos_pct = float(self.model_meta.get("pos_pct", 0.01))
        amount_usd = equity * pos_pct
        # If symbol is like 'BTCUSDT' price denom assumed USD-like; convert if exchange requires
        amount = max(1e-8, amount_usd / max(1e-9, price))

        entry: Dict[str, Any] = {
            "time": ts,
            "symbol": self.symbol,
            "side": side,
            "pred": pred,
            "price": price,
            "amount": amount,
            "paper": bool(self.paper),
        }

        if self.paper or self.exchange is None:
            # Simulate order
            entry["status"] = "simulated"
            self._write_ledger(entry)
            self._enqueue_log(f"Paper trade simulated: {side} {amount:.8f} @ {price:.2f}")
            return

        # Live order with ccxt
        try:
            # exchange order creation: adapt market vs limit params as required by your exchange
            symbol = self.symbol
            order_symbol = symbol
            # Some exchanges use 'BTC/USDT'; try to adapt if markets available
            try:
                if hasattr(self.exchange, "markets") and symbol not in self.exchange.markets:
                    # try to find matching market by simple heuristics
                    alt = symbol
                    if "/" not in symbol and "USDT" in symbol:
                        alt = symbol.replace("USDT", "/USDT")
                    if alt in self.exchange.markets:
                        order_symbol = alt
            except Exception:
                pass

            order = None
            try:
                if side == "buy" and hasattr(self.exchange, "create_market_buy_order"):
                    order = self.exchange.create_market_buy_order(order_symbol, amount)
                elif side == "sell" and hasattr(self.exchange, "create_market_sell_order"):
                    order = self.exchange.create_market_sell_order(order_symbol, amount)
                else:
                    # generic
                    order = self.exchange.create_order(order_symbol, "market", side, amount)
            except Exception:
                # fallback generic
                order = self.exchange.create_order(order_symbol, "market", side, amount)
            entry["status"] = "placed"
            entry["order"] = str(order)
            self._write_ledger(entry)
            self._enqueue_log(f"Live order placed: {side} {amount:.8f} {order_symbol} (order info logged).")
        except Exception as e:
            self._enqueue_log(f"Live order failed: {e}")
            self.logger.exception("Order placement error")
            entry["status"] = "error"
            entry["error"] = str(e)
            self._write_ledger(entry)

    def _write_ledger(self, entry: Dict[str, Any]):
        """Append a CSV line to artifacts/ledger.csv. Basic thread-safe append."""
        try:
            ledger = self.ledger_path
            exists = ledger.exists()
            # Ensure parent dir exists
            ledger.parent.mkdir(parents=True, exist_ok=True)
            # If header absent, write header then row
            if not exists:
                headers = list(entry.keys())
                with ledger.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(headers)
                    writer.writerow([entry.get(h, "") for h in headers])
            else:
                # append, preserving header order by reading header
                with ledger.open("r", encoding="utf-8") as fh:
                    reader = csv.reader(fh)
                    try:
                        existing_header = next(reader)
                    except StopIteration:
                        existing_header = list(entry.keys())
                # write row based on existing_header order (fill missing with "")
                with ledger.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    row = [entry.get(h, "") for h in existing_header]
                    writer.writerow(row)
        except Exception:
            self.logger.exception("Failed to write ledger")

    # ----------------------------
    # Data helpers
    # ----------------------------
    def _load_recent_rows(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Load most recent rows for symbol/timeframe from sqlite table."""
        if not self.sqlite_path.exists():
            self._enqueue_log(f"SQLite path not found: {self.sqlite_path}")
            return None
        try:
            con = sqlite3.connect(str(self.sqlite_path), timeout=5.0)
            q = f"SELECT * FROM {self.sqlite_table} WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT ?"
            df = pd.read_sql_query(q, con, params=[self.symbol, self.timeframe, int(limit)])
            con.close()
            if df.empty:
                return df
            # ensure chronological ascending order
            if "ts" in df.columns and "timestamp" not in df.columns:
                try:
                    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
                except Exception:
                    # leave ts as-is
                    pass
            sort_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else df.columns[0])
            df = df.sort_values(sort_col).reset_index(drop=True)
            return df
        except Exception as e:
            self._enqueue_log(f"SQLite read failed: {e}")
            self.logger.exception("SQLite read")
            return None

    def _build_sequences_internal(self, df_feats: pd.DataFrame, feature_cols: List[str], seq_len: int, horizon: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal sequence builder compatible with GUI's implementation.
        Returns X (N, seq_len, F), y (N,)
        """
        df = df_feats.copy().reset_index(drop=True)
        present = [c for c in feature_cols if c in df.columns]
        if len(present) == 0:
            return np.zeros((0, seq_len, 0), dtype=dtype), np.zeros((0,), dtype=dtype)
        # convert datetimes to numeric
        for c in present:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c]).astype("int64") / 1e9
        # ensure close exists
        if "close" not in df.columns:
            raise RuntimeError("close column missing for sequence building")
        df["__logc"] = np.log(df["close"].replace(0, np.nan))
        df = df.dropna().reset_index(drop=True)
        M = len(df)
        min_rows = seq_len + horizon
        if M < min_rows:
            return np.zeros((0, seq_len, len(present)), dtype=dtype), np.zeros((0,), dtype=dtype)
        N = M - seq_len - horizon + 1
        F = len(present)
        X = np.zeros((N, seq_len, F), dtype=dtype)
        y = np.zeros((N,), dtype=dtype)
        mat = df[present].astype(dtype).values
        logc = df["__logc"].values
        idx = 0
        for i in range(N):
            X[idx] = mat[i:i+seq_len, :]
            y[idx] = logc[i+seq_len+horizon-1] - logc[i+seq_len-1]
            idx += 1
        return X, y

    def close(self, wait: bool = True, timeout: float = 5.0):
        """
        Graceful shutdown helper for higher-level apps.
        Signals loop stop, joins thread, and attempts to free model resources.
        """
        self._enqueue_log("TradingDaemon.close() called. Stopping daemon...", level=logging.INFO)
        try:
            self._stop_event.set()
            if self._thread and self._thread.is_alive() and wait:
                self._thread.join(timeout=timeout)
        except Exception:
            self.logger.exception("Error while joining daemon thread")

        # Try to delete model to free memory (best-effort)
        try:
            if getattr(self, "model", None) is not None:
                # remove reference; torch will free on GC (if GPU used, user should call torch.cuda.empty_cache externally)
                self.model = None
                # if torch available, clear cached memory (best-effort)
                try:
                    import torch as _torch
                    if hasattr(_torch, "cuda") and _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            self.logger.exception("Error while cleaning model resources")

        self._enqueue_log("TradingDaemon closed.", level=logging.INFO)



# ----------------------------
# If run as script -> small smoke test
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TradingDaemon smoke check (no live trading).")
    parser.add_argument("--sqlite", default="data_manager/exports/marketdata_base.db")
    parser.add_argument("--table", default="ohlcv")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="30m")
    parser.add_argument("--paper", action="store_true", help="run in paper (simulated) mode")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args()

    td = TradingDaemon(
        sqlite_path=args.sqlite,
        sqlite_table=args.table,
        symbol=args.symbol,
        timeframe=args.timeframe,
        paper=True,
        seq_len=args.seq_len,
        poll_interval=args.poll,
        auto_load_artifacts=True,
    )

    td._enqueue_log("Starting smoke test: loading model & scaler if present...")
    # load_model_and_scaler already intentó autoload en constructor si auto_load_artifacts=True
    td._enqueue_log("Running single iteration (smoke) ...")
    try:
        td.iteration_once()
        td._enqueue_log("Smoke iteration complete.")
    except Exception as e:
        td._enqueue_log(f"Smoke iteration raised: {e}")
        traceback.print_exc()
