# Status Tab - Integración de Dashboard Multi-Horizonte

**Fecha**: 2025-10-30
**Archivos creados**:
- ✅ `multi_horizon_fan_inference.py` - Función `predict_single_point_multi_horizon()`
- ✅ `dashboard_visualizations_simple.py` - Función `plot_prediction_fan_live_simple()`
- ✅ `TRADEAPP_STRUCTURE_ANALYSIS.md` - Análisis completo de la estructura

**Archivos a modificar**:
- `trading_daemon.py` - Añadir capacidades multi-horizonte
- `TradeApp.py` - Extender Status tab con dashboard

---

## Resumen de la Integración

El sistema integrará el dashboard de predicciones multi-horizonte **directamente en el Status tab** (no como tab separado), aprovechando:
1. **WebSocket** ya existente → datos en vivo en SQLite
2. **TradingDaemon** ya existente → inferencias periódicas
3. Solo necesitamos **extender** la lógica existente

**Flujo**:
```
WebSocket → SQLite → Daemon (cada 5s) → Predicciones multi-horizonte → Queue → GUI actualiza canvas
```

---

## Modificación 1: Trading Daemon (trading_daemon.py)

### 1.1 Añadir imports al inicio del archivo

**Ubicación**: Después de los imports existentes (línea ~60)

```python
# Añadir este import
try:
    from multi_horizon_fan_inference import predict_single_point_multi_horizon
except ImportError:
    predict_single_point_multi_horizon = None
```

### 1.2 Modificar `__init__` method

**Ubicación**: Dentro del método `__init__` de TradingDaemon (línea ~110)

**Buscar**:
```python
def __init__(
    self,
    sqlite_path: str,
    sqlite_table: str,
    symbol: str,
    timeframe: str,
    model_path: Optional[str] = None,
    ...
):
    # ... código existente ...
```

**Añadir al final del `__init__` (antes del último comentario o return)**:
```python
    # Multi-horizon predictions (NEW for dashboard integration)
    import queue as _queue_mod
    self.predictions_queue = _queue_mod.Queue(maxsize=10)
    self.multi_horizon_mode = False  # Toggle from GUI
    self.multi_horizon_horizons = [1, 3, 5, 10, 15, 20, 30]
```

### 1.3 Añadir método `iteration_once_multi_horizon`

**Ubicación**: Después del método `iteration_once` (línea ~801)

```python
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
            self._enqueue_log("No data available for multi-horizon inference", level=logging.WARNING)
            return {}

        # 2. Compute features
        if fibo is None:
            self._enqueue_log("fiboevo not available for feature computation", level=logging.ERROR)
            return {}

        df_feats = fibo.add_technical_features(
            close=df["close"].values,
            high=df["high"].values if "high" in df.columns else None,
            low=df["low"].values if "low" in df.columns else None,
            volume=df["volume"].values if "volume" in df.columns else None,
            dropna_after=True
        )

        if len(df_feats) < self.seq_len:
            self._enqueue_log(f"Insufficient data after features: {len(df_feats)} < {self.seq_len}", level=logging.WARNING)
            return {}

        # 3. Get feature columns
        feature_cols = self.model_meta.get("feature_cols", [])
        if not feature_cols:
            self._enqueue_log("No feature_cols in model_meta", level=logging.ERROR)
            return {}

        # 4. Multi-horizon prediction
        if predict_single_point_multi_horizon is None:
            self._enqueue_log("predict_single_point_multi_horizon not available", level=logging.ERROR)
            return {}

        with self._artifact_lock:
            predictions = predict_single_point_multi_horizon(
                df=df_feats,
                model=self.model,
                meta=self.model_meta,
                scaler=self.model_scaler,
                device=self.device if hasattr(self, 'device') else torch.device("cpu"),
                horizons=self.multi_horizon_horizons,
                method="scaling"
            )

        # 5. Push to queue for GUI
        try:
            self.predictions_queue.put_nowait(predictions)
        except Exception:
            # Queue full, discard oldest
            try:
                self.predictions_queue.get_nowait()
                self.predictions_queue.put_nowait(predictions)
            except:
                pass

        return predictions

    except Exception as e:
        self._enqueue_log(f"Multi-horizon inference error: {e}", level=logging.ERROR)
        import traceback
        self._enqueue_log(traceback.format_exc(), level=logging.DEBUG)
        return {}
```

### 1.4 Modificar método `run_loop`

**Ubicación**: Método `run_loop` (línea ~383)

**Reemplazar** el contenido del `while` loop con:

```python
def run_loop(self):
    """Main loop: inference + trading (with multi-horizon support)."""
    self._enqueue_log("Daemon loop started")

    while not self._stop_flag.is_set():
        try:
            if self.multi_horizon_mode:
                # Multi-horizon mode: generate fan predictions
                predictions = self.iteration_once_multi_horizon()

                # Extract native horizon for trading (if needed)
                native_h = self.model_meta.get("horizon", 10)
                if native_h in predictions:
                    pred_data = predictions[native_h]
                    pred_log_ret = pred_data.get("log_return", 0)

                    # Trading decision (only on native horizon)
                    if self._should_trade(pred_log_ret):
                        # Note: Full trading execution would need complete context
                        # For now, just log the signal
                        self._enqueue_log(
                            f"[Multi-horizon] Trading signal: {pred_log_ret:.4f} (not executing in multi-horizon mode)",
                            level=logging.INFO
                        )
            else:
                # Standard single-horizon mode
                self.iteration_once()

        except Exception as e:
            self._enqueue_log(f"Loop iteration error: {e}", level=logging.ERROR)
            import traceback
            self._enqueue_log(traceback.format_exc(), level=logging.DEBUG)

        # Sleep until next iteration
        self._stop_flag.wait(self.poll_interval)

    self._enqueue_log("Daemon loop stopped")
```

---

## Modificación 2: TradeApp Status Tab (TradeApp.py)

### 2.1 Añadir imports al inicio

**Ubicación**: Después de imports existentes (línea ~100)

```python
# Multi-horizon dashboard imports (NEW)
try:
    from dashboard_visualizations_simple import plot_prediction_fan_live_simple
except ImportError:
    plot_prediction_fan_live_simple = None
```

### 2.2 Añadir variables de instancia en `__init__`

**Ubicación**: Método `__init__` de TradingAppExtended (línea ~300)

**Añadir después de las variables de Status tab**:

```python
    # Multi-horizon dashboard state (NEW)
    self.multi_horizon_enabled_var = BooleanVar(value=False)
    self.last_pred_update_var = StringVar(value="Never")
    self.pred_fan_fig = None
    self.pred_fan_ax = None
    self.pred_fan_canvas = None
    self.pred_summary_tree = None
```

### 2.3 Extender método `_build_status_tab`

**Ubicación**: Final del método `_build_status_tab` (línea ~2150)

**Añadir al final del método, ANTES del cierre**:

```python
    # ========== MULTI-HORIZON PREDICTION DASHBOARD (NEW) ==========

    # Section header
    Label(tab, text="Live Multi-Horizon Predictions",
          font=("Arial", 12, "bold")).pack(anchor="w", padx=6, pady=(15,4))

    # Control buttons
    frm_pred_ctrl = Frame(tab)
    frm_pred_ctrl.pack(fill=X, padx=6, pady=4)

    chk_mh = Checkbutton(frm_pred_ctrl, text="Enable Multi-Horizon Mode",
                         variable=self.multi_horizon_enabled_var,
                         command=self._toggle_multi_horizon_mode)
    chk_mh.pack(side=LEFT, padx=4)

    Button(frm_pred_ctrl, text="Refresh Predictions",
           command=self._manual_refresh_predictions).pack(side=LEFT, padx=4)

    Label(frm_pred_ctrl, text="Last update:").pack(side=LEFT, padx=(10,2))
    Label(frm_pred_ctrl, textvariable=self.last_pred_update_var).pack(side=LEFT)

    # Prediction fan canvas (matplotlib)
    frm_canvas = Frame(tab, relief=SUNKEN, borderwidth=1)
    frm_canvas.pack(fill=BOTH, expand=True, padx=6, pady=6)

    self.pred_fan_fig = Figure(figsize=(14, 5), dpi=100, facecolor='#f0f0f0')
    self.pred_fan_ax = self.pred_fan_fig.add_subplot(111)
    self.pred_fan_canvas = FigureCanvasTkAgg(self.pred_fan_fig, master=frm_canvas)
    self.pred_fan_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    # Initial placeholder
    self.pred_fan_ax.text(0.5, 0.5,
                         "Enable Multi-Horizon Mode and start daemon to see predictions",
                         ha="center", va="center", fontsize=11, color="gray")
    self.pred_fan_ax.set_xlim(0, 1)
    self.pred_fan_ax.set_ylim(0, 1)
    self.pred_fan_ax.axis("off")
    self.pred_fan_canvas.draw()

    # Predictions summary table
    frm_pred_table = Frame(tab)
    frm_pred_table.pack(fill=X, padx=6, pady=(0,6))

    Label(frm_pred_table, text="Predictions Summary",
          font=("Arial", 10, "bold")).pack(anchor="w", pady=2)

    cols = ("Horizon", "Target Time", "Predicted Price", "Change ($)", "Change (%)", "95% CI", "Signal")
    self.pred_summary_tree = ttk.Treeview(frm_pred_table, columns=cols, height=7, show="headings")

    for col in cols:
        self.pred_summary_tree.heading(col, text=col)
        width = 120 if col != "95% CI" else 180
        self.pred_summary_tree.column(col, width=width, anchor=CENTER if col == "Signal" else W)

    self.pred_summary_tree.pack(fill=X)

    # Scrollbar for table
    sb = ttk.Scrollbar(frm_pred_table, orient=VERTICAL, command=self.pred_summary_tree.yview)
    sb.pack(side=RIGHT, fill=Y)
    self.pred_summary_tree.configure(yscrollcommand=sb.set)
```

### 2.4 Añadir métodos de soporte

**Ubicación**: Al final de la clase TradingAppExtended (línea ~3400)

```python
# ========== MULTI-HORIZON DASHBOARD METHODS (NEW) ==========

def _toggle_multi_horizon_mode(self):
    """Toggle multi-horizon mode in daemon."""
    enabled = self.multi_horizon_enabled_var.get()

    if self.daemon:
        self.daemon.multi_horizon_mode = enabled
        self._enqueue_log(f"Multi-horizon mode: {'ON' if enabled else 'OFF'}")

        if enabled:
            # Start predictions queue polling
            self.root.after(100, self._poll_predictions_queue)
    else:
        self._enqueue_log("Start daemon first to enable multi-horizon mode")
        self.multi_horizon_enabled_var.set(False)

def _manual_refresh_predictions(self):
    """Manually trigger prediction update (useful for debugging)."""
    if self.daemon and self.daemon.multi_horizon_mode:
        def _bg():
            preds = self.daemon.iteration_once_multi_horizon()
            if preds:
                self._enqueue_log(f"Manual refresh: got {len(preds)} horizons")

        threading.Thread(target=_bg, daemon=True).start()
    else:
        self._enqueue_log("Enable multi-horizon mode first")

def _poll_predictions_queue(self):
    """
    Poll daemon's predictions_queue and update UI.
    Called periodically when multi-horizon mode is enabled.
    """
    if not self.multi_horizon_enabled_var.get():
        return  # Stopped

    if self.daemon is None or not hasattr(self.daemon, "predictions_queue"):
        # Daemon not running, retry later
        self.root.after(5000, self._poll_predictions_queue)
        return

    try:
        # Try to get predictions (non-blocking)
        predictions = self.daemon.predictions_queue.get_nowait()

        if predictions:
            # Get historical data for plot
            if self.daemon:
                df_history = self.daemon._load_recent_rows(limit=200)
            else:
                df_history = self.df_loaded.tail(200) if self.df_loaded is not None else None

            # Update display
            self._update_prediction_fan_display(predictions, df_history)

            # Update timestamp
            from datetime import datetime
            self.last_pred_update_var.set(datetime.now().strftime("%H:%M:%S"))

    except queue.Empty:
        pass  # No new predictions
    except Exception as e:
        self._enqueue_log(f"Predictions poll error: {e}")

    # Schedule next poll
    interval_ms = int(self.refresh_interval_var.get() * 1000)
    self.root.after(interval_ms, self._poll_predictions_queue)

def _update_prediction_fan_display(self, predictions: Dict[int, Dict], df_history: pd.DataFrame):
    """
    Update prediction fan canvas and summary table.

    Args:
        predictions: {horizon: {price, log_return, volatility, ci_lower_95, ci_upper_95, ...}}
        df_history: Recent OHLCV data for historical context
    """
    try:
        # Clear axis
        ax = self.pred_fan_ax
        ax.clear()

        if df_history is None or df_history.empty:
            ax.text(0.5, 0.5, "No historical data available",
                   ha="center", va="center", transform=ax.transAxes)
            self.pred_fan_canvas.draw()
            return

        # Check if visualization function available
        if plot_prediction_fan_live_simple is None:
            ax.text(0.5, 0.5, "Visualization module not available\nInstall matplotlib",
                   ha="center", va="center", transform=ax.transAxes, color="red")
            self.pred_fan_canvas.draw()
            return

        # Plot
        plot_prediction_fan_live_simple(
            ax=ax,
            df_history=df_history,
            predictions_dict=predictions,
            show_confidence=True,
            colormap="viridis",
            n_history=100
        )

        self.pred_fan_canvas.draw()

        # Update summary table
        self._update_prediction_summary_table(predictions)

    except Exception as e:
        self._enqueue_log(f"Failed to update prediction display: {e}")
        import traceback
        self._enqueue_log(traceback.format_exc(), level=logging.DEBUG)

def _update_prediction_summary_table(self, predictions: Dict[int, Dict]):
    """Update predictions summary table with latest data."""
    try:
        from datetime import datetime, timedelta

        # Clear existing rows
        for item in self.pred_summary_tree.get_children():
            self.pred_summary_tree.delete(item)

        # Populate with predictions
        for h in sorted(predictions.keys()):
            pred = predictions[h]

            # Calculate target time (assuming 1h timeframe)
            target_time = (datetime.now() + timedelta(hours=h)).strftime("%m-%d %H:%M")

            price = f"${pred.get('price', 0):.2f}"
            change_usd = pred.get('change_usd', 0)
            change_pct = pred.get('change_pct', 0)
            ci = f"[${pred.get('ci_lower_95', 0):.0f}, ${pred.get('ci_upper_95', 0):.0f}]"

            # Signal
            if change_pct > 0.1:
                signal = "↑"
            elif change_pct < -0.1:
                signal = "↓"
            else:
                signal = "→"

            # Color coding
            tag = "positive" if change_pct > 0 else "negative" if change_pct < 0 else "neutral"

            self.pred_summary_tree.insert("", "end", values=(
                f"{h}h",
                target_time,
                price,
                f"${change_usd:+.2f}",
                f"{change_pct:+.2f}%",
                ci,
                signal
            ), tags=(tag,))

        # Configure tag colors
        self.pred_summary_tree.tag_configure("positive", foreground="darkgreen")
        self.pred_summary_tree.tag_configure("negative", foreground="darkred")
        self.pred_summary_tree.tag_configure("neutral", foreground="gray")

    except Exception as e:
        self._enqueue_log(f"Failed to update prediction table: {e}")
```

---

## Pasos de Integración

1. **Hacer backup de archivos**:
   ```bash
   copy trading_daemon.py trading_daemon.py.backup
   copy TradeApp.py TradeApp.py.backup
   ```

2. **Aplicar modificaciones a `trading_daemon.py`**:
   - Añadir import de `predict_single_point_multi_horizon`
   - Modificar `__init__` para añadir `predictions_queue` y flags
   - Añadir método `iteration_once_multi_horizon`
   - Modificar método `run_loop`

3. **Aplicar modificaciones a `TradeApp.py`**:
   - Añadir import de `plot_prediction_fan_live_simple`
   - Añadir variables de instancia en `__init__`
   - Extender `_build_status_tab` con canvas y tabla
   - Añadir métodos de soporte al final de la clase

4. **Probar la integración**:
   ```bash
   python TradeApp.py
   ```

5. **Verificar funcionamiento**:
   - Navegar a Status tab
   - Click "Start Daemon"
   - Check "Enable Multi-Horizon Mode"
   - Observar que el canvas se actualiza cada 5 segundos
   - Verificar que la tabla muestra predicciones

---

## Troubleshooting

### Error: "predict_single_point_multi_horizon not available"
**Solución**: Verificar que `multi_horizon_fan_inference.py` está en el directorio del proyecto

### Error: "matplotlib not available"
**Solución**:
```bash
pip install matplotlib
```

### Canvas no se actualiza
**Solución**:
- Verificar que daemon está corriendo (`self.daemon is not None`)
- Verificar que multi-horizon mode está habilitado
- Revisar logs para errores

### Predictions queue siempre vacío
**Solución**:
- Verificar que `daemon.multi_horizon_mode = True`
- Verificar que el modelo está cargado (`daemon.model is not None`)
- Verificar que hay datos en SQLite

---

## Archivos Finales

Después de la integración, tendrás:

```
BitCorn_Farmer/
├── trading_daemon.py (modificado)
├── TradeApp.py (modificado)
├── multi_horizon_fan_inference.py (nuevo)
├── dashboard_visualizations_simple.py (nuevo)
├── TRADEAPP_STRUCTURE_ANALYSIS.md (nuevo)
└── STATUS_TAB_INTEGRATION_INSTRUCTIONS.md (este archivo)
```

---

## Próximos Pasos Opcionales

1. **Añadir más controles**:
   - Slider para ajustar horizontes
   - Toggle para confidence bands
   - Selector de colormap

2. **Mejorar visualización**:
   - Añadir probabilidad layers
   - Soporte para tema oscuro
   - Indicadores de rendimiento del modelo

3. **Persistencia**:
   - Guardar predicciones en SQLite
   - Exportar gráficos a PNG
   - Historial de predicciones

4. **Alertas**:
   - Notificar cuando predicción supera threshold
   - Email/Telegram integration
   - Trading signals automáticos

---

**Autor**: Claude (Anthropic)
**Fecha**: 2025-10-30
