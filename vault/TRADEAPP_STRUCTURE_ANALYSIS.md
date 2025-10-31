# TradeApp.py - Análisis Completo de Estructura

**Fecha de análisis**: 2025-10-30
**Archivo principal**: TradeApp.py (3,648 líneas)
**Daemon**: trading_daemon.py (1,041 líneas)

---

## Tabla de Contenidos
1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura General](#arquitectura-general)
3. [Clases Principales](#clases-principales)
4. [Módulos y Dependencias](#módulos-y-dependencias)
5. [Flujo de Datos](#flujo-de-datos)
6. [Status Tab - Análisis Detallado](#status-tab---análisis-detallado)
7. [Sistema de Inferencias](#sistema-de-inferencias)
8. [Puntos de Integración](#puntos-de-integración)

---

## Resumen Ejecutivo

**TradeApp** es una aplicación GUI de trading de criptomonedas basada en:
- **Framework**: Tkinter con ttk.Notebook (tabs)
- **Modelo ML**: LSTM dual-head (PyTorch) para predicción de returns + volatilidad
- **Datos**: SQLite database (OHLCV data) + WebSocket live streaming (Binance)
- **Trading**: Daemon en background que hace inferencias periódicas y ejecuta trades (paper/live)
- **Features**: 39 indicadores técnicos (Fibonacci, MA, RSI, ATR, TD Sequential, etc.)

**Características principales**:
- Preview/análisis de datos
- Entrenamiento de modelos LSTM
- Backtesting con métricas detalladas
- Auditoría de features para detectar data leakage
- Status dashboard con daemon de trading automático
- WebSocket para datos en vivo
- Ledger de transacciones

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                         TradeApp GUI                             │
│                    (TradingAppExtended class)                    │
│                                                                  │
│  ┌──────────┬─────────┬──────────┬────────┬─────────────────┐  │
│  │ Preview  │ Training│ Backtest │ Audit  │ Status Tab      │  │
│  │          │         │          │        │ ┌─────────────┐ │  │
│  │          │         │          │        │ │  WebSocket  │ │  │
│  │          │         │          │        │ │  Controls   │ │  │
│  │          │         │          │        │ ├─────────────┤ │  │
│  │          │         │          │        │ │   Daemon    │ │  │
│  │          │         │          │        │ │   Start/Stop│ │  │
│  │          │         │          │        │ ├─────────────┤ │  │
│  │          │         │          │        │ │ Last Rows   │ │  │
│  │          │         │          │        │ │   Table     │ │  │
│  └──────────┴─────────┴──────────┴────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Controls
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TradingDaemon                               │
│                   (Background Thread)                            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Main Loop (poll_interval: 5s default)                      │ │
│  │                                                             │ │
│  │  1. Load recent data from SQLite                           │ │
│  │  2. Compute features (fiboevo.add_technical_features)      │ │
│  │  3. Build sequences (last seq_len rows)                    │ │
│  │  4. Apply scaler + model inference                         │ │
│  │  5. Predict log_return + volatility                        │ │
│  │  6. Trading decision logic                                 │ │
│  │  7. Execute trade (paper or live via ccxt)                 │ │
│  │  8. Write to ledger                                        │ │
│  │  9. Sleep poll_interval seconds                            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Reads/Writes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│                                                                  │
│  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │  SQLite DB       │  │  Artifacts Dir  │  │  Ledger CSV    │ │
│  │  ┌────────────┐  │  │  ┌───────────┐  │  │                │ │
│  │  │ ohlcv table│  │  │  │model_best │  │  │  Trades log    │ │
│  │  │ ts, o,h,l,c│  │  │  │  .pt      │  │  │  timestamps    │ │
│  │  │ volume     │  │  │  ├───────────┤  │  │  predictions   │ │
│  │  └────────────┘  │  │  │meta.json  │  │  │  P&L           │ │
│  │                  │  │  ├───────────┤  │  │                │ │
│  │  WebSocket ─────▶│  │  │scaler.pkl │  │  │                │ │
│  │  (live data)     │  │  └───────────┘  │  │                │ │
│  └──────────────────┘  └─────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Clases Principales

### 1. **_QueueLoggingHandler** (Líneas 167-209)
Manejador de logging que envía mensajes a una queue para mostrar en GUI.

**Métodos**:
- `__init__(q, max_msg_len)`
- `emit(record)`: Envía log a queue

---

### 2. **TradingAppExtended** (Líneas 249-3590)
Clase principal de la aplicación GUI.

#### 2.1 **Atributos de Estado**

**Modelo & Datos**:
```python
self.df_loaded: Optional[pd.DataFrame]    # Datos cargados en memoria
self.model: Optional[torch.nn.Module]     # Modelo LSTM cargado
self.model_scaler: Optional[Any]          # StandardScaler para features
self.model_meta: Optional[Dict]           # Metadatos del modelo (feature_cols, seq_len, horizon, etc.)
self.feature_cols_used: Optional[List]    # Columnas de features usadas
```

**Status Tab & Daemon**:
```python
self.daemon: Optional[TradingDaemon]      # Instancia del daemon de trading
self.ws_app: Optional[WebSocketApp]       # Cliente WebSocket
self.ws_thread: Optional[threading.Thread] # Thread del WebSocket
self.ws_queue: queue.Queue                # Queue para mensajes del WebSocket
self.db_writer_thread: Optional[Thread]   # Thread que escribe WS data a SQLite
```

**Variables de UI (StringVar, IntVar, etc.)**:
```python
# Status tab settings
self.inference_interval_var: DoubleVar    # Intervalo de inferencia (segundos)
self.trade_interval_var: DoubleVar        # Intervalo de trading (segundos)
self.refresh_interval_var: DoubleVar      # Intervalo de refresh UI (segundos)
self.websocket_url_var: StringVar         # URL del WebSocket
self.websocket_status_var: StringVar      # Estado del WebSocket ("Connected"/"Disconnected")

# Model metadata display (read-only)
self.model_exchange_var: StringVar
self.model_symbol_var: StringVar
self.model_timeframe_var: StringVar
self.model_seq_len_var: StringVar
self.model_horizon_var: StringVar
self.model_features_var: StringVar
```

**Tabs & UI Widgets**:
```python
self.root: Tk                             # Ventana principal
self.nb: ttk.Notebook                     # Notebook con tabs
self.log_queue: queue.Queue               # Queue para logs
self.log_text: ScrolledText               # Widget de texto para logs
self.status_data_tree: ttk.Treeview       # Tabla de últimas filas de datos
```

#### 2.2 **Métodos Principales** (Por Categoría)

**Inicialización & UI**:
- `__init__(root)` (252): Constructor principal
- `_build_ui()` (1623): Construye toda la interfaz
- `_build_preview_tab()` (1670): Tab de preview de datos
- `_build_train_tab()` (1682): Tab de entrenamiento
- `_build_backtest_tab()` (1710): Tab de backtesting
- `_build_audit_tab()` (1719): Tab de auditoría de features
- `_build_status_tab()` (2040): **Tab de status con daemon** ⭐
- `_extend_status_tab_ui()` (1430): Extiende status tab con controles WebSocket

**Logging**:
- `_enqueue_log(msg, level)` (1554): Envía mensaje a queue de logs
- `_poll_log_queue()` (1572): Lee queue y muestra en UI (llamado periódicamente)

**Carga de Datos**:
- `_load_data()` (1748): Carga datos desde SQLite (con progreso)
- `_run_bg_load(path, table, symbol, tf, config)` (1758): Worker en background
- `_fetch_last_rows_status(limit)` (2449): Carga últimas filas para Status tab
- `_populate_status_data_tree(df)` (2533): Llena tabla de últimas filas

**Entrenamiento**:
- `_training_worker_prepare_only()` (2887): Prepara dataset para entrenamiento
- `_train_model_worker()` (3032): Entrena modelo LSTM
- `_prepare_and_train_worker()` (3173): Prepara y entrena en un solo paso

**Inferencia & Predicción**:
- `_get_latest_prediction()` (2571): **Genera predicción única en horizonte nativo** ⭐
- `_run_forecast()` (548): Genera forecasts multi-paso (autoregresivo)
- `_plot_history_on_forecast()` (508): Dibuja histórico en canvas

**Status Tab & Daemon** ⭐⭐⭐:
- `_apply_status_settings()` (2361): Aplica intervalos de inferencia/trade/refresh
- `_connect_websocket()` (2382): Conecta WebSocket de Binance
- `_disconnect_websocket()` (908): Desconecta WebSocket
- `_db_writer_loop()` (927): Loop que escribe mensajes WebSocket → SQLite
- `_refresh_status()` (2430): Actualiza metadatos del modelo en UI
- `_start_daemon()` (3479): **Inicia daemon de trading** ⭐
- `_stop_daemon()` (3541): **Detiene daemon de trading** ⭐
- `_load_model_file()` (3555): Carga modelo desde archivo (para daemon)

**Backtesting**:
- `_backtest_worker()` (3344): Ejecuta backtest con métricas

**Utilidades**:
- `_clear_status_data_tree()` (2528): Limpia tabla de datos

---

### 3. **TradingDaemon** (trading_daemon.py, Líneas 78-976)
Daemon en background que ejecuta trading automático.

#### 3.1 **Atributos Principales**

**Configuración**:
```python
self.sqlite_path: str                     # Ruta a SQLite DB
self.sqlite_table: str                    # Tabla (e.g., "ohlcv")
self.symbol: str                          # Par de trading (e.g., "BTCUSDT")
self.timeframe: str                       # Timeframe (e.g., "1h")
self.exchange_id: Optional[str]           # Exchange ID para ccxt
self.paper: bool                          # Paper trading vs live
self.seq_len: int                         # Sequence length (default: 32)
self.poll_interval: float                 # Intervalo del loop (segundos)
```

**Artifacts**:
```python
self.model: Optional[torch.nn.Module]     # Modelo LSTM
self.model_scaler: Optional[Any]          # StandardScaler
self.model_meta: Dict                     # Metadatos (feature_cols, horizon, etc.)
self.artifacts_dir: Path                  # Directorio de artifacts
```

**Trading State**:
```python
self.ledger_path: Path                    # Ruta al ledger CSV
self.position_size: float                 # Tamaño de posición ($)
self.last_trade_ts: float                 # Timestamp del último trade
self.trade_cooldown_sec: float            # Cooldown entre trades
```

**Threading**:
```python
self._stop_flag: threading.Event          # Flag para detener el loop
self._thread: Optional[Thread]            # Thread del daemon
self._config_lock: threading.Lock         # Lock para config
self._artifact_lock: threading.Lock       # Lock para model/scaler
self.ui_log_queue: queue.Queue            # Queue para logs a la UI
```

#### 3.2 **Métodos Principales**

**Lifecycle**:
- `__init__(...)` (88): Constructor con configuración
- `start_loop()` (366): Inicia el loop en background thread
- `stop(wait, timeout)` (376): Detiene el loop
- `run_loop()` (383): **Main loop del daemon** ⭐
- `close(wait, timeout)` (976): Alias de stop()

**Artifacts Management**:
- `load_model_and_scaler(model_path, scaler_path, meta_path)` (409): **Carga modelo/scaler/meta** ⭐
- `save_config(path)` (309): Guarda configuración a JSON
- `update_from_dict(cfg, save, reload_artifacts)` (337): Actualiza config desde dict

**Inference Loop** ⭐⭐⭐:
- `iteration_once()` (627): **UNA iteración del loop** ⭐⭐⭐
  1. Carga datos recientes (`_load_recent_rows()`)
  2. Computa features (`fiboevo.add_technical_features()`)
  3. Construye secuencias (`_build_sequences_internal()`)
  4. Aplica scaler
  5. Hace inferencia con modelo
  6. Predice log_return + volatility
  7. Decide si hacer trade (`_should_trade()`)
  8. Ejecuta trade si aplica (`_execute_trade()`)
  9. Escribe en ledger (`_write_ledger()`)

**Trading Logic**:
- `_should_trade(pred)` (801): Lógica de decisión de trading
- `_execute_trade(pred, df, feats, feature_cols)` (808): Ejecuta trade (paper/live)
- `_write_ledger(entry)` (881): Escribe entrada en ledger CSV

**Data Access**:
- `_load_recent_rows(limit)` (914): **Carga últimas N filas de SQLite** ⭐
- `_build_sequences_internal(df_feats, feature_cols, seq_len, horizon)` (941): Construye secuencias X, y

**Logging**:
- `_enqueue_log(msg, level)` (231): Envía log a queue de UI
- `attach_log_queue(q, replace)` (293): Conecta queue de logs

---

## Módulos y Dependencias

### Dependencias Principales

**Core Python**:
- `threading`: Threads para WebSocket, daemon, background tasks
- `queue`: Comunicación thread-safe entre GUI y workers
- `sqlite3`: Acceso a base de datos
- `json`: Configuración y metadatos
- `logging`: Sistema de logs
- `csv`: Ledger de trades

**GUI**:
- `tkinter` + `tkinter.ttk`: Framework de UI
- `tkinter.scrolledtext.ScrolledText`: Widget de logs

**Data Science**:
- `numpy`: Arrays numéricos
- `pandas`: DataFrames para OHLCV data
- `matplotlib`: Plots en GUI (FigureCanvasTkAgg)

**Machine Learning**:
- `torch` (PyTorch): Red neural LSTM
- `joblib`: Serialización de scaler (StandardScaler)

**Trading**:
- `ccxt`: Exchange connectivity (paper/live trading)
- `websocket-client`: WebSocket para live data (Binance)

**Custom Modules**:
- `fiboevo`: Features técnicas (add_technical_features, LSTM2Head, prepare_input_for_model)
- `trading_daemon`: TradingDaemon class
- `config_manager`: Gestión de configuración (opcional)
- `fp_utils`: Utilidades (opcional)

**Inference Modules** (NEW):
- `multi_horizon_inference`: Predicciones en horizonte único
- `multi_horizon_fan_inference`: Predicciones multi-horizonte con fan
- `future_forecast_fan`: Forecasts al futuro real

---

## Flujo de Datos

### 1. **Datos Históricos** (SQLite)
```
SQLite DB (Binance_BTCUSDT_1h.db)
    ├── Table: ohlcv
    │   └── Columns: ts, open, high, low, close, volume
    │
    └── Usado por:
        ├── TradeApp._load_data() → Carga para preview/training
        ├── TradeApp._fetch_last_rows_status() → Muestra en Status tab
        └── TradingDaemon._load_recent_rows() → Inferencia periódica
```

### 2. **Datos en Vivo** (WebSocket)
```
Binance WebSocket (wss://stream.binance.com/...)
    │
    ├── Streams: btcusdt@aggTrade, btcusdt@depth
    │
    └──▶ TradeApp._connect_websocket()
         └──▶ WebSocketApp.run_forever() (thread)
              └──▶ on_message(ws, msg)
                   └──▶ self.ws_queue.put(msg)
                        └──▶ TradeApp._db_writer_loop() (thread)
                             └──▶ Parsea mensaje
                                  └──▶ INSERT INTO ohlcv (...)
                                       └──▶ SQLite DB actualizado
```

### 3. **Features Engineering**
```
Raw OHLCV Data
    │
    └──▶ fiboevo.add_technical_features(close, high, low, volume)
         │
         └──▶ Calcula 39 features:
              ├── log_close, log_ret_1, log_ret_5
              ├── sma_5, sma_20, sma_50
              ├── ema_5, ema_20, ema_50
              ├── bb_m, bb_std, bb_up, bb_dn, bb_width
              ├── rsi_14, atr_14
              ├── raw_vol_10, raw_vol_30
              ├── Fibonacci retracements (fib_r_236, fib_r_382, ...)
              ├── Fibonacci extensions (fibext_1272, fibext_1618, ...)
              ├── TD Sequential (td_buy_setup, td_sell_setup)
              └── ret_1, ret_5
```

### 4. **Inference Pipeline** (Daemon)
```
TradingDaemon.iteration_once()
    │
    ├── 1. Load data
    │   └──▶ _load_recent_rows(limit=1000)
    │        └──▶ SELECT * FROM ohlcv ORDER BY ts DESC LIMIT 1000
    │
    ├── 2. Compute features
    │   └──▶ fiboevo.add_technical_features(...)
    │        └──▶ DataFrame con 39 features
    │
    ├── 3. Build sequence
    │   └──▶ _build_sequences_internal(df_feats, feature_cols, seq_len=32, horizon=10)
    │        └──▶ X: [seq_len, n_features] (últimas 32 filas)
    │        └──▶ y: [log_return, volatility] (en t+horizon)
    │
    ├── 4. Scale features
    │   └──▶ model_scaler.transform(X)
    │
    ├── 5. Inference
    │   └──▶ model(X_scaled)
    │        └──▶ [pred_log_ret, pred_vol] (horizonte nativo: 10 horas)
    │
    ├── 6. Trading decision
    │   └──▶ _should_trade(pred_log_ret)
    │        └──▶ Basado en threshold (e.g., |pred| > 0.005)
    │
    ├── 7. Execute trade (if triggered)
    │   └──▶ _execute_trade(pred, df, feats, feature_cols)
    │        └──▶ Paper: simula orden
    │        └──▶ Live: ccxt.create_market_order(...)
    │
    └── 8. Log trade
        └──▶ _write_ledger(entry)
             └──▶ Append to ledger CSV
```

---

## Status Tab - Análisis Detallado

### Ubicación en Código
**Método constructor**: `_build_status_tab()` (línea 2040)
**Extensiones**: `_extend_status_tab_ui()` (línea 1430)

### Componentes del Status Tab

```
┌──────────────────────────────────────────────────────────────────┐
│ Status Tab                                                       │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Settings Panel                                               │ │
│ │ ┌─────────────┬──────────────┬───────────────┬─────────────┐ │ │
│ │ │ Inference   │ Trade        │ Refresh       │ [Apply]     │ │ │
│ │ │ Interval: 5s│ Interval: 10s│ Interval: 5s  │             │ │ │
│ │ └─────────────┴──────────────┴───────────────┴─────────────┘ │ │
│ │ ┌─────────────────────────────────────────────────────────┐  │ │
│ │ │ WebSocket URL: wss://stream.binance.com/...            │  │ │
│ │ │ [Connect] [Disconnect]  Status: Connected               │  │ │
│ │ └─────────────────────────────────────────────────────────┘  │ │
│ │ ┌─────────────────────────────────────────────────────────┐  │ │
│ │ │ API Key: ********  API Secret: ********                │  │ │
│ │ └─────────────────────────────────────────────────────────┘  │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Daemon Controls                                              │ │
│ │ [Start Daemon] [Stop Daemon] [Load Model] [Refresh Status]  │ │
│ │ [Fetch Last Data] [Show Forecast (preview)] [Websocket Pane]│ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Model Metadata (read-only)                                   │ │
│ │ Exchange: Binance │ Symbol: BTCUSDT │ Timeframe: 1h         │ │
│ │ SeqLen: 32 │ Horizon: 10 │ Features: 39                     │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Last Data Rows (Treeview)                                    │ │
│ │ ┌────────────┬───────┬───────┬───────┬───────┬────────────┐ │ │
│ │ │ timestamp  │ open  │ high  │ low   │ close │ volume     │ │ │
│ │ ├────────────┼───────┼───────┼───────┼───────┼────────────┤ │ │
│ │ │ 2025-10-17 │106431 │107234 │105987 │106432 │ 15234.56   │ │ │
│ │ │ 2025-10-17 │106432 │106876 │106123 │106543 │ 12456.78   │ │ │
│ │ │    ...     │  ...  │  ...  │  ...  │  ...  │    ...     │ │ │
│ │ └────────────┴───────┴───────┴───────┴───────┴────────────┘ │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ [ESPACIO DISPONIBLE PARA DASHBOARD DE PREDICCIONES]  ⭐⭐⭐      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Funcionalidades Actuales

#### 1. **WebSocket Data Streaming**
- **URL**: Configurable (default: Binance BTCUSDT aggTrade + depth)
- **Botones**: Connect, Disconnect
- **Status indicator**: "Connected" / "Disconnected"
- **Thread**: `ws_thread` ejecuta `WebSocketApp.run_forever()`
- **Queue**: `ws_queue` recibe mensajes del WebSocket
- **Writer thread**: `_db_writer_loop()` consume queue y escribe a SQLite
- **Formato de mensajes**:
  ```json
  {
    "stream": "btcusdt@aggTrade",
    "data": {
      "e": "aggTrade",
      "E": 1234567890,
      "s": "BTCUSDT",
      "p": "106432.50",
      "q": "0.15",
      "T": 1234567890,
      ...
    }
  }
  ```

#### 2. **Trading Daemon**
- **Start Daemon**: Inicia `TradingDaemon` en background
- **Stop Daemon**: Detiene daemon gracefully
- **Model Loading**: Carga modelo desde archivo (artifacts/ o custom path)
- **Inference Loop**: Ejecuta `iteration_once()` cada `poll_interval` segundos
- **Paper/Live**: Configurable (default: paper trading)
- **Ledger**: Guarda trades en CSV (artifacts/ledger_*.csv)

#### 3. **Data Display**
- **Last Rows Table**: Muestra últimas 50 filas de SQLite
- **Refresh**: Botón para refrescar manualmente
- **Fetch Last Data**: Carga datos frescos del daemon

#### 4. **Settings**
- **Inference Interval**: Frecuencia de inferencias del daemon
- **Trade Interval**: Cooldown entre trades
- **Refresh Interval**: Frecuencia de actualización de UI
- **Apply Button**: Aplica cambios (y los pasa al daemon si está corriendo)

---

## Sistema de Inferencias

### Inferencias en TradeApp (GUI)

**Método**: `_get_latest_prediction()` (línea 2571)
**Triggered by**: Botón "Get Latest Prediction" en Status tab
**Thread**: Background (daemon=True)

**Pipeline**:
1. Carga datos recientes (1000 filas) desde SQLite o daemon
2. Computa features con `fiboevo.add_technical_features()`
3. Detecta feature columns (desde `daemon.model_meta` o GUI state)
4. Aplica scaler (desde `daemon.model_scaler` o GUI scaler)
5. Construye secuencia de últimas `seq_len` filas
6. Hace inferencia con modelo (desde `daemon.model` o GUI model)
7. Obtiene `[pred_log_ret, pred_vol]` en horizonte nativo (10h)
8. Convierte log-return a precio: `price_pred = current_price * exp(pred_log_ret)`
9. Muestra resultado en logs

**Características**:
- ✅ Single-shot prediction (no loop)
- ✅ Solo horizonte nativo (10 horas)
- ✅ No multi-horizonte
- ✅ No visualización (solo texto en logs)

### Inferencias en TradingDaemon (Background)

**Método**: `iteration_once()` (línea 627)
**Triggered by**: `run_loop()` cada `poll_interval` segundos (default: 5s)
**Thread**: `daemon._thread` (background daemon thread)

**Pipeline** (idéntico a GUI pero con trading logic):
1. `_load_recent_rows(limit=1000)`
2. `fiboevo.add_technical_features()`
3. `_build_sequences_internal()` → secuencia de últimas 32 filas
4. `model_scaler.transform()`
5. `model(X_scaled)` → `[pred_log_ret, pred_vol]`
6. **Trading decision**: `_should_trade(pred_log_ret)` → True si |pred| > threshold
7. **Execute trade**: `_execute_trade()` → Paper order o ccxt live order
8. **Log trade**: `_write_ledger()` → Append a CSV

**Características**:
- ✅ Continuous loop (periódico)
- ✅ Solo horizonte nativo (10 horas)
- ✅ No multi-horizonte
- ✅ Trading execution
- ✅ Ledger logging

---

## Puntos de Integración

### Donde Integrar Dashboard Multi-Horizonte

#### **Opción Elegida**: Integrar con Status Tab ⭐⭐⭐

**Razones**:
1. ✅ Status tab ya tiene WebSocket → datos en vivo
2. ✅ Daemon ya hace inferencias periódicas → podemos extenderlo a multi-horizonte
3. ✅ Hay espacio disponible debajo de la tabla de datos
4. ✅ Coherencia: todas las funciones "live" en un solo tab
5. ✅ Evita duplicación de lógica

**Ubicación física en Status tab**:
```
Status Tab Layout (ACTUAL):
┌──────────────────────────────────────┐
│ Settings Panel                       │ ← Ya existe
├──────────────────────────────────────┤
│ Daemon Controls                      │ ← Ya existe
├──────────────────────────────────────┤
│ Model Metadata                       │ ← Ya existe
├──────────────────────────────────────┤
│ Last Data Rows Table (Treeview)      │ ← Ya existe
└──────────────────────────────────────┘

Status Tab Layout (PROPUESTO):
┌──────────────────────────────────────┐
│ Settings Panel                       │ ← Mantener
├──────────────────────────────────────┤
│ Daemon Controls + Prediction Controls│ ← Extender con botones dashboard
├──────────────────────────────────────┤
│ Model Metadata                       │ ← Mantener
├──────────────────────────────────────┤
│ Last Data Rows Table (Treeview)      │ ← Mantener (o reducir altura)
├──────────────────────────────────────┤
│ ┌────────────────────────────────┐   │
│ │  MULTI-HORIZON PREDICTION FAN  │   │ ← NUEVO CANVAS ⭐⭐⭐
│ │  (matplotlib FigureCanvasTkAgg)│   │
│ │                                │   │
│ │  [Historical price line]       │   │
│ │  [Fan lines: 1h,3h,5h,10h...]  │   │
│ │  [Confidence bands]            │   │
│ │  [Probability layers]          │   │
│ └────────────────────────────────┘   │
├──────────────────────────────────────┤
│ ┌────────────────────────────────┐   │
│ │  PREDICTIONS TABLE             │   │ ← NUEVA TABLA ⭐
│ │  Horizon | Price | Change | CI │   │
│ │  1h      | $XXX  | +X.X%  | ...│   │
│ │  3h      | $XXX  | +X.X%  | ...│   │
│ │  ...                           │   │
│ └────────────────────────────────┘   │
└──────────────────────────────────────┘
```

---

### Modificaciones Necesarias

#### 1. **Extender TradingDaemon** (trading_daemon.py)

**Añadir método**:
```python
def iteration_once_multi_horizon(self, horizons: List[int] = None) -> Dict[int, Dict]:
    """
    Versión extendida de iteration_once() que genera predicciones
    multi-horizonte usando multi_horizon_fan_inference.

    Args:
        horizons: Lista de horizontes [1, 3, 5, 10, 15, 20, 30]

    Returns:
        Dict con predicciones por horizonte:
        {
            1: {"price": 106500, "log_ret": 0.0005, "vol": 0.2, "ci_lower": 106400, "ci_upper": 106600},
            3: {"price": 106700, ...},
            ...
        }
    """
    if horizons is None:
        horizons = [1, 3, 5, 10, 15, 20, 30]

    # Load recent data
    df = self._load_recent_rows(limit=1000)
    if df is None or df.empty:
        return {}

    # Compute features
    df_feats = fiboevo.add_technical_features(...)

    # Use multi_horizon_fan_inference
    from multi_horizon_fan_inference import predict_single_point_multi_horizon

    predictions = predict_single_point_multi_horizon(
        df=df_feats,
        model=self.model,
        meta=self.model_meta,
        scaler=self.model_scaler,
        device=self.device,
        horizons=horizons,
        method="scaling"
    )

    return predictions
```

**Modificar `run_loop()`**:
```python
def run_loop(self):
    while not self._stop_flag.is_set():
        try:
            # Opción A: Solo inferencia nativa (legacy)
            # self.iteration_once()

            # Opción B: Multi-horizonte (nuevo)
            predictions = self.iteration_once_multi_horizon()

            # Push predictions to queue for GUI
            if hasattr(self, "predictions_queue"):
                self.predictions_queue.put(predictions)

            # Trading logic solo usa horizonte nativo (h=10)
            native_pred = predictions.get(self.model_meta["horizon"], {})
            if self._should_trade(native_pred.get("log_ret", 0)):
                self._execute_trade(native_pred, ...)
        except Exception as e:
            self._enqueue_log(f"iteration_once error: {e}")

        self._stop_flag.wait(self.poll_interval)
```

#### 2. **Extender TradeApp Status Tab** (TradeApp.py)

**En `_build_status_tab()` (línea 2040), añadir al final**:
```python
# --- Multi-Horizon Prediction Dashboard ---
Label(tab, text="Live Multi-Horizon Predictions", font=("Arial", 12, "bold")).pack(anchor="w", padx=6, pady=(10,2))

# Canvas frame
frame_canvas = Frame(tab)
frame_canvas.pack(fill=BOTH, expand=True, padx=6, pady=6)

# Create matplotlib figure for prediction fan
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

self.pred_fan_fig = Figure(figsize=(12, 6), dpi=100)
self.pred_fan_ax = self.pred_fan_fig.add_subplot(111)
self.pred_fan_canvas = FigureCanvasTkAgg(self.pred_fan_fig, master=frame_canvas)
self.pred_fan_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

# Initial empty plot
self.pred_fan_ax.text(0.5, 0.5, "Start daemon to see live predictions",
                      ha="center", va="center", fontsize=14)
self.pred_fan_canvas.draw()

# Predictions table (below canvas)
frame_pred_table = Frame(tab)
frame_pred_table.pack(fill=X, padx=6, pady=6)

Label(frame_pred_table, text="Predictions Summary", font=("Arial", 10, "bold")).pack(anchor="w")

self.pred_summary_tree = ttk.Treeview(frame_pred_table, height=7)
self.pred_summary_tree["columns"] = ("Horizon", "Target Time", "Predicted Price", "Change ($)", "Change (%)", "95% CI", "Signal")
self.pred_summary_tree.column("#0", width=0, stretch=NO)
for col in self.pred_summary_tree["columns"]:
    self.pred_summary_tree.heading(col, text=col)
    self.pred_summary_tree.column(col, width=100)
self.pred_summary_tree.pack(fill=X)

# Dummy data
for h in [1, 3, 5, 10, 15, 20, 30]:
    self.pred_summary_tree.insert("", "end", values=(f"{h}h", "...", "...", "...", "...", "...", "..."))
```

**Añadir método para actualizar gráfico**:
```python
def _update_prediction_fan_display(self, predictions: Dict[int, Dict], df_history: pd.DataFrame):
    """
    Actualiza el canvas de prediction fan con nuevas predicciones.

    Args:
        predictions: Dict {horizon: {price, log_ret, vol, ci_lower, ci_upper}}
        df_history: DataFrame con histórico reciente (últimas 200 filas)
    """
    try:
        ax = self.pred_fan_ax
        ax.clear()

        # Import visualization function
        from dashboard_visualizations import plot_prediction_fan_live

        # Plot
        plot_prediction_fan_live(
            ax=ax,
            df_history=df_history,
            predictions_dict=predictions,
            show_confidence=True,
            show_probability_layers=False,  # Puede ser toggle
            colormap="viridis"
        )

        self.pred_fan_canvas.draw()

        # Update table
        self._update_prediction_summary_table(predictions)

    except Exception as e:
        self._enqueue_log(f"Failed to update prediction fan: {e}")

def _update_prediction_summary_table(self, predictions: Dict[int, Dict]):
    """Actualiza la tabla de resumen de predicciones."""
    try:
        # Clear table
        for item in self.pred_summary_tree.get_children():
            self.pred_summary_tree.delete(item)

        # Populate with new predictions
        for h in sorted(predictions.keys()):
            pred = predictions[h]
            target_time = "..." # Calculate from current time + h
            price = f"${pred['price']:.2f}"
            change_usd = f"${pred['price'] - pred.get('current_price', 0):.2f}"
            change_pct = f"{pred.get('change_pct', 0):.2f}%"
            ci = f"[${pred['ci_lower']:.0f}, ${pred['ci_upper']:.0f}]"
            signal = "↑" if pred.get("log_ret", 0) > 0 else "↓" if pred.get("log_ret", 0) < 0 else "→"

            self.pred_summary_tree.insert("", "end", values=(
                f"{h}h", target_time, price, change_usd, change_pct, ci, signal
            ))
    except Exception as e:
        self._enqueue_log(f"Failed to update prediction table: {e}")
```

**Añadir polling de predictions queue**:
```python
def _poll_predictions_queue(self):
    """
    Llamado periódicamente (cada refresh_interval) para leer predictions
    del daemon y actualizar UI.
    """
    if self.daemon is None:
        self.root.after(5000, self._poll_predictions_queue)  # Retry in 5s
        return

    try:
        # Get predictions from daemon's queue (non-blocking)
        if hasattr(self.daemon, "predictions_queue"):
            try:
                predictions = self.daemon.predictions_queue.get_nowait()

                # Get recent history for plot
                df_history = self.daemon._load_recent_rows(limit=200)

                # Update display (must be from main thread)
                self._update_prediction_fan_display(predictions, df_history)

            except queue.Empty:
                pass  # No new predictions yet
    except Exception as e:
        self._enqueue_log(f"Predictions queue poll error: {e}")

    # Schedule next poll
    interval_ms = int(self.refresh_interval_var.get() * 1000)
    self.root.after(interval_ms, self._poll_predictions_queue)
```

**Iniciar polling cuando daemon arranca**:
```python
def _start_daemon(self):
    # ... código existente ...

    # Add predictions queue to daemon
    import queue as _queue_mod
    self.daemon.predictions_queue = _queue_mod.Queue(maxsize=10)

    self.daemon.start_loop()

    # Start predictions queue polling
    self.root.after(100, self._poll_predictions_queue)

    # ... resto del código ...
```

---

### Resumen de Cambios

| Archivo | Cambios | Líneas Estimadas |
|---------|---------|------------------|
| `trading_daemon.py` | Añadir `iteration_once_multi_horizon()` | +150 |
| `trading_daemon.py` | Modificar `run_loop()` para usar multi-horizonte | +20 |
| `trading_daemon.py` | Añadir `predictions_queue` attribute | +5 |
| `TradeApp.py` | Extender `_build_status_tab()` con canvas y tabla | +80 |
| `TradeApp.py` | Añadir `_update_prediction_fan_display()` | +50 |
| `TradeApp.py` | Añadir `_update_prediction_summary_table()` | +30 |
| `TradeApp.py` | Añadir `_poll_predictions_queue()` | +40 |
| `TradeApp.py` | Modificar `_start_daemon()` para iniciar polling | +10 |
| **TOTAL** | | **~385 líneas** |

---

## Conclusiones

### Ventajas de Integrar con Status Tab

1. ✅ **Reutiliza infraestructura existente**:
   - WebSocket ya descarga datos en vivo
   - Daemon ya hace inferencias periódicas
   - Solo necesitamos extender la lógica de inferencia

2. ✅ **Coherencia de UI**:
   - Todo lo "live" en un solo tab
   - Usuario no necesita cambiar de tab para ver predicciones

3. ✅ **Performance**:
   - Daemon ya está en background thread
   - No duplicamos workers ni threads
   - Queue-based communication (thread-safe)

4. ✅ **Sincronización automática**:
   - Predicciones se actualizan cada `poll_interval` (5s default)
   - Datos siempre frescos desde SQLite
   - No necesitamos cache complejo

5. ✅ **Mínimos cambios**:
   - Solo ~385 líneas de código
   - No rompe funcionalidad existente
   - Backward compatible

### Próximos Pasos

1. **Crear helper function** `predict_single_point_multi_horizon()` en `multi_horizon_fan_inference.py`
2. **Implementar cambios en** `trading_daemon.py`
3. **Implementar cambios en** `TradeApp.py`
4. **Testing**:
   - Verificar que daemon genera predicciones multi-horizonte
   - Verificar que UI se actualiza correctamente
   - Verificar que no hay race conditions
   - Verificar que WebSocket + daemon + predictions funcionan concurrentemente

---

**Autor**: Claude (Anthropic)
**Fecha**: 2025-10-30
**Versión**: 1.0
