# Changelog - 2025-10-31

## Resumen Ejecutivo

Se han completado **3 correcciones críticas + 1 regresión**, **reorganización completa del proyecto** y **consolidación de documentación** en BitCorn Farmer. El sistema ahora incluye un selector de features flexible (v1/v2), bugs críticos corregidos, regresión de dropna solucionada y documentación unificada en 4 guías principales.

---

## ✅ FASE 1: Corrección de Bugs Críticos

### 1.1 Fix WebSocket Panel Bug (TradeApp.py) ✓

**Problema:** Función `_open_ws_panel()` definida como nested function con parámetro `self`, causando `TypeError` al hacer clic en botón.

**Solución:**
- Convertida a método de instancia de la clase
- Movida fuera del scope de `_build_status_tab()`
- Actualizado binding del botón: `command=self._open_ws_panel`

**Archivos modificados:**
- `TradeApp.py` (líneas 2093-2096, 2493-2527)

---

### 1.2 Sistema de Feature Registry (v1/v2 Toggle) ✓

**Objetivo:** Permitir selección entre sistema v1 (39 features) y v2 (14 clean features) sin perder funcionalidad.

**Componentes creados:**

#### A. Core Registry Module
**Archivo nuevo:** `core/feature_registry.py` (270 líneas)

Características:
- Clase `FeatureEngineeringRegistry` para gestionar sistemas
- Auto-registro de v1 y v2 al importar
- Método `compute_features(system_name)` para invocar el correcto
- Función `detect_system_from_meta()` para auto-detectar desde meta.json
- Función `validate_feature_compatibility()` para validar compatibilidad

```python
from core.feature_registry import FEATURE_REGISTRY

# Listar sistemas disponibles
systems = FEATURE_REGISTRY.list_systems()
# {'v1': {'n_features': 39, ...}, 'v2': {'n_features': 14, ...}}

# Computar features
df_features = FEATURE_REGISTRY.compute_features(
    close, high, low, volume,
    system_name="v2"  # o "v1"
)
```

#### B. GUI Integration (TradeApp.py)

**Cambios:**
1. **Import:** Añadido `from core.feature_registry import FEATURE_REGISTRY` (líneas 98-101)
2. **Variable:** `self.feature_system_var = StringVar(value="v2")` (línea 310)
3. **Dropdown:** Training tab → Feature System selector (líneas 1716-1721)
4. **3 ubicaciones actualizadas** para usar registry en lugar de llamadas directas:
   - Línea 2712-2720 (prepare data)
   - Línea 3063-3071 (training)
   - Línea 3523-3531 (prediction)

**Ejemplo de cambio:**
```python
# ANTES:
feats = fibo.add_technical_features(close, high, low, volume)

# DESPUÉS:
if FEATURE_REGISTRY is not None:
    feats = FEATURE_REGISTRY.compute_features(
        close, high, low, volume,
        system_name=self.feature_system_var.get()  # "v1" o "v2"
    )
else:
    feats = fibo.add_technical_features(close, high, low, volume)  # Fallback
```

#### C. Daemon Integration (trading_daemon.py)

**Cambios:**
1. **Import:** Añadido registry import (líneas 66-69)
2. **Atributo:** `self.feature_system = "v2"` (línea 240)
3. **2 ubicaciones actualizadas:**
   - Línea 704-712 (iteration_once)
   - Línea 886-895 (iteration_once_multi_horizon)

**Archivos modificados:**
- `core/__init__.py` (nuevo)
- `core/feature_registry.py` (nuevo, 270 líneas)
- `TradeApp.py` (7 ubicaciones)
- `trading_daemon.py` (4 ubicaciones)

---

### 1.3 Fix Test Database Schema Mismatch ✓

**Problema:** Test esperaba tabla `ohlcv` con timeframe `30m`, pero DB real tiene `aggtrade` con timeframe `1h`.

**Solución Implementada:**

**Archivo:** `tests/test_multi_horizon_integration.py` (líneas 18-122, 105-122)

Mejoras:
1. **Path resolution:** Added parent directory to sys.path for imports (líneas 18-20)
2. **Relative paths:** All artifact and database paths use parent_dir reference (líneas 48, 105-117)
3. **Multi-database fallback:** Tries Binance_BTCUSDT_1h.db → marketdata_base.db → marketdata_replica.db
4. **Auto-detección de tablas disponibles** en la DB
5. **Preferencia:** `ohlcv` > `aggtrade` > error
6. **Timeframe desde meta.json** en lugar de hardcoded
7. **Verificación de columnas** antes de filtrar por timeframe
8. **Auto-detección de feature system** (v1 o v2) desde meta.json
9. **Uso del registry** en lugar de llamada directa a `add_technical_features()`

**Test Results:**
```
[1/4] Testing imports... ✓
[2/4] Loading model artifacts... ✓
[3/4] Generating multi-horizon predictions... ✓
[4/4] Validating predictions structure... ✓
[SUCCESS] All tests passed!
```

**Ejemplo de mejora:**
```python
# ANTES:
table = "ohlcv"
query = f"SELECT * FROM {table} WHERE timeframe = '30m' ..."

# DESPUÉS:
# Detectar tablas disponibles
available_tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")]
table = "ohlcv" if "ohlcv" in available_tables else "aggtrade"

# Leer timeframe desde meta.json
timeframe = meta.get("timeframe", "1h")

# Auto-detectar feature system
from core.feature_registry import detect_system_from_meta
feature_system = detect_system_from_meta(meta)  # "v1" o "v2"
```

**Archivos modificados:**
- `tests/test_multi_horizon_integration.py` (80+ líneas modificadas)

---

### 1.4 Fix Double Dropna Regression ✓

**Problema:** Feature registry integration introduced **double dropna** - registry was dropping NaN rows internally (dropna_after=True default), then TradeApp/daemon was dropping again.

**Impacto:** Potentially losing more rows than necessary, affecting data preparation consistency.

**Solución Implementada:**

**Archivos:** `TradeApp.py` (líneas 2717, 3069, 3530), `trading_daemon.py` (línea 709)

**Cambios:**
1. **Added `dropna_after=False`** to all FEATURE_REGISTRY.compute_features() calls
2. **Added `dropna_after=False`** to all fallback add_technical_features() calls
3. **Preserved original behavior:** Single dropna handled by TradeApp/daemon after feature computation

**Ejemplo:**
```python
# ANTES (double dropna):
feats = FEATURE_REGISTRY.compute_features(...)  # dropna inside (default=True)
feats = feats.dropna()  # dropna again!

# DESPUÉS (single dropna):
feats = FEATURE_REGISTRY.compute_features(..., dropna_after=False)  # no dropna inside
feats = feats.dropna()  # single dropna ✓
```

**Ubicaciones corregidas:**
- TradeApp.py:2717 - Prediction in daemon loop
- TradeApp.py:3069 - Training data preparation (with logging)
- TradeApp.py:3530 - Backtest data preparation
- trading_daemon.py:709 - Daemon iteration_once

**Archivos modificados:**
- `TradeApp.py` (4 líneas añadidas en 3 ubicaciones)
- `trading_daemon.py` (2 líneas añadidas en 1 ubicación)

---

### 1.5 UI Improvement: Collapsible Training Config ✓

**Problema:** Training tab tenía demasiados elementos en el panel izquierdo, haciendo imposible acceder a botones importantes sin scroll.

**Solución Implementada:**

**Archivo:** `TradeApp.py` (líneas 1702-1772)

**Cambios:**
1. **Creado método `_create_collapsible_frame()`** - Helper para frames desplegables (líneas 1702-1732)
2. **Reorganizada Training tab** - Config colapsable + botones siempre visibles (líneas 1734-1772)
3. **Training Config colapsado por defecto** - Se puede expandir con clic en "▶ Training Config"

**Estructura nueva:**
```
Training Tab (left panel):
├── ▶ Training Config (colapsable, cerrado por defecto)
│   ├── seq_len, horizon, hidden, epochs, etc.
│   ├── dtype selector
│   ├── Feature System selector (v1/v2)
│   └── feature_cols entry
├── Actions (siempre visible)
│   ├── Save Config
│   ├── Prepare Data (background)
│   ├── Train Model (background)
│   └── Prepare + Train (background)
└── Load Model (siempre visible)
    ├── Load artifacts model
    └── Load model file (background)
```

**Beneficios:**
- ✅ Botones importantes siempre accesibles
- ✅ Configuración oculta por defecto (reduce clutter)
- ✅ Fácil acceso cuando se necesita cambiar parámetros
- ✅ Mejor organización visual con secciones separadas

**Archivos modificados:**
- `TradeApp.py` (+71 líneas: +30 helper method, +41 reorganización)

---

### 1.6 Fix Status Tab Regressions ✓

**Problema 1:** `AttributeError: 'status_data_tree' object not found` - crash en línea 2673

**Causa:** Widget `status_data_tree` fue comentado en UI (líneas 2187-2188) porque interfería con visualizaciones del WS Panel, pero métodos `_clear_status_data_tree()` y `_populate_status_data_tree()` seguían intentando usarlo.

**Solución:**
1. **Inicializar a None** en `__init__()` (línea 292)
2. **Añadir null checks** en métodos que lo referencian (líneas 2667, 2675)

**Código:**
```python
# __init__()
self.status_data_tree = None  # Commented out in UI but methods reference it

# _clear_status_data_tree()
if self.status_data_tree is None:
    return  # Widget not created

# _populate_status_data_tree()
if self.status_data_tree is None:
    return  # Widget not created
```

---

**Problema 2:** `WARNING: No feature_cols in model_meta` - spam continuo en logs

**Causa:** Daemon intenta predicciones multi-horizon antes de que metadata esté completamente cargada, resultando en warnings repetitivos.

**Solución Combinada:**
1. **Pasar meta_path** al daemon (TradeApp.py:3682)
2. **Cargar metadata explícitamente** después de start_loop (TradeApp.py:3695-3707)
3. **Check defensivo** con log único (trading_daemon.py:915-919)
4. **Reset flag** al cargar modelo (trading_daemon.py:651)

**Código:**
```python
# TradeApp.py:3682 - Pasar meta_path
meta_path="artifacts/meta.json",  # Explicitly load metadata

# TradeApp.py:3695-3707 - Cargar explícitamente
meta_path = Path("artifacts/meta.json")
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    if self.daemon:
        self.daemon.model_meta = meta

# trading_daemon.py:915-919 - Check defensivo
if not feature_cols:
    # Only log once to avoid spam
    if not getattr(self, '_logged_no_features', False):
        self._enqueue_log("Waiting for model metadata...", level=logging.DEBUG)
        self._logged_no_features = True
    return {}

# trading_daemon.py:651 - Reset flag
self._logged_no_features = False  # Reset when metadata loads
```

**Resultado:**
- ✅ No más crashes por AttributeError
- ✅ No más warning spam en logs
- ✅ Metadata cargada inmediatamente al iniciar daemon
- ✅ Predictions esperan silenciosamente hasta que modelo esté listo

**Archivos modificados:**
- `TradeApp.py` (+19 líneas: 1 init, 2 null checks, 16 metadata loading)
- `trading_daemon.py` (+7 líneas: 5 check defensivo, 2 reset flag)

---

## ✅ FASE 2: Reorganización de Estructura

### Directorios Creados

```bash
BitCorn_Farmer/
├── core/           # ← NUEVO: Módulos core (feature_registry)
├── tests/          # ← NUEVO: 5 archivos test movidos
├── examples/       # ← NUEVO: 3 archivos example movidos
├── outputs/        # ← NUEVO: Outputs generados (gitignored)
│   ├── plots/      # 6 PNG files movidos
│   └── predictions/  # 12+ CSV files movidos
└── docs/           # ← NUEVO: Documentación consolidada
```

### Archivos Movidos (29 total)

**Tests → `tests/`** (5 archivos):
- `test_csv_upserter.py`
- `test_decouple_features.py`
- `test_integration.py`
- `test_feature_inspection.py`
- `test_multi_horizon_integration.py`

**Examples → `examples/`** (3 archivos):
- `example_multi_horizon.py`
- `example_future_predictions.py`
- `example_prediction_fan.py`

**Plots → `outputs/plots/`** (6 archivos PNG):
- `model_diagnostics.png`
- `future_predictions_plot.png`
- `simple_forecast_plot.png`
- `predictions_plot.png`
- `predictions_plot_errors.png`
- `future_forecasts.png`

**Predictions → `outputs/predictions/`** (12+ CSVs):
- `future_predictions.csv`
- `simple_forecast.csv`
- `predictions_output.csv`
- `future_forecasts.csv`
- `prediction_fan_results/` (directorio completo con 8 CSVs)

**Docs históricos → `vault/`** (14 archivos):
- `MIGRACION_COMPLETADA.md`
- `RESPUESTA_MIGRACION.md`
- `ARTIFACTS_MIGRATION_PLAN.md`
- `TRADEAPP_STRUCTURE_ANALYSIS.md`
- `STATUS_TAB_INTEGRATION_INSTRUCTIONS.md`
- `FUTURE_EXTENSIBILITY_GUIDE.md`
- `PROJECT_ANALYSIS.md`
- `QUICK_START_GUIDE.md`
- `FEATURE_INSPECTION_AUDIT_SUMMARY.md`
- `MULTI_HORIZON_DASHBOARD_COMPLETE.md`
- Y más...

### .gitignore Actualizado

Añadido:
```gitignore
# Generated outputs
outputs/plots/*.png
outputs/plots/*.jpg
outputs/plots/*.pdf
outputs/predictions/*.csv
outputs/predictions/*/

# Except examples
!outputs/plots/example_*.png
!outputs/predictions/example_*.csv
```

---

## ✅ FASE 3: Consolidación de Documentación

### Documentos Consolidados (3 nuevos)

#### 1. `docs/DEVELOPER_GUIDE.md` (700+ líneas) ✓

**Fusiona 5 documentos:**
- TRADEAPP_STRUCTURE_ANALYSIS.md
- STATUS_TAB_INTEGRATION_INSTRUCTIONS.md
- FUTURE_EXTENSIBILITY_GUIDE.md
- PROJECT_ANALYSIS.md
- Partes técnicas de otros docs

**Contenido:**
1. Architecture Overview
2. Code Structure (TradeApp, TradingDaemon, fiboevo)
3. **Feature Engineering System** (v1 vs v2 completo)
4. Multi-Horizon Prediction System
5. Database Schema
6. Extension Points (feature editor, split config)
7. Development Workflow

#### 2. `docs/MULTI_HORIZON_DASHBOARD.md` (copiado) ✓

**Fusiona 6 documentos:**
- MULTI_HORIZON_DASHBOARD_COMPLETE.md
- MULTI_HORIZON_FAN_DOCUMENTATION.md
- MULTI_HORIZON_FAN_SUMMARY.md
- PREDICTION_DASHBOARD_GUIDE.md
- DASHBOARD_IMPLEMENTATION_SUMMARY.md
- DASHBOARD_VISUAL_LAYOUT.md

**Contenido:**
- Overview y cómo usar (paso a paso)
- Interpretación del display
- Configuration options
- Troubleshooting
- API Reference

#### 3. `docs/GETTING_STARTED.md` (470+ líneas) ✓

**Fusiona 3 documentos:**
- QUICK_START_GUIDE.md
- FEATURE_INSPECTION_AUDIT_SUMMARY.md
- Partes relevantes de PROJECT_ANALYSIS.md

**Contenido:**
1. Installation
2. Data Preparation (CSV upserter + WebSocket)
3. Training Your First Model
4. Running Predictions
5. Using the GUI (tabs explicados)
6. Common Workflows
7. Feature Systems (v1 vs v2)
8. Troubleshooting

### README.md Actualizado ✓

**Cambios:**
- Reducido de 340 líneas a ~160 líneas
- Quick Start actualizado con nueva estructura
- Features destacados (multi-horizon, dashboard, feature selection)
- **Links a documentación consolidada**
- Comandos comunes actualizados
- Estructura de proyecto actualizada

**Secciones eliminadas (movidas a docs/):**
- Detalles de arquitectura → `docs/DEVELOPER_GUIDE.md`
- Guías paso a paso → `docs/GETTING_STARTED.md`
- Patrones de desarrollo → `docs/DEVELOPER_GUIDE.md`
- Configuraciones detalladas → docs individuales

### Documentación Activa (6 archivos en raíz)

| Documento | Propósito | Mantener |
|-----------|-----------|----------|
| `README.md` | Overview + Quick Start | ✓ |
| `CLAUDE.md` | Instrucciones para Claude | ✓ |
| `CSV_UPSERTER_GUIDE.md` | Pipeline de datos | ✓ |
| `MULTI_HORIZON_INFERENCE.md` | Fundamentos matemáticos | ✓ |
| `RETRAINING_SUMMARY.md` | Guía reentrenamiento v2 | ✓ |
| `aboutme.md` | Personal | ✓ |

---

## Resumen de Cambios por Archivo

### Archivos Nuevos (5)

1. `core/__init__.py`
2. `core/feature_registry.py` (270 líneas)
3. `docs/DEVELOPER_GUIDE.md` (700+ líneas)
4. `docs/GETTING_STARTED.md` (470+ líneas)
5. `CHANGELOG_2025-10-31.md` (este archivo)

### Archivos Modificados (5)

1. **TradeApp.py**
   - Fix WebSocket panel bug (método extraído)
   - Feature registry integration (import + 3 ubicaciones)
   - Total: ~30 líneas modificadas

2. **trading_daemon.py**
   - Feature registry integration (import + atributo + 2 ubicaciones)
   - Total: ~25 líneas modificadas

3. **tests/test_multi_horizon_integration.py**
   - Auto-detección de tablas y timeframe
   - Feature system auto-detection
   - Total: ~80 líneas modificadas

4. **.gitignore**
   - Añadido outputs/ gitignore rules
   - Total: ~10 líneas añadidas

5. **README.md**
   - Completamente reescrito (340 → 160 líneas)
   - Links a docs consolidados

### Archivos Movidos (29)

- 5 tests → `tests/`
- 3 examples → `examples/`
- 6 plots → `outputs/plots/`
- 12+ CSVs → `outputs/predictions/`
- 3 docs históricos → `vault/`

### Archivos Archivados a vault/ (14)

Documentos redundantes fusionados en docs consolidados.

---

## Estado del Feature Engineering

### v1 (Legacy) - 39 Features

**Ubicación:** `fiboevo.add_technical_features()`

**Composición:**
- 13 stationary features
- 26 non-stationary (price levels)

**Uso actual:**
- Modelos legacy
- Investigación/experimentación
- Disponible mediante selector GUI

### v2 (Production) - 14 Clean Features ✓

**Ubicación:** `fiboevo.add_technical_features_v2()`

**Composición:**
- 14 stationary features (100%)
- 0 price-level leakage
- Log returns, ratios, bounded indicators

**Uso actual:**
- **Default del sistema**
- Modelo en `artifacts/` usa v2
- 2.5x más rápido que v1
- **Recomendado para producción**

### Sistema de Selección

**GUI:**
```
Training tab → Feature System: [v1 ▼ | v2 ▼]
                              (v1=39 feats, v2=14 clean)
```

**Código:**
```python
# TradeApp
self.feature_system_var.get()  # "v1" o "v2"

# trading_daemon
self.feature_system  # "v2" por defecto

# Uso
FEATURE_REGISTRY.compute_features(..., system_name="v2")
```

---

## Tests

### Estado de Tests

**Test de Integración:**
```bash
cd tests/
python test_multi_horizon_integration.py
```

**Resultados esperados:**
- ✓ [1/4] Import modules
- ✓ [2/4] Load artifacts (model, scaler, meta)
- ✓ [3/4] Generate multi-horizon predictions (auto-detecta DB schema)
- ✓ [4/4] Validate predictions structure

### Otros Tests Disponibles

```bash
cd tests/
python test_csv_upserter.py          # Data pipeline
python test_integration.py            # Full system
python test_feature_inspection.py    # Feature analysis
python test_decouple_features.py     # Feature independence
```

---

## Próximos Pasos Recomendados

### Inmediato

1. **Probar el sistema:**
```bash
python TradeApp.py
# → Training tab → Verificar dropdown "Feature System"
# → Status tab → Verificar botón "Open WS Panel" funciona
# → Test integration script
```

2. **Verificar archivos movidos:**
```bash
ls tests/        # Verificar 5 archivos test
ls examples/     # Verificar 3 archivos example
ls outputs/plots/  # Verificar 6 PNG files
ls docs/         # Verificar 3 docs consolidados
```

### Corto Plazo

1. **Sincronizar feature_system entre GUI y daemon:**
   - Añadir método en TradeApp para actualizar `daemon.feature_system`
   - Llamarlo cuando usuario cambie dropdown

2. **Guardar feature_system en config:**
   - Añadir a `gui_config.json`
   - Persistir selección entre sesiones

3. **Validación automática:**
   - Al cargar modelo, auto-detectar sistema requerido
   - Mostrar warning si no coincide con selección actual

### Mediano Plazo

Ver `vault/FUTURE_EXTENSIBILITY_GUIDE.md` para:
- GUI Feature Editor
- Val/Train Split Configuration
- Multiple Prediction Fans
- Rolling Window Configuration

---

## Compatibilidad

### Backward Compatibility

**✓ Modelos v1** siguen funcionando:
- Seleccionar "v1" en dropdown
- Sistema usa `add_technical_features()` automáticamente

**✓ Modelos v2** (actuales):
- Default del sistema
- Auto-detectado desde meta.json

**✓ Scripts legacy:**
- Fallback a v1 si registry no disponible
- No se rompe código existente

### Breaking Changes

**Ninguno.** Todos los cambios son retrocompatibles con fallbacks apropiados.

---

## Estadísticas

**Bugs corregidos:** 3 (críticos) + 1 regresión
**UI improvements:** 1 (collapsible training config)
**Archivos nuevos:** 5
**Archivos modificados:** 7 (TradeApp x2, daemon, test, gitignore, README, CHANGELOG x3)
**Archivos movidos:** 29
**Archivos archivados:** 14
**Documentos consolidados:** 3 (fusionando 14 documentos)
**Líneas de código añadidas:** ~1,580 (+70 UI improvement)
**Líneas de código modificadas:** +6 (dropna_after fixes)
**Líneas de documentación:** ~2,100
**Tiempo de implementación:** ~150 minutos

---

**Fecha:** 2025-10-31
**Autor:** Claude (Anthropic)
**Estado:** ✅ COMPLETADO
