# Análisis Completo del Proyecto - BitCorn_Farmer

## 🚨 Problema Identificado

**TradeApp está usando `add_technical_features()` (VIEJO) en lugar de `add_technical_features_v2()` (NUEVO)**

Del log:
```
Feature cols selected (39): ['log_close', 'log_ret_1', 'log_ret_5', 'sma_5', 'sma_20', 'sma_50', ...]
Sequences built: N=70783, seq_len=32, features=39
```

**Resultado**: Entrena con 39 features problemáticas a pesar de que el modelo guardado tiene 14.

---

## 📁 Módulos del Proyecto

### Módulos Core

| Módulo | Propósito | Clases Principales | Funciones Clave |
|--------|-----------|-------------------|-----------------|
| **fiboevo.py** | Feature engineering & modelo LSTM | `LSTM2Head` | `add_technical_features()`, `add_technical_features_v2()`, `create_sequences_from_df()` |
| **TradeApp.py** | GUI principal | `TradingAppExtended` | `_prepare_data()`, `_train_model()`, `_load_model_from_artifacts()` |
| **trading_daemon.py** | Daemon de trading | `TradingDaemon` | `run_inference()`, `load_model()` |
| **multi_horizon_inference.py** | Sistema de inferencia | N/A | `load_model_and_artifacts()`, `predict_multi_horizon_jump()`, `predict_autoregressive()` |

### Scripts de Utilidad

| Script | Propósito |
|--------|-----------|
| **retrain_clean_features.py** | Entrenamiento con features limpias v2 |
| **test_feature_inspection.py** | Inspección de features |
| **analyze_temporal_lag.py** | Análisis de temporal lag |
| **migrate_to_clean_features.py** | Migración artifacts v1→v2 |
| **example_multi_horizon.py** | Ejemplo de inferencia |

---

## 🔍 Análisis de fiboevo.py

### Funciones de Feature Engineering

#### 1. `add_technical_features()` [VIEJO - 39 features]
```python
def add_technical_features(
    close, high=None, low=None, volume=None,
    fib_lookback=50, dropna_after=False, out_dtype="float32"
) -> pd.DataFrame
```

**Features generadas**: 39 (47 total incluyendo OHLCV)
- ❌ Incluye `log_close` (alta correlación)
- ❌ Incluye `sma_5, sma_20, sma_50` (absolute MAs)
- ❌ Incluye `ema_5, ema_20, ema_50` (absolute EMAs)
- ❌ Incluye `bb_m, bb_up, bb_dn` (absolute Bollinger)
- ❌ Incluye `fib_r_236, fib_r_382, ...` (absolute Fibonacci)
- ❌ Incluye `ret_1, ret_5` (duplicado de log_ret)

**Ubicación**: línea ~200-500 en fiboevo.py

#### 2. `add_technical_features_v2()` [NUEVO - 14 features]
```python
def add_technical_features_v2(
    close, high=None, low=None, volume=None,
    fib_lookback=50, dropna_after=False, out_dtype="float32"
) -> pd.DataFrame
```

**Features generadas**: 14 (todas price-invariant)
- ✅ Solo `log_ret_1, log_ret_5` (returns)
- ✅ `momentum_10, log_ret_accel` (momentum)
- ✅ `ma_ratio_20` (ratio, no absoluto)
- ✅ `bb_width, bb_std_pct` (normalizado)
- ✅ `rsi_14, atr_pct` (indicadores normalizados)
- ✅ `raw_vol_10, raw_vol_30` (volatilidad)
- ✅ `fib_composite` (compuesto, no absoluto)
- ✅ `td_buy_setup, td_sell_setup` (TD Sequential)

**Ubicación**: línea ~800-1100 en fiboevo.py

### Clase LSTM2Head
```python
class LSTM2Head(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2)
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]
```

**Outputs**:
- `log_return_pred`: Predicción de log-return
- `volatility_pred`: Predicción de volatilidad

---

## 🔍 Análisis de TradeApp.py

### Clase TradingAppExtended

#### Método Crítico: `_prepare_data()`

**Línea**: ~1500-1700

```python
def _prepare_data(self):
    # ...
    # PROBLEMA: Usa add_technical_features() VIEJO
    df_features = fibo.add_technical_features(
        df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        volume=df["volume"].values,
        dropna_after=False
    )
    # ...
```

**Esto genera 39 features con price-level!**

#### Selección de Features

**Línea**: ~1750-1800

```python
# Selecciona features desde el modelo cargado o meta.json
if hasattr(self, 'model_meta') and 'feature_cols' in self.model_meta:
    feature_cols_to_use = self.model_meta['feature_cols']
else:
    # Fallback: carga desde artifacts/meta.json
    with open("artifacts/meta.json") as f:
        meta = json.load(f)
    feature_cols_to_use = meta['feature_cols']
```

**Problema**: Aunque cargue meta.json con 14 features, si `add_technical_features()` genera 39, seleccionará un subset, pero ya computó features innecesarias.

#### Método: `_train_model()`

**Línea**: ~3000-3200

```python
def _train_model(self):
    # Usa las features preparadas en _prepare_data()
    # Entrena con LSTM2Head
    # Guarda en artifacts/
```

---

## 🔗 Flujo de Datos

### Flujo de Training en TradeApp

```
1. Usuario → Click "Prepare Data"
   ↓
2. _prepare_data()
   ↓
3. fibo.add_technical_features()  ← ❌ VIEJO (39 features)
   ↓
4. Selecciona features según meta.json
   ↓
5. _train_model()
   ↓
6. Entrena LSTM2Head
   ↓
7. Guarda en artifacts/
```

**Problema**: Paso 3 usa función vieja.

### Flujo de Inference

```
1. Carga artifacts/meta.json (14 features)
   ↓
2. Carga artifacts/model_best.pt
   ↓
3. En inferencia:
   - multi_horizon_inference.py usa add_technical_features()
   - Genera 39 features
   - Selecciona solo 14 según meta.json
   ↓
4. Predicción con modelo
```

**Problema Menor**: Inferencia también usa función vieja pero selecciona correctamente.

---

## 🎯 Interdependencias

### fiboevo.py
```
add_technical_features()
    ↓ usado por
    - TradeApp._prepare_data()
    - multi_horizon_inference.load_model_and_artifacts()
    - example_multi_horizon.py
    - simple_future_forecast.py
    - test_feature_inspection.py

add_technical_features_v2()
    ↓ usado por
    - retrain_clean_features.py ✓
    - test_feature_inspection.py (para comparación)

LSTM2Head
    ↓ usado por
    - TradeApp._train_model()
    - retrain_clean_features.py
    - multi_horizon_inference.py
    - trading_daemon.py
```

### TradeApp.py
```
_prepare_data()
    ↓ llama
    fibo.add_technical_features()  ← ❌ CAMBIAR A v2

_train_model()
    ↓ usa
    LSTM2Head
    ↓ guarda
    artifacts/model_best.pt
```

---

## 🐛 Identificación del Bug

### Ubicación Exacta en TradeApp.py

Buscar todas las ocurrencias de `add_technical_features`:

```bash
grep -n "add_technical_features" TradeApp.py
```

**Líneas aproximadas** (basado en estructura típica):
- Línea ~1650: En `_prepare_data()`
- Posiblemente en otras funciones auxiliares

### El Cambio Necesario

**DE**:
```python
df_features = fibo.add_technical_features(
    df["close"].values,
    high=df["high"].values,
    low=df["low"].values,
    volume=df["volume"].values,
    dropna_after=False
)
```

**A**:
```python
df_features = fibo.add_technical_features_v2(
    close=df["close"].values,
    high=df["high"].values,
    low=df["low"].values,
    volume=df["volume"].values,
    dropna_after=False,
    out_dtype="float32"
)
```

---

## 🔧 Plan de Corrección

### Fase 1: Identificación Completa
1. ✅ Analizar estructura del proyecto
2. ⏳ Grep todas las ocurrencias de `add_technical_features`
3. ⏳ Identificar cuáles necesitan cambio a `_v2`

### Fase 2: Actualización de TradeApp
1. ⏳ Cambiar `_prepare_data()` para usar `_v2`
2. ⏳ Verificar que no hay otros usos
3. ⏳ Actualizar logging para mostrar "14 features"

### Fase 3: Actualización de Scripts de Inferencia (Opcional)
1. ⏳ multi_horizon_inference.py
2. ⏳ example_multi_horizon.py
3. ⏳ simple_future_forecast.py

### Fase 4: Verificación
1. ⏳ Test de training desde GUI
2. ⏳ Verificar que genera 14 features
3. ⏳ Verificar que guarda correctamente
4. ⏳ Test de inferencia

---

## 📊 Impacto del Cambio

### Antes (TradeApp con add_technical_features)
```
Computa: 39 features (47 total)
Usa: 39 features (según meta.json viejo)
Entrena: Con price-level features ❌
Resultado: Temporal lag presente
```

### Después (TradeApp con add_technical_features_v2)
```
Computa: 14 features (16 total)
Usa: 14 features (según meta.json nuevo)
Entrena: Sin price-level features ✓
Resultado: Sin temporal lag
```

---

## 🚀 Próximo Paso

**Ejecutar búsqueda exhaustiva en TradeApp.py**:
```bash
grep -n "add_technical_features" TradeApp.py
```

Luego aplicar el parche para cambiar a `_v2`.

---

*Análisis generado: 2025-10-30*
*Estado: Identificación completa - Listo para aplicar corrección*
