# Resumen de Reentrenamiento del Modelo LSTM

## 🔍 Diagnóstico del Problema

### Problema Identificado
El modelo anterior mostraba **"naive forecasting"** - las predicciones simplemente copiaban el precio actual con ajustes mínimos, resultando en un lag visual de 12h.

### Métricas del Modelo Anterior (v1)
```
MAE: $1,242 (solo 3% mejor que persistence forecast $1,280)
Directional Accuracy: 54.8% (apenas mejor que 50% random)
Variance Ratio: 0.15 (predicciones 6.7x más conservadoras que realidad)
Correlation: 0.226 (muy baja)
```

### Causa Raíz: **DATA LEAKAGE**

**Features problemáticas (17 eliminadas):**
1. `log_close` - Precio absoluto → modelo memoriza P_t
2. `sma_5`, `sma_20`, `sma_50` - Moving averages absolutas
3. `ema_5`, `ema_20`, `ema_50` - Exponential moving averages absolutas
4. `bb_m`, `bb_up`, `bb_dn` - Bandas de Bollinger absolutas
5. `fib_r_*`, `fibext_*` (8 features) - Niveles Fibonacci absolutos

**Correlación con close:** >0.99 para todas las features de precio absoluto

---

## ✅ Solución Implementada

### 1. **Nuevas Features Limpias** (`add_technical_features_v2()`)

**14 features sin data leakage:**

```python
# Returns (históricos, válidos)
- log_ret_1, log_ret_5
- momentum_10  # NEW: tasa de cambio 10 períodos
- log_ret_accel  # NEW: aceleración de returns

# Ratios normalizados (no absolutos)
- ma_ratio_20  # NEW: (close - sma_20) / sma_20
- bb_width  # ancho de Bollinger normalizado
- bb_std_pct  # NEW: std normalizado por precio

# Indicadores técnicos (scale-independent)
- rsi_14
- atr_pct  # NEW: ATR normalizado por close
- raw_vol_10, raw_vol_30

# Fibonacci composite (reemplaza 8 features individuales)
- fib_composite  # NEW: ponderado con inverse distance weighting

# TD Sequential
- td_buy_setup, td_sell_setup
```

**Características clave:**
- ✅ Sin información de precio absoluto
- ✅ Solo ratios, returns, y distancias normalizadas
- ✅ Fibonacci composite con ponderación inteligente
- ✅ Reducción de 39 → 14 features (más eficiente)

### 2. **Función de Pérdida Mejorada** (`combined_loss()`)

```python
total_loss = MSE_return
           + 0.5 * MSE_volatility
           + 0.3 * directional_loss  # NEW: penaliza dirección incorrecta
           + 0.2 * variance_loss      # NEW: fuerza predicciones con varianza realista
```

**Componentes:**
- `directional_accuracy_loss`: Penaliza predicciones con signo incorrecto
- `variance_matching_loss`: Previene "hedging" (predicciones demasiado conservadoras)

### 3. **Arquitectura Mejorada**

```python
# Antes (v1)
LSTM2Head(input_size=39, hidden=92, layers=2, dropout=0.1)

# Ahora (v2)
LSTM2Head(input_size=14, hidden=128, layers=3, dropout=0.2)
```

**Cambios:**
- ✅ Hidden size: 92 → 128 (+39% capacidad)
- ✅ Layers: 2 → 3 (+50% profundidad)
- ✅ Dropout: 0.1 → 0.2 (mejor regularización)
- ✅ Features: 39 → 14 (más limpias, menos ruido)

### 4. **Script de Reentrenamiento** (`retrain_clean_features.py`)

**Características:**
- Cosine annealing learning rate scheduler
- Early stopping based on directional accuracy (no solo loss)
- Tracking de métricas adicionales (dir_acc, var_ratio, corr)
- Temporal split (sin shuffling) para validación realista
- Guarda historial de entrenamiento en CSV

**Uso:**
```bash
python retrain_clean_features.py \
  --epochs 100 \
  --lr 0.001 \
  --batch-size 256 \
  --hidden 128 \
  --layers 3 \
  --use-combined-loss
```

---

## 🎯 Métricas Objetivo (v2)

| Métrica | v1 (Actual) | Target v2 | Mejora |
|---------|------------|-----------|---------|
| MAE improvement vs persistence | 3% | >15% | 5x |
| Directional Accuracy | 54.8% | >60% | +5.2 pts |
| Variance Ratio | 0.15 | >0.5 | 3.3x |
| Correlation | 0.226 | >0.5 | 2.2x |

---

## 📁 Archivos Modificados/Creados

### Modificados:
1. **`fiboevo.py`**
   - Línea 569-727: `add_technical_features_v2()` agregada
   - Línea 874-971: Loss functions mejoradas agregadas
   - No se modificó código existente (backward compatible)

### Creados:
2. **`retrain_clean_features.py`** (675 líneas)
   - Script completo de reentrenamiento
   - Usa features v2 y combined_loss
   - Tracking mejorado de métricas

3. **`diagnose_model.py`** (450 líneas)
   - Herramienta de diagnóstico
   - Identifica data leakage
   - Compara con persistence baseline
   - Genera plots de análisis

4. **`RETRAINING_SUMMARY.md`** (este archivo)
   - Documentación completa
   - Guía de uso

### Directorios:
5. **`artifacts_v2/`** (se creará al entrenar)
   - `model_best.pt` - Modelo reentrenado
   - `meta.json` - Metadata con 14 features limpias
   - `scaler.pkl` - Scaler entrenado solo con training data
   - `training_history.csv` - Historial de métricas por época

---

## 🚀 Cómo Usar

### Paso 1: Verificar Features Actuales
```bash
python diagnose_model.py
```
Esto muestra:
- Data leakage en v1
- Baseline metrics
- Distribución de predicciones vs realidad

### Paso 2: Reentrenar con Features Limpias
```bash
# Training básico (100 epochs)
python retrain_clean_features.py --epochs 100

# Training extendido con custom params
python retrain_clean_features.py \
  --epochs 200 \
  --lr 0.0005 \
  --batch-size 512 \
  --hidden 256 \
  --patience 20
```

### Paso 3: Evaluar Nuevo Modelo
```bash
# Usar multi_horizon_inference con artifacts_v2
python example_multi_horizon.py \
  --model artifacts_v2/model_best.pt \
  --meta artifacts_v2/meta.json \
  --scaler artifacts_v2/scaler.pkl
```

### Paso 4: Comparar v1 vs v2
```python
# Ver training history
import pandas as pd
history = pd.read_csv("artifacts_v2/training_history.csv")
print(history.tail())

# Comparar métricas finales
print(f"Final val dir_acc: {history['val_dir_acc'].iloc[-1]:.2f}%")
print(f"Final val corr: {history['val_corr'].iloc[-1]:.4f}")
```

---

## 📊 Monitoreo Durante Entrenamiento

El script muestra cada 10 epochs:
```
Epoch 10/100
  Train Loss: 0.000425 | Val Loss: 0.000512
  Train Dir Acc: 58.2% | Val Dir Acc: 56.8%
  Train Var Ratio: 0.4523 | Val Var Ratio: 0.4198
  Train Corr: 0.4872 | Val Corr: 0.4534
  → Best model saved (dir_acc=56.8%)
```

**Señales de éxito:**
- ✅ Dir Acc increasing hacia >60%
- ✅ Var Ratio increasing hacia >0.5
- ✅ Corr increasing hacia >0.5
- ✅ Train/Val no divergen demasiado (no overfitting)

**Señales de problemas:**
- ❌ Dir Acc stuck around 50-52% → model not learning patterns
- ❌ Var Ratio stuck around 0.1-0.2 → still hedging
- ❌ Corr stuck around 0.2 → predictions not correlating
- ❌ Train/Val diverging → overfitting (increase dropout)

---

## 🔧 Troubleshooting

### Problema 1: Training muy lento
**Solución:** Reducir batch size o usar GPU
```bash
python retrain_clean_features.py --batch-size 128
```

### Problema 2: Overfitting (train/val diverge)
**Solución:** Aumentar dropout o reducir hidden size
```bash
python retrain_clean_features.py --dropout 0.3 --hidden 96
```

### Problema 3: Underfitting (métricas no mejoran)
**Solución:** Aumentar capacity o epochs
```bash
python retrain_clean_features.py --hidden 256 --layers 4 --epochs 200
```

### Problema 4: Features v2 no se computan correctamente
**Solución:** Verificar que add_technical_features_v2() funciona:
```python
from fiboevo import add_technical_features_v2
import numpy as np

# Test con datos sintéticos
close = np.random.randn(1000).cumsum() + 100
df = add_technical_features_v2(close)
print(df.columns)  # Debe tener 14 features
print(df.isnull().sum())  # Verificar NaNs
```

---

## 📈 Próximos Pasos Opcionales

### 1. Multi-Horizon Architecture
Implementar h=12 cabezas separadas para predicciones densas:
```python
class LSTM12Head(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, ...)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 2)  # (ret, vol) para cada horizonte
            for _ in range(12)
        ])
```

### 2. Attention Mechanism
Agregar attention para que el modelo aprenda qué timesteps son más relevantes:
```python
class LSTM2HeadWithAttention(nn.Module):
    ...
    self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
```

### 3. Ensemble de Modelos
Entrenar múltiples modelos con seeds diferentes y promediar predicciones:
```bash
for seed in 42 43 44 45 46; do
    python retrain_clean_features.py --seed $seed --output-dir artifacts_v2_seed_$seed
done
```

### 4. Data Augmentation
Entrenar con múltiples símbolos o timeframes:
- BTCUSDT 1h + ETHUSDT 1h
- BTCUSDT 1h + BTCUSDT 4h resampled

---

## 📝 Checklist de Validación

Antes de usar el modelo v2 en producción:

- [ ] Dir Acc >60% en validation
- [ ] Var Ratio >0.5 (predicciones con varianza realista)
- [ ] Corr >0.5 (predicciones correlacionadas con realidad)
- [ ] MAE improvement >15% vs persistence
- [ ] No divergencia train/val (no overfitting)
- [ ] Plots muestran predicciones alineadas con realidad
- [ ] Backtesting en período out-of-sample
- [ ] Stable performance en diferentes market regimes

---

## 🎓 Lecciones Aprendidas

1. **Data leakage es sutil:** Features con correlación >0.95 con target son red flags
2. **Directional accuracy importa más que MSE:** Predecir dirección correcta > magnitud exacta
3. **Variance matching previene hedging:** Modelo debe ser "valiente" con predicciones
4. **Loss function es crítica:** MSE solo no es suficiente para time series
5. **Temporal split es esencial:** Random split leak información del futuro
6. **Feature engineering > model complexity:** 14 features limpias > 39 con leakage
7. **Monitor múltiples métricas:** No confiar solo en loss, ver dir_acc, corr, var_ratio

---

## 📚 Referencias

- Diagnóstico completo: `model_diagnostics.png`
- Código de features: `fiboevo.py` línea 569-727
- Código de loss: `fiboevo.py` línea 874-971
- Script de entrenamiento: `retrain_clean_features.py`
- Herramienta de diagnóstico: `diagnose_model.py`

---

**Autor:** Claude (Anthropic)
**Fecha:** 2025-10-15
**Versión:** 2.0
