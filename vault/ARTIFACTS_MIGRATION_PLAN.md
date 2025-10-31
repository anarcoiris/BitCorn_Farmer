# Plan de Migración: artifacts/ → artifacts_v2/

## 🚨 Problema Identificado

Actualmente hay **DOS sistemas en paralelo** que pueden causar confusión:

### Sistema VIEJO (`artifacts/`)
```
artifacts/
├── model_best.pt      (39 features, horizon=10)
├── meta.json          (26 price-level features ⚠️)
└── scaler.pkl

Características:
❌ 39 features (26 son price-level)
❌ 13 features con |corr| > 0.90
❌ Causa temporal lag
❌ Directional accuracy: ~56%
```

### Sistema NUEVO (`artifacts_v2/`)
```
artifacts_v2/
├── model_best.pt      (14 features, horizon=4)
├── meta.json          (0 price-level features ✓)
└── scaler.pkl

Características:
✓ 14 features (0 price-level)
✓ 0 features con |corr| > 0.90
✓ Sin temporal lag
✓ Val directional accuracy: 51.24%
```

---

## 🔍 Dónde Se Usa Cada Sistema

### Scripts que usan `artifacts/` (VIEJO)
```python
# example_multi_horizon.py
MODEL_PATH = "artifacts/model_best.pt"      # ❌ VIEJO

# simple_future_forecast.py
MODEL_PATH = "artifacts/model_best.pt"      # ❌ VIEJO

# example_future_predictions.py
MODEL_PATH = "artifacts/model_best.pt"      # ❌ VIEJO
```

### TradeApp.py usa `artifacts/` (VIEJO)
```python
# Línea 3143: Training guarda en artifacts/
artifacts_dir = Path("artifacts")           # ❌ VIEJO

# Línea 3242: Loading carga desde artifacts/
artifacts_dir = Path("artifacts")           # ❌ VIEJO
```

### ⚠️ **RIESGO**:
Si entrenas desde el GUI, **sobreescribirás** el modelo viejo en `artifacts/` con un **nuevo modelo que usa las mismas 39 features problemáticas**!

---

## ✅ Solución Propuesta

### Opción 1: Deprecar Completamente el Sistema Viejo (RECOMENDADO)

1. **Renombrar artifacts/ → artifacts_deprecated/**
2. **Renombrar artifacts_v2/ → artifacts/**
3. **Actualizar todos los scripts** para usar el nuevo artifacts/
4. **Actualizar TradeApp.py** para:
   - Entrenar con `add_technical_features_v2()`
   - Guardar en artifacts/
   - Cargar desde artifacts/

### Opción 2: Sistema de Configuración (Más flexible pero más complejo)

Agregar variable de configuración para elegir qué sistema usar:

```python
# config.json
{
  "artifact_version": "v2",  # o "v1"
  "artifact_paths": {
    "v1": "artifacts/",
    "v2": "artifacts_v2/"
  }
}
```

---

## 🛠️ Implementación Recomendada (Opción 1)

### Paso 1: Backup y Reorganización
```bash
# Backup del sistema viejo
mv artifacts artifacts_deprecated

# Promover el nuevo sistema
mv artifacts_v2 artifacts

# Verificar
ls artifacts/
# Debe mostrar: model_best.pt, meta.json, scaler.pkl (con 14 features)
```

### Paso 2: Actualizar Scripts

Ya no es necesario cambiar nada en los scripts - ahora apuntan automáticamente al sistema correcto!

### Paso 3: Actualizar TradeApp.py Training

Necesitamos que el training use `add_technical_features_v2()` en lugar de `add_technical_features()`.

---

## 📋 Checklist de Migración

- [ ] **Backup**: `cp -r artifacts artifacts_deprecated`
- [ ] **Reorganizar**:
  - [ ] `mv artifacts_v2 artifacts_new_temp`
  - [ ] `mv artifacts artifacts_old_temp`
  - [ ] `mv artifacts_new_temp artifacts`
  - [ ] (opcional) `rm -rf artifacts_old_temp`
- [ ] **Actualizar TradeApp.py** para usar `add_technical_features_v2()`
- [ ] **Probar entrenamiento** desde GUI
- [ ] **Probar inferencia** desde GUI
- [ ] **Probar scripts**:
  - [ ] `python example_multi_horizon.py` (debe usar 14 features)
  - [ ] `python simple_future_forecast.py` (debe usar 14 features)
  - [ ] `python test_feature_inspection.py` (debe reportar 0 price-level features)
- [ ] **Verificar temporal lag eliminado**

---

## 🎯 Estado Actual de Archivos

### Scripts de Ejemplo
| Script | Ruta Default | Features Usadas | Temporal Lag |
|--------|-------------|-----------------|--------------|
| `example_multi_horizon.py` | `artifacts/` | 39 (VIEJO) | ❌ SÍ |
| `simple_future_forecast.py` | `artifacts/` | 39 (VIEJO) | ❌ SÍ |
| `example_future_predictions.py` | `artifacts/` | 39 (VIEJO) | ❌ SÍ |

### GUI (TradeApp.py)
| Función | Ruta | Features | Temporal Lag |
|---------|------|----------|--------------|
| Training | `artifacts/` | 39 (VIEJO) | ❌ SÍ |
| Loading | `artifacts/` | 39 (VIEJO) | ❌ SÍ |
| Inference | `artifacts/` | 39 (VIEJO) | ❌ SÍ |

### ⚠️ **PROBLEMA**: Ningún componente usa `artifacts_v2/` por defecto!

---

## 🚀 Script de Migración Automática

Voy a crear un script para hacer la migración de forma segura...
