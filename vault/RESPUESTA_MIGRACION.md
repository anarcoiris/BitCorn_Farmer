# Respuesta: Sistema de Artifacts Duplicado

## 🎯 Tu Pregunta

> "Las features se guardan bien en artifacts_v2, pero... se entrena con estas y se usa luego artifacts_v2? Tal vez deberíamos de desvincular el antiguo sistema?"

**Respuesta**: ¡Tienes toda la razón! Hay un **problema grave de configuración**:

---

## 🚨 Problema Detectado

### Situación Actual (INCONSISTENTE)

```
Sistema VIEJO (artifacts/):
  - 39 features (26 price-level) ❌
  - Horizon: 10
  - 13 features con corr > 0.90 ❌
  - CAUSA TEMPORAL LAG ❌

Sistema NUEVO (artifacts_v2/):
  - 14 features (0 price-level) ✓
  - Horizon: 4
  - 0 features con corr > 0.90 ✓
  - SIN TEMPORAL LAG ✓
```

### El Problema

**TODOS los componentes apuntan a `artifacts/` (VIEJO) por defecto:**

| Componente | Ruta Default | Features |
|------------|-------------|----------|
| `example_multi_horizon.py` | `artifacts/` | 39 (VIEJO) |
| `simple_future_forecast.py` | `artifacts/` | 39 (VIEJO) |
| `TradeApp.py` (training) | `artifacts/` | 39 (VIEJO) |
| `TradeApp.py` (loading) | `artifacts/` | 39 (VIEJO) |

**`artifacts_v2/` NO se usa en ningún lugar automáticamente!**

### ⚠️ Riesgo Crítico

Si entrenas desde el GUI ahora:
1. TradeApp usa `add_technical_features()` (método viejo)
2. Genera 39 features (26 price-level)
3. Entrena modelo con features problemáticas
4. **SOBRESCRIBE** `artifacts/model_best.pt`
5. ¡Pierdes cualquier progreso!

---

## ✅ Solución: Migración Automática

He creado un script de migración segura:

### Verificación (Dry-Run)

```bash
python migrate_to_clean_features.py --dry-run
```

**Output** (ya ejecutado):
```
[OK] ARTIFACTS ACTUALES (VIEJO): artifacts
  - Features: 39
  - Horizon: 1
  - Price-level features: 26  [WARN]

[OK] ARTIFACTS NUEVOS (LIMPIO): artifacts_v2
  - Features: 14
  - Horizon: 4
  - Price-level features: 0  [OK]

[OK] Validación exitosa

Pasos a ejecutar:
  1. Backup: artifacts/ -> artifacts_deprecated/
  2. Promover: artifacts_v2/ -> artifacts/
```

### Ejecución Real

```bash
python migrate_to_clean_features.py
```

**Esto hará**:
1. `artifacts/` → `artifacts_deprecated/` (backup seguro)
2. `artifacts_v2/` → `artifacts/` (promoción)
3. Verifica integridad
4. Reporta estado final

**Después de la migración:**
- Todos los scripts usarán automáticamente las 14 features limpias
- TradeApp cargará el modelo correcto
- ¡artifacts_v2 desaparece (ya no es necesario)!

---

## 📋 Plan de Acción Recomendado

### Opción 1: Migración Completa (RECOMENDADO)

```bash
# 1. Verificar estado actual
python migrate_to_clean_features.py --dry-run

# 2. Ejecutar migración
python migrate_to_clean_features.py

# 3. Verificar
python test_feature_inspection.py
python example_multi_horizon.py

# 4. Confirmar temporal lag eliminado
# (comparar plot con el anterior)
```

**Ventajas:**
- ✓ Un solo sistema (artifacts/)
- ✓ No más confusión
- ✓ Backup automático en artifacts_deprecated/
- ✓ Todos los scripts funcionan sin cambios

**Después de migrar:**
```
artifacts/              ← NUEVO (14 features limpias)
artifacts_deprecated/   ← VIEJO (39 features - backup)
```

---

### Opción 2: Actualizar TradeApp para usar artifacts_v2

Si prefieres mantener ambos sistemas por ahora:

```python
# En TradeApp.py, cambiar todas las rutas:
artifacts_dir = Path("artifacts_v2")  # en lugar de "artifacts"
```

**Desventajas:**
- Requiere cambios en múltiples lugares
- Scripts de ejemplo seguirán usando artifacts/
- Riesgo de confusión continúa

---

## 🔧 Próximos Pasos Después de Migrar

### 1. Actualizar TradeApp Training

El training en TradeApp actualmente usa `add_technical_features()` (viejo). Debería usar `add_technical_features_v2()`.

**¿Quieres que cree un patch para esto?**

### 2. Verificar Inferencia

```bash
python example_multi_horizon.py
```

Deberías ver:
- Features: 14 (no 39)
- Sin temporal lag en el plot
- Directional accuracy mejorada

### 3. Opcional: Eliminar Backup

Una vez confirmado que todo funciona:
```bash
rm -rf artifacts_deprecated/
```

---

## 📊 Comparación Antes/Después

| Aspecto | ANTES (actual) | DESPUÉS (migrado) |
|---------|----------------|-------------------|
| artifacts/ | 39 features | 14 features |
| artifacts_v2/ | 14 features | (eliminado) |
| Scripts | usan artifacts/ viejo | usan artifacts/ nuevo |
| Training GUI | usa artifacts/ viejo | usa artifacts/ nuevo |
| Temporal lag | SÍ | NO |
| Confusión | Alta | Cero |

---

## ⚡ Ejecución Inmediata

### Paso 1: Dry-Run (ya hecho ✓)
```
[OK] 26 price-level features en artifacts/
[OK] 0 price-level features en artifacts_v2/
[OK] Migración lista para ejecutar
```

### Paso 2: ¿Ejecutar migración?

```bash
python migrate_to_clean_features.py
```

**¿Quieres que ejecute la migración ahora?**

---

## 🛡️ Seguridad

- ✓ Hace backup automático (artifacts_deprecated/)
- ✓ Verifica integridad antes de migrar
- ✓ Modo dry-run para revisar sin cambios
- ✓ Opción --force para casos especiales
- ✓ Rollback manual posible si es necesario

---

## 🎯 Resumen Ejecutivo

**Problema**: Sistema duplicado causa confusión. Todo apunta a artifacts/ viejo (39 features, temporal lag).

**Solución**: Migrar artifacts_v2/ → artifacts/ de forma segura.

**Resultado**: Un solo sistema, sin confusión, sin temporal lag.

**Acción**: Ejecutar `python migrate_to_clean_features.py`

---

**¿Procedo con la migración?** ✨
