# Instalación con CUDA 11.8 - BitCorn Farmer

Guía paso a paso para configurar el entorno con **PyTorch CUDA 11.8** y **Python 3.10.11**.

---

## 📋 **Requisitos Previos**

### 1. **Python 3.10.11**
Verifica tu versión:
```bash
py -3.10 --version
```
Debe mostrar: `Python 3.10.11`

### 2. **NVIDIA GPU con CUDA 11.8**
Verifica que tienes una GPU NVIDIA:
```bash
nvidia-smi
```

Debe mostrar:
- **CUDA Version**: 11.8 o superior (12.x es compatible con 11.8)
- **GPU Name**: Tu modelo de GPU (ej: RTX 3060, GTX 1660, etc.)

---

## 🚀 **Instalación Paso a Paso**

### **Paso 1: Desinstalar PyTorch anterior (si existe)**

```bash
pip uninstall torch torchvision torchaudio -y
```

### **Paso 2: Instalar dependencias base**

```bash
pip install -r requirements.txt
```

Esto instalará:
- numpy 1.24+
- pandas 2.0+
- scikit-learn 1.3+
- matplotlib 3.7+
- ccxt, joblib, tqdm, etc.

### **Paso 3: Instalar PyTorch con CUDA 11.8**

**Para Windows:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**Para Linux:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**Descarga estimada:** ~2.5 GB

### **Paso 4: Verificar instalación**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Salida esperada:**
```
PyTorch: 2.1.2+cu118
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3060  (o tu modelo)
```

---

## ✅ **Verificación Completa**

### **Test 1: Operación básica en GPU**

```python
import torch

# Crear tensor en GPU
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()

# Operación en GPU
z = torch.matmul(x, y)

print(f"Tensor device: {z.device}")  # Debe mostrar: cuda:0
print("✓ GPU operations working!")
```

### **Test 2: Verificar que TradeApp detecta CUDA**

```bash
py -3.10 TradeApp.py
```

En la GUI:
1. Ve a **Training tab**
2. Click en "Prepare Data"
3. Click en "Train Model"

Deberías ver en el log:
```
✓ CUDA detected: NVIDIA GeForce RTX XXXX
Using device: cuda:0
```

---

## 🔧 **Solución de Problemas**

### **Problema 1: "CUDA available: False"**

**Causa:** PyTorch instalado sin soporte CUDA (versión CPU).

**Solución:**
```bash
# 1. Desinstalar PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. Reinstalar con CUDA 11.8 (copia el comando completo)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 3. Verificar nuevamente
python -c "import torch; print(torch.cuda.is_available())"
```

### **Problema 2: "DLL load failed" o "cudnn64_8.dll not found"**

**Causa:** Drivers NVIDIA desactualizados o corruptos.

**Solución:**
1. Descargar drivers más recientes: https://www.nvidia.com/Download/index.aspx
2. Instalar drivers
3. Reiniciar PC
4. Verificar con `nvidia-smi`

### **Problema 3: "torch.version.cuda returns None"**

**Causa:** Instalaste la versión CPU de PyTorch.

**Solución:**
```bash
# Verifica que el comando de instalación incluya --index-url
pip show torch
# Si "Location" no contiene "cu118", reinstalar con el comando del Paso 3
```

### **Problema 4: Versión de CUDA incorrecta**

**Causa:** Instalaste PyTorch para CUDA 12.1 pero tienes CUDA 11.8.

**Solución:**
```bash
# Desinstalar
pip uninstall torch torchvision torchaudio -y

# Reinstalar específicamente para CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

### **Problema 5: "RuntimeError: CUDA out of memory"**

**Causa:** GPU sin suficiente VRAM para el batch size.

**Solución:**
En TradeApp:
- Training tab → batch_size: Reducir de 64 a 32 o 16
- O usar MSE loss en vez de heteroscedastic_nll (usa menos memoria)

---

## 📊 **Mejoras de Rendimiento**

### **Con CUDA vs CPU**

| Operación | CPU (i7-10700) | GPU (RTX 3060) | Speedup |
|-----------|----------------|----------------|---------|
| Training (50 epochs) | ~45 min | ~3 min | **15x** |
| Inference (1000 pred) | ~2 s | ~0.15 s | **13x** |
| Data preparation | ~5 s | ~5 s | 1x (CPU-bound) |

### **VRAM Requerida**

| Configuración | VRAM Estimada |
|---------------|---------------|
| Single-horizon (h=6, batch=64) | ~1.5 GB |
| Multi-horizon (5 horizons, batch=64) | ~2.5 GB |
| Multi-horizon (5 horizons, batch=32) | ~1.5 GB |
| Multi-horizon (10 horizons, batch=64) | ~4.0 GB |

**GPU Recomendadas:**
- **Mínimo:** GTX 1660 (6 GB VRAM)
- **Recomendado:** RTX 3060 (12 GB VRAM)
- **Ideal:** RTX 4070+ (12+ GB VRAM)

---

## 🎯 **Configuración Recomendada**

### **Para GPUs de 6 GB VRAM:**
```python
# En TradeApp Training tab:
batch_size = 32
seq_len = 128
horizons = "1,3,6,12,24"  # 5 horizons
loss_type = "mse"  # Más eficiente en memoria
```

### **Para GPUs de 8+ GB VRAM:**
```python
# En TradeApp Training tab:
batch_size = 64
seq_len = 128
horizons = "1,3,6,12,24"
loss_type = "heteroscedastic_nll"  # Mejor performance
```

### **Para GPUs de 12+ GB VRAM:**
```python
# Configuración óptima:
batch_size = 128
seq_len = 256
horizons = "1,2,3,6,12,18,24"  # 7 horizons
loss_type = "heteroscedastic_nll"
hidden_size = 256  # Mayor capacidad
```

---

## 📝 **Comandos Rápidos de Referencia**

```bash
# Verificar versión de Python
py -3.10 --version

# Verificar GPU
nvidia-smi

# Instalar dependencias
pip install -r requirements.txt

# Instalar PyTorch CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Verificar PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Listar paquetes instalados
pip list | grep torch
```

---

## 🆘 **Soporte**

Si después de seguir esta guía sigues teniendo problemas:

1. **Verifica versiones instaladas:**
   ```bash
   pip list | grep -E "(torch|numpy|pandas|scikit)"
   ```

2. **Copia la salida completa del comando de verificación:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); import sys; print(f'Python: {sys.version}')"
   ```

3. **Comparte la salida en el issue tracker o con soporte**

---

**Fecha:** 2025-11-01
**Versión:** 1.0
**Compatible con:** Python 3.10.11, CUDA 11.8, PyTorch 2.1.2
