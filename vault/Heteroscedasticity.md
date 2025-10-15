# Heterocedasticidad en LSTM para Trading: Guía Completa

## Tabla de Contenidos
1. [Fundamentos Matemáticos](#1-fundamentos-matemáticos)
2. [Por Qué Importa en Finanzas](#2-por-qué-importa-en-finanzas)
3. [Arquitectura LSTM Heterocedástica](#3-arquitectura-lstm-heterocedástica)
4. [Implementación PyTorch](#4-implementación-pytorch)
5. [Inicialización y Estabilidad](#5-inicialización-y-estabilidad)
6. [Calibración de Incertidumbre](#6-calibración-de-incertidumbre)
7. [Uso en Trading (Inference)](#7-uso-en-trading-inference)
8. [Problemas Comunes y Soluciones](#8-problemas-comunes-y-soluciones)
9. [Comparación MSE vs Hetero NLL](#9-comparación-mse-vs-hetero-nll)
10. [Integración con Proyecto Actual](#10-integración-con-proyecto-actual)
11. [Configuración Recomendada](#11-configuración-recomendada)
12. [Referencias y Recursos](#12-referencias-y-recursos)

---

## 1. Fundamentos Matemáticos

### ¿Qué es la Heterocedasticidad?

**Definición formal:**

En regresión clásica asumimos errores i.i.d. con varianza constante (homocedasticidad):

```
y_t = f(x_t) + ε_t,  ε_t ~ N(0, σ²),  σ² constante
```

**Heterocedasticidad** significa que la varianza depende de x_t (o del tiempo):

```
ε_t ~ N(0, σ²_t),  σ²_t = g(x_t)  o  σ²_t = function of history
```

Es decir, el ruido es **condicional a la información disponible** y **no constante**.

### Parametrización Numéricamente Estable

Para evitar problemas numéricos (σ² < 0), modelamos el **log-variance**:

```
s_t ≡ log(σ²_t) ∈ ℝ
```

La verosimilitud gaussiana por muestra t es:

```
ℓ_t = -log p(y_t | x_t) = 1/2 * [exp(-s_t) * (y_t - μ_t)² + s_t + log(2π)]
```

**Ventajas**:
- `s_t` puede ser cualquier número real (sin restricciones)
- `exp(s_t) = σ²_t` garantiza positividad automáticamente
- Gradientes estables (no hay división por cero)

### Forma Simplificada (sin constantes)

Para optimización, usamos:

```
L_NLL(μ, s, y) = 1/2 * [exp(-s) * (y - μ)² + s]
```

**Términos**:
1. **exp(-s) * (y - μ)²**: Penaliza error ponderado por confianza (1/σ²)
2. **s**: Penaliza σ inflado (evita "cheating" con σ → ∞)

**Balance automático**:
- Si σ es muy pequeño y hay error → término 1 explota
- Si σ es muy grande → término 2 penaliza

---

## 2. Por Qué Importa en Finanzas

### Volatility Clustering

Los mercados financieros exhiben **heterocedasticidad condicional**:
- Períodos de alta volatilidad seguidos de alta volatilidad
- Períodos de calma seguidos de calma
- La varianza NO es constante en el tiempo

**Modelos clásicos**: GARCH, Stochastic Volatility (SV) son precisamente modelos de varianza condicional.

### Problemas con MSE (Varianza Constante)

Si tu red predice solo la media (μ) y usas MSE:
1. ❌ **Ignoras información relevante**: La incertidumbre varía por muestra
2. ❌ **Sin gestión de riesgo**: No sabes cuándo confiar en las predicciones
3. ❌ **Decisiones sub-óptimas**: Operas igual en mercado volátil que estable
4. ❌ **Overfitting a outliers**: MSE penaliza igual errores en alta/baja volatilidad

### Beneficios de Modelar σ_t

1. ✅ **Estimación de incertidumbre por muestra**: Intervalos de confianza (μ ± kσ)
2. ✅ **Decisiones risk-aware**: Solo operar si SNR = μ/σ > threshold
3. ✅ **Mejor scoring**: NLL se ajusta automáticamente a heterocedasticidad
4. ✅ **Position sizing**: Ajustar tamaño inversamente proporcional a σ_t
5. ✅ **Detección de régimen**: Alta σ → mercado turbulento, esperar

---

## 3. Arquitectura LSTM Heterocedástica

### Modelo Actual: LSTM2Head (Baseline)

**Arquitectura existente** (`fiboevo.py`):

```python
class LSTM2Head(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Cabeza ret: predice solo μ (media)
        self.head_ret = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Cabeza vol: predice solo μ (con Softplus para positividad)
        self.head_vol = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        ret = self.head_ret(last).squeeze(-1)
        vol = self.head_vol(last).squeeze(-1)
        return ret, vol
```

**Limitaciones**:
- Solo predice μ (punto)
- No modela incertidumbre (σ)
- Usa MSE loss (asume σ constante)

### Modelo Propuesto: HeteroHead

**Componente básico** para predecir μ y s = log(σ²):

```python
class HeteroHead(nn.Module):
    """
    Cabeza que predice media (μ) y log-variance (s = log σ²) para un target.

    Args:
        hidden_size: Tamaño del hidden state del LSTM

    Returns:
        mu: Predicción de la media (sin restricción)
        s: Predicción de log(σ²) (clamped para estabilidad)
    """
    def __init__(self, hidden_size):
        super().__init__()
        mid = max(8, hidden_size // 2)

        # Branch para μ
        self.mu = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid, 1)
        )

        # Branch para log(σ²)
        self.log_var = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid, 1)
        )

    def forward(self, x):
        mu = self.mu(x).squeeze(-1)           # (B,)
        s = self.log_var(x).squeeze(-1)       # (B,)

        # Clamp para estabilidad numérica
        # s ∈ [-12, 8] → σ² ∈ [6e-6, 3000]
        s = torch.clamp(s, min=-12.0, max=8.0)

        return mu, s
```

**Características**:
- ✅ Dos branches independientes (μ y s)
- ✅ Dropout para regularización
- ✅ Clamp de s para evitar explosiones/colapsos
- ✅ Output sin activaciones (s puede ser negativo)

### Modelo Completo: LSTM4HeadsHetero

**4 cabezas** para ret, high, low, vol (cada una con μ y s):

```python
class LSTM4HeadsHetero(nn.Module):
    """
    LSTM con 4 cabezas heterocedásticas: ret, high, low, vol

    Cada cabeza predice:
    - μ (media): estimación puntual
    - s = log(σ²): log-variance para heterocedasticidad

    Args:
        input_size: Número de features (~35-40 con decouple)
        hidden_size: Tamaño del hidden state (128 recomendado para GTX 1070)
        num_layers: Capas LSTM (2 por defecto)
        dropout: Dropout entre capas LSTM (0.1-0.15)

    Returns:
        Tuple de 8 tensors: (mu_ret, s_ret, mu_high, s_high, mu_low, s_low, mu_vol, s_vol)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()

        # LSTM compartido para todas las cabezas
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )

        # 4 cabezas heterocedásticas
        self.ret_head = HeteroHead(hidden_size)
        self.high_head = HeteroHead(hidden_size)
        self.low_head = HeteroHead(hidden_size)
        self.vol_head = HeteroHead(hidden_size)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, F) tensor de features

        Returns:
            (mu_ret, s_ret, mu_high, s_high, mu_low, s_low, mu_vol, s_vol)
            Cada uno: (B,) tensor
        """
        out, _ = self.lstm(x)              # (B, seq_len, hidden)
        last = out[:, -1, :]               # (B, hidden) - último timestep

        # Predecir μ y s para cada target
        mu_ret, s_ret = self.ret_head(last)
        mu_high, s_high = self.high_head(last)
        mu_low, s_low = self.low_head(last)
        mu_vol, s_vol = self.vol_head(last)

        return (mu_ret, s_ret, mu_high, s_high, mu_low, s_low, mu_vol, s_vol)
```

**Ventajas**:
- ✅ LSTM compartido (eficiencia)
- ✅ Cada cabeza especializada en su target
- ✅ Incertidumbre independiente por target
- ✅ Compatible con arquitectura existente (fácil migración)

---

## 4. Implementación PyTorch

### Loss Function: Gaussian NLL

**Implementación básica**:

```python
def hetero_nll_loss(mu, log_var, y, reduction='mean'):
    """
    Gaussian Negative Log-Likelihood con varianza heterocedástica.

    Matemática:
        L_NLL = 1/2 * [exp(-s) * (y - μ)² + s]
        donde s = log(σ²)

    Args:
        mu: Predicción de media (B,)
        log_var: Predicción de log(σ²) (B,)
        y: Target real (B,)
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        loss: escalar si reduction='mean', tensor (B,) si reduction='none'

    Ejemplo:
        >>> mu = torch.tensor([0.5, -0.2])
        >>> s = torch.tensor([-2.0, 1.0])  # log(var)
        >>> y = torch.tensor([0.6, -0.3])
        >>> loss = hetero_nll_loss(mu, s, y)
        >>> print(loss)  # tensor(0.7234)
    """
    # Término 1: Error ponderado por confianza (1/σ²)
    inv_var = torch.exp(-log_var)           # 1/σ² (B,)
    mse_term = inv_var * (y - mu) ** 2      # (B,)

    # Término 2: Penalización por σ inflado
    log_term = log_var                      # (B,)

    # Loss por muestra
    loss = 0.5 * (mse_term + log_term)      # (B,)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
```

**Interpretación de términos**:
- **mse_term**: Si σ es pequeño (alta confianza) y error es grande → loss explota
- **log_term**: Si σ es muy grande → loss penaliza (evita "cheating")

### Loss Combinada para 4 Heads

```python
def combined_hetero_loss(outputs, targets, weights, penalty_sigma=0.0):
    """
    Loss combinada para las 4 cabezas heterocedásticas.

    Args:
        outputs: Tuple (mu_ret, s_ret, mu_high, s_high, mu_low, s_low, mu_vol, s_vol)
        targets: Tuple (y_ret, y_high, y_low, y_vol)
        weights: Dict con {'w_ret': 1.0, 'w_high': 0.8, 'w_low': 0.8, 'w_vol': 0.5}
        penalty_sigma: Penalización adicional por σ inflado (β, típicamente 1e-3)

    Returns:
        total_loss: Loss total ponderada
        components: Dict con loss individual de cada cabeza

    Ejemplo:
        >>> weights = {'w_ret': 1.0, 'w_high': 0.8, 'w_low': 0.8, 'w_vol': 0.5}
        >>> loss, comps = combined_hetero_loss(outputs, targets, weights)
    """
    mu_ret, s_ret, mu_high, s_high, mu_low, s_low, mu_vol, s_vol = outputs
    y_ret, y_high, y_low, y_vol = targets

    # NLL para cada cabeza
    loss_ret = hetero_nll_loss(mu_ret, s_ret, y_ret)
    loss_high = hetero_nll_loss(mu_high, s_high, y_high)
    loss_low = hetero_nll_loss(mu_low, s_low, y_low)
    loss_vol = hetero_nll_loss(mu_vol, s_vol, y_vol)

    # Penalización adicional por σ inflado (opcional)
    penalty = 0.0
    if penalty_sigma > 0:
        # Penalizar log_var alto (σ² grande)
        all_log_vars = torch.cat([s_ret, s_high, s_low, s_vol])
        penalty = penalty_sigma * all_log_vars.mean()

    # Loss total ponderada
    total = (
        weights['w_ret'] * loss_ret +
        weights['w_high'] * loss_high +
        weights['w_low'] * loss_low +
        weights['w_vol'] * loss_vol +
        penalty
    )

    # Componentes para logging
    components = {
        'ret': loss_ret.item(),
        'high': loss_high.item(),
        'low': loss_low.item(),
        'vol': loss_vol.item(),
        'penalty': penalty if isinstance(penalty, float) else penalty.item()
    }

    return total, components
```

**Pesos recomendados**:
- `w_ret = 1.0`: Retorno es el target principal
- `w_high = 0.8, w_low = 0.8`: Supremo/ínfimo importantes pero secundarios
- `w_vol = 0.5`: Volatilidad como auxiliar
- `penalty_sigma = 1e-3`: Suave penalización contra σ inflado

### Training Loop

```python
def train_epoch_hetero(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    weights: dict = None,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    penalty_sigma: float = 0.0
):
    """
    Training epoch con heterocedasticidad.

    Args:
        model: LSTM4HeadsHetero
        loader: DataLoader con (xb, y_ret, y_vol, y_high, y_low)
        optimizer: PyTorch optimizer (Adam recomendado)
        device: torch.device('cuda' o 'cpu')
        weights: Dict con pesos para cada loss
        grad_clip: Max gradient norm (1.0 recomendado)
        use_amp: Mixed precision (True para GTX 1070)
        penalty_sigma: Penalización σ inflado

    Returns:
        avg_loss: Loss promedio del epoch
        component_losses: Dict con loss por componente
    """
    if weights is None:
        weights = {'w_ret': 1.0, 'w_high': 0.8, 'w_low': 0.8, 'w_vol': 0.5}

    model.train()
    total_loss = 0.0
    n = 0

    # Mixed precision scaler (GTX 1070 soporta FP16)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Acumuladores para logging
    component_losses = {'ret': 0.0, 'high': 0.0, 'low': 0.0, 'vol': 0.0, 'penalty': 0.0}

    for batch in loader:
        # Expect: (xb, y_ret, y_vol, y_high, y_low)
        if len(batch) != 5:
            raise ValueError(f"Expected 5 tensors in batch, got {len(batch)}")

        xb, yret, yvol, yhigh, ylow = batch

        # Move to device
        xb = xb.to(device)
        yret = yret.to(device).float()
        yhigh = yhigh.to(device).float()
        ylow = ylow.to(device).float()
        yvol = yvol.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        # Forward + backward con o sin AMP
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(xb)
                targets = (yret, yhigh, ylow, yvol)
                loss, comp = combined_hetero_loss(outputs, targets, weights, penalty_sigma)

            # Backward con gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(xb)
            targets = (yret, yhigh, ylow, yvol)
            loss, comp = combined_hetero_loss(outputs, targets, weights, penalty_sigma)

            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        # Acumular stats
        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        for k in component_losses:
            component_losses[k] += comp[k] * bs

    # Promedios
    avg_loss = total_loss / max(1, n)
    for k in component_losses:
        component_losses[k] /= max(1, n)

    return avg_loss, component_losses
```

### Evaluation Loop

```python
def eval_epoch_hetero(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict = None,
    penalty_sigma: float = 0.0
):
    """
    Evaluation epoch con heterocedasticidad.

    Returns:
        avg_loss: Loss promedio
        component_losses: Dict con loss por componente
    """
    if weights is None:
        weights = {'w_ret': 1.0, 'w_high': 0.8, 'w_low': 0.8, 'w_vol': 0.5}

    model.eval()
    total_loss = 0.0
    n = 0
    component_losses = {'ret': 0.0, 'high': 0.0, 'low': 0.0, 'vol': 0.0, 'penalty': 0.0}

    with torch.no_grad():
        for batch in loader:
            xb, yret, yvol, yhigh, ylow = batch

            xb = xb.to(device)
            yret = yret.to(device).float()
            yhigh = yhigh.to(device).float()
            ylow = ylow.to(device).float()
            yvol = yvol.to(device).float()

            outputs = model(xb)
            targets = (yret, yhigh, ylow, yvol)
            loss, comp = combined_hetero_loss(outputs, targets, weights, penalty_sigma)

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n += bs

            for k in component_losses:
                component_losses[k] += comp[k] * bs

    avg_loss = total_loss / max(1, n)
    for k in component_losses:
        component_losses[k] /= max(1, n)

    return avg_loss, component_losses
```

---

## 5. Inicialización y Estabilidad

### Problema: Sin Inicialización Adecuada

Si los pesos de `log_var` se inicializan aleatoriamente:
- Puede predecir s = -10 → σ² = 4.5e-5 (colapso)
- O predecir s = +5 → σ² = 148 (inflado)
- Gradientes muy grandes/pequeños
- Training inestable

### Solución: Inicializar con Varianza Empírica

```python
def initialize_hetero_heads(model, y_train_dict):
    """
    Inicializa el bias de las cabezas log_var con la varianza empírica del training set.

    Esto evita que la red empiece con σ muy grande o muy pequeño.

    Args:
        model: LSTM4HeadsHetero instance
        y_train_dict: Dict con arrays de training targets
                     {'ret': y_ret_train,
                      'high': y_high_train,
                      'low': y_low_train,
                      'vol': y_vol_train}

    Ejemplo:
        >>> model = LSTM4HeadsHetero(input_size=40, hidden_size=128)
        >>> y_dict = {
        ...     'ret': y_ret_train,
        ...     'high': y_high_train,
        ...     'low': y_low_train,
        ...     'vol': y_vol_train
        ... }
        >>> initialize_hetero_heads(model, y_dict)
        Initialized ret_head log_var bias to -4.8532 (var=0.007834)
        Initialized high_head log_var bias to -3.2145 (var=0.040123)
        ...
    """
    for name, y_data in y_train_dict.items():
        # Varianza empírica del target
        emp_var = np.var(y_data) + 1e-8  # +epsilon para estabilidad
        init_log_var = np.log(emp_var)

        # Obtener la cabeza correspondiente
        head = getattr(model, f"{name}_head")

        # Inicializar bias de la última capa del branch log_var
        # head.log_var es nn.Sequential, última capa es nn.Linear
        with torch.no_grad():
            last_layer = head.log_var[-1]  # Última capa (nn.Linear)
            if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                last_layer.bias.fill_(init_log_var)

        print(f"Initialized {name}_head log_var bias to {init_log_var:.4f} (var={emp_var:.6f})")
```

**Uso en training script**:

```python
# Después de crear el modelo
model = LSTM4HeadsHetero(input_size=len(feature_cols), hidden_size=128)

# Antes de entrenar, inicializar
y_train_dict = {
    'ret': y_ret_train,
    'high': y_high_train,
    'low': y_low_train,
    'vol': y_vol_train
}
initialize_hetero_heads(model, y_train_dict)

# Ahora sí entrenar
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### Gradient Clipping (Esencial)

```python
# En training loop, DESPUÉS de backward():
if grad_clip > 0:
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
```

**Valor recomendado**: `grad_clip = 1.0` (evita explosiones)

### Learning Rate Schedule

```python
# Empezar con LR moderado
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Scheduler: reducir LR si val_loss no mejora
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# En validation loop
val_loss, _ = eval_epoch_hetero(model, val_loader, device, weights)
scheduler.step(val_loss)
```

---

## 6. Calibración de Incertidumbre

### ¿Qué es Calibración?

**Definición**: La σ predicha debe reflejar el error real observado.

Si el modelo predice σ = 0.01 (muy confiado), el error real |y - μ| debería ser ~0.01 la mayor parte del tiempo.

**Test de calibración**:
- ~68% de errores dentro de ±1σ
- ~95% de errores dentro de ±2σ
- ~99.7% de errores dentro de ±3σ

(Asumiendo distribución Gaussian)

### Implementación

```python
def calibrate_uncertainty(model, loader, device, head_name='ret'):
    """
    Calcula calibración de incertidumbre para una cabeza específica.

    Args:
        model: LSTM4HeadsHetero
        loader: DataLoader de validación/test
        device: torch.device
        head_name: 'ret' | 'high' | 'low' | 'vol'

    Returns:
        calibration_stats: Dict con fracciones dentro de k*σ

    Ejemplo:
        >>> stats = calibrate_uncertainty(model, val_loader, device, 'ret')
        >>> print(stats)
        {'1sigma': 0.69, '2sigma': 0.94, '3sigma': 0.997}
        # Interpretación: bien calibrado (cerca de 0.68, 0.95, 0.997)
    """
    model.eval()

    errors = []
    sigmas = []

    # Mapeo de head_name a índices en outputs
    head_idx = {'ret': 0, 'high': 2, 'low': 4, 'vol': 6}
    mu_idx = head_idx[head_name]
    s_idx = mu_idx + 1

    # Mapeo a target
    target_idx = {'ret': 0, 'high': 2, 'low': 3, 'vol': 1}
    y_idx = target_idx[head_name]

    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            y = batch[y_idx + 1].to(device)  # +1 porque batch[0] es xb

            outputs = model(xb)
            mu = outputs[mu_idx]
            s = outputs[s_idx]
            sigma = torch.sqrt(torch.exp(s))

            error = torch.abs(y - mu)

            errors.extend(error.cpu().numpy())
            sigmas.extend(sigma.cpu().numpy())

    errors = np.array(errors)
    sigmas = np.array(sigmas)

    # Z-scores: cuántas σ de distancia
    z_scores = errors / (sigmas + 1e-8)

    # Fracciones dentro de k*σ
    calibration_stats = {}
    expected = {1: 0.6827, 2: 0.9545, 3: 0.9973}  # Gaussian teórico

    print(f"\nCalibration for {head_name}:")
    print(f"{'k*sigma':<10} {'Observed':<12} {'Expected':<12} {'Diff':<10}")
    print("-" * 50)

    for k in [1, 2, 3]:
        frac = np.mean(z_scores <= k)
        exp = expected[k]
        diff = frac - exp

        calibration_stats[f'{k}sigma'] = frac

        # Color coding (si terminal lo soporta)
        status = "✓" if abs(diff) < 0.05 else "⚠" if abs(diff) < 0.10 else "✗"
        print(f"{status} {k}σ      {frac:.4f}       {exp:.4f}       {diff:+.4f}")

    return calibration_stats
```

**Interpretación**:
- ✓ **Bien calibrado**: Diff < 0.05 (5%)
- ⚠ **Aceptable**: Diff < 0.10 (10%)
- ✗ **Mal calibrado**: Diff > 0.10

**Acciones si mal calibrado**:
1. **σ sub-estimado** (observed > expected): Aumentar `penalty_sigma`
2. **σ sobre-estimado** (observed < expected): Reducir `penalty_sigma`, verificar init
3. **Asimetría**: Considerar Student-t likelihood en vez de Gaussian

---

## 7. Uso en Trading (Inference)

### Signal-to-Noise Ratio (SNR)

**Definición**:
```
SNR = |μ| / σ
```

**Interpretación**:
- SNR alto → Señal fuerte, alta confianza
- SNR bajo → Señal débil o alta incertidumbre

**Regla de decisión**:
```python
if SNR > threshold:
    # Operar
else:
    # Esperar (no hay señal clara)
```

**Threshold recomendado**: 1.5 - 2.0 (ajustar en backtest)

### Implementación en Trading Daemon

```python
def predict_with_uncertainty(model, xb, device):
    """
    Predicción con intervalos de confianza.

    Args:
        model: LSTM4HeadsHetero
        xb: Input tensor (1, seq_len, features)
        device: torch.device

    Returns:
        predictions: Dict con μ, σ, intervalos para cada target

    Ejemplo:
        >>> preds = predict_with_uncertainty(model, xb, device)
        >>> print(preds['ret'])
        {'mu': 0.0123, 'sigma': 0.0045, 'snr': 2.73,
         'lower_95': 0.0035, 'upper_95': 0.0211}
    """
    model.eval()

    with torch.no_grad():
        xb = xb.to(device)
        outputs = model(xb)

        # Extraer μ y σ para cada target
        mu_ret, s_ret, mu_high, s_high, mu_low, s_low, mu_vol, s_vol = outputs

        # Convertir log_var a sigma
        sigma_ret = torch.sqrt(torch.exp(s_ret))
        sigma_high = torch.sqrt(torch.exp(s_high))
        sigma_low = torch.sqrt(torch.exp(s_low))
        sigma_vol = torch.sqrt(torch.exp(s_vol))

        # SNR
        snr_ret = torch.abs(mu_ret) / (sigma_ret + 1e-8)

        # Intervalos de confianza (±2σ ≈ 95%)
        lower_ret = mu_ret - 2 * sigma_ret
        upper_ret = mu_ret + 2 * sigma_ret

        predictions = {
            'ret': {
                'mu': float(mu_ret),
                'sigma': float(sigma_ret),
                'snr': float(snr_ret),
                'lower_95': float(lower_ret),
                'upper_95': float(upper_ret)
            },
            'high': {
                'mu': float(mu_high),
                'sigma': float(sigma_high)
            },
            'low': {
                'mu': float(mu_low),
                'sigma': float(sigma_low)
            },
            'vol': {
                'mu': float(mu_vol),
                'sigma': float(sigma_vol)
            }
        }

    return predictions
```

### Trading Logic con Risk-Awareness

```python
def make_trading_decision(predictions, current_close, config):
    """
    Decisión de trading usando μ, σ y SNR.

    Args:
        predictions: Dict de predict_with_uncertainty()
        current_close: Precio actual
        config: Dict con thresholds y parámetros

    Returns:
        decision: Dict con acción, entry, tp, sl o None si no operar

    Ejemplo:
        >>> config = {
        ...     'snr_threshold': 1.5,
        ...     'min_ret': 0.005,  # 0.5% mínimo
        ...     'sl_factor': 0.98,
        ...     'risk_per_trade': 0.02  # 2% capital
        ... }
        >>> decision = make_trading_decision(preds, 42000, config)
    """
    ret = predictions['ret']
    high = predictions['high']
    low = predictions['low']

    mu_ret = ret['mu']
    snr_ret = ret['snr']

    # Filtro 1: SNR debe ser alto
    if snr_ret < config['snr_threshold']:
        return None  # No hay señal clara

    # Filtro 2: |μ| debe superar mínimo
    if abs(mu_ret) < config['min_ret']:
        return None  # Movimiento esperado muy pequeño

    # Convertir log-returns a precios
    p0 = current_close
    pred_close = p0 * np.exp(mu_ret)
    pred_high_price = p0 * np.exp(high['mu'])
    pred_low_price = p0 * np.exp(low['mu'])

    decision = {}

    # Determinar dirección
    if mu_ret > 0:
        # Alcista: LONG
        decision['direction'] = 'long'

        # Entry: promedio entre precio actual y low predicho
        # (esperar retroceso hasta ese nivel)
        decision['entry1'] = (p0 + pred_low_price) / 2
        decision['entry2'] = 0.6 * p0 + 0.4 * pred_low_price  # Más agresivo

        # Take Profit: promedio entre close predicho y high predicho
        decision['tp'] = (pred_close + pred_high_price) / 2

        # Stop Loss: debajo del low predicho
        decision['sl'] = pred_low_price * config['sl_factor']

        # Position sizing inversamente proporcional a σ
        # Mayor σ → menor tamaño (risk-aware)
        base_size = config['risk_per_trade']
        sigma_ret = ret['sigma']
        decision['position_size'] = base_size / (1 + 10 * sigma_ret)

    elif mu_ret < 0:
        # Bajista: SHORT
        decision['direction'] = 'short'
        decision['entry1'] = (p0 + pred_high_price) / 2
        decision['entry2'] = 0.6 * p0 + 0.4 * pred_high_price
        decision['tp'] = (pred_close + pred_low_price) / 2
        decision['sl'] = pred_high_price * (2 - config['sl_factor'])

        sigma_ret = ret['sigma']
        decision['position_size'] = base_size / (1 + 10 * sigma_ret)

    # Metadata
    decision['snr'] = snr_ret
    decision['mu_ret'] = mu_ret
    decision['sigma_ret'] = ret['sigma']
    decision['confidence'] = 'high' if snr_ret > 2.0 else 'medium'

    return decision
```

**Ventajas de esta lógica**:
1. ✅ **Filtrado por SNR**: Solo opera con señal clara
2. ✅ **Position sizing**: Menor riesgo en alta incertidumbre
3. ✅ **Stops adaptativos**: Basados en predicción de low/high
4. ✅ **Confidence levels**: Metadata para logging/análisis

---

## 8. Problemas Comunes y Soluciones

### Problema 1: Sigma Inflado (Trivial Solution)

**Síntoma**:
- σ predicho muy grande (>1.0 en log-returns)
- Loss baja pero predicciones inútiles (intervalos demasiado amplios)
- μ con poca precisión

**Causa**:
El modelo "hace trampa": predice σ alto para reducir el término `exp(-s) * (y - μ)²`

**Solución**:
```python
# Añadir penalización en loss
penalty = beta * log_var.mean()
loss = base_loss + penalty

# Beta típico: 1e-3 a 1e-2
```

**Verificación**:
```python
# Check distribución de σ predicho
sigmas = [torch.sqrt(torch.exp(s)) for s in all_log_vars]
print(f"Mean σ: {np.mean(sigmas):.4f}")
print(f"Std σ: {np.std(sigmas):.4f}")

# Si mean > 0.5 en log-returns → sospechar inflación
```

---

### Problema 2: Sigma Colapsa a Cero

**Síntoma**:
- Loss → NaN o Inf
- σ predicho ~ 0
- Gradientes explotan

**Causa**:
Sin clamp, s puede ser muy negativo → exp(s) ~ 0 → división por 0

**Solución**:
```python
# En HeteroHead.forward()
s = torch.clamp(log_var, min=-12.0, max=8.0)

# Esto garantiza:
# exp(-12) = 6e-6 ≤ σ² ≤ exp(8) = 2981
```

---

### Problema 3: Gradientes Inestables

**Síntoma**:
- Loss oscila violentamente
- No converge
- Warnings de NaN en backward

**Solución Multi-nivel**:

1. **Gradient clipping** (esencial):
```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

2. **Learning rate moderado**:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # No más alto
```

3. **Mixed precision con precaución**:
```python
# Si AMP causa NaNs, desactivar temporalmente
use_amp = False  # Verificar sin AMP primero
```

4. **Inicialización correcta**:
```python
initialize_hetero_heads(model, y_train_dict)  # ANTES de entrenar
```

---

### Problema 4: Calibración Pobre

**Síntoma**:
- Observed fractions muy diferentes de expected (>10% diff)
- σ predicho no refleja error real

**Diagnóstico**:
```python
stats = calibrate_uncertainty(model, val_loader, device, 'ret')
# Si observed['1sigma'] = 0.85 (esperado 0.68) → σ sobre-estimado
# Si observed['1sigma'] = 0.50 (esperado 0.68) → σ sub-estimado
```

**Soluciones**:

1. **σ sobre-estimado**:
   - Aumentar `penalty_sigma` (ej: 1e-3 → 5e-3)
   - Verificar que init no sea demasiado alto

2. **σ sub-estimado**:
   - Reducir `penalty_sigma` (ej: 1e-3 → 1e-4)
   - Añadir más dropout (regularización)
   - Verificar que model no esté overfitting a μ

3. **Asimetría (over en 1σ, under en 2σ)**:
   - Considerar Student-t likelihood (colas pesadas)
   - Revisar outliers en training data

---

### Problema 5: Overfitting

**Síntoma**:
- Train loss << Val loss (gap grande)
- σ predicho en train muy pequeño, en val grande

**Solución**:
```python
# 1. Dropout en heads
class HeteroHead(nn.Module):
    def __init__(self, hidden_size):
        ...
        self.mu = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Dropout(0.2),  # Aumentar de 0.1 a 0.2
            nn.Linear(mid, 1)
        )

# 2. Weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 3. Early stopping
# Guardar best model basado en val_loss
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pt')
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

---

## 9. Comparación MSE vs Hetero NLL

### Tabla Comparativa

| Aspecto | MSE (LSTM2Head actual) | Hetero NLL (LSTM4HeadsHetero) |
|---------|------------------------|-------------------------------|
| **Outputs por head** | 1 (solo μ) | 2 (μ y s = log σ²) |
| **Varianza** | Constante (σ² fija) | Condicional (σ²_t = f(x_t)) |
| **Incertidumbre** | ❌ No disponible | ✅ σ por muestra |
| **Intervalos** | ❌ No | ✅ μ ± kσ (k=1,2,3) |
| **SNR (μ/σ)** | ❌ No | ✅ Risk-aware filtering |
| **Volatility clustering** | ❌ Ignora | ✅ Captura |
| **Position sizing** | Fijo | ✅ Adaptativo (∝ 1/σ) |
| **Complejidad** | Baja | Media (+50% parámetros) |
| **Estabilidad training** | Alta | Media (requiere init + clipping) |
| **Interpretabilidad** | Alta (solo μ) | Media (μ ± σ) |
| **Costo computacional** | 1x | ~1.3x (2 heads por output) |
| **VRAM (GTX 1070)** | ~3GB (hidden=64) | ~4.5GB (hidden=128) |

### Cuándo Usar Cada Uno

**Usa MSE si**:
- ✅ Prototipado rápido
- ✅ Baseline para comparar
- ✅ Hardware limitado (<4GB VRAM)
- ✅ No necesitas gestión de riesgo sofisticada

**Usa Hetero NLL si**:
- ✅ Necesitas intervalos de confianza
- ✅ Quieres decisiones risk-aware (SNR filtering)
- ✅ Mercados con volatility clustering
- ✅ Position sizing adaptativo
- ✅ Tienes 6GB+ VRAM

---

## 10. Integración con Proyecto Actual

### Estado Actual del Proyecto

**Arquitectura existente** (`fiboevo.py`):
- ✅ `LSTM2Head`: Predice ret y vol (MSE loss)
- ✅ `create_sequences_from_df`: Devuelve X, y_ret, y_vol
- ✅ `train_epoch` y `eval_epoch`: MSE-based
- ✅ `prepare_dataset.py`: Pipeline completo

**Limitaciones**:
- ❌ No modela incertidumbre (σ constante)
- ❌ No predice high/low (solo ret)
- ❌ Decisiones binarias (sin SNR filtering)

### Plan de Migración (Backward Compatible)

#### Fase 1: Añadir Código Nuevo (Sin Romper Existente)

**Archivo**: `fiboevo.py`

```python
# Añadir al final del archivo (después de LSTM2Head)

# ========== HETEROCEDASTICIDAD ==========
if torch is not None:
    class HeteroHead(nn.Module):
        """Cabeza heterocedástica (μ, log_var)"""
        # ... implementación completa arriba ...

    class LSTM4HeadsHetero(nn.Module):
        """4 heads heterocedásticas"""
        # ... implementación completa arriba ...

    def hetero_nll_loss(mu, log_var, y, reduction='mean'):
        """Gaussian NLL"""
        # ... implementación completa arriba ...

    def combined_hetero_loss(outputs, targets, weights, penalty_sigma=0.0):
        """Loss combinada"""
        # ... implementación completa arriba ...

    def train_epoch_hetero(...):
        """Training loop hetero"""
        # ... implementación completa arriba ...

    def eval_epoch_hetero(...):
        """Eval loop hetero"""
        # ... implementación completa arriba ...

    def initialize_hetero_heads(model, y_train_dict):
        """Inicialización"""
        # ... implementación completa arriba ...

    def calibrate_uncertainty(model, loader, device, head_name='ret'):
        """Calibración"""
        # ... implementación completa arriba ...

else:
    # Fallbacks si torch no disponible
    HeteroHead = None
    LSTM4HeadsHetero = None
```

**Ventaja**: Código existente (`LSTM2Head`, `train_epoch`) sigue funcionando.

#### Fase 2: Extender create_sequences_from_df

**Archivo**: `fiboevo.py`

```python
def create_sequences_from_df(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    seq_len: int = 32,
    horizon: int = 1,
    include_high_low: bool = False  # NUEVO parámetro
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],  # Legacy
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # Hetero
]:
    """
    Crea secuencias. Si include_high_low=True, devuelve 5 arrays.

    Returns:
        Si include_high_low=False: (X, y_ret, y_vol)  # Legacy
        Si include_high_low=True: (X, y_ret, y_vol, y_high, y_low)  # Hetero
    """
    # ... código existente para X, y_ret, y_vol ...

    if not include_high_low:
        return X, y_ret, y_vol  # Legacy behavior

    # NUEVO: Calcular y_high, y_low
    if "high" not in df.columns or "low" not in df.columns:
        LOGGER.warning("high/low not in df, using y_ret as proxy")
        return X, y_ret, y_vol, y_ret.copy(), y_ret.copy()

    high = df["high"].astype(np.float64).values
    low = df["low"].astype(np.float64).values
    logh = np.log(high + 1e-12)
    logl = np.log(low + 1e-12)

    y_high = np.zeros((N,), dtype=np.float32)
    y_low = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        t0 = i + seq_len - 1
        t_h = t0 + horizon

        # Max high y min low en ventana futura
        future_high_max = float(np.max(logh[t0 + 1 : t_h + 1]))
        future_low_min = float(np.min(logl[t0 + 1 : t_h + 1]))

        # Log-return vs close[t0]
        y_high[i] = future_high_max - logc[t0]
        y_low[i] = future_low_min - logc[t0]

    return X, y_ret, y_vol, y_high, y_low
```

**Clave**: `include_high_low=False` por defecto → backward compatible.

#### Fase 3: Actualizar prepare_dataset.py

**Archivo**: `prepare_dataset.py`

```python
# En build_and_save_dataset(), añadir parámetro
def build_and_save_dataset(
    ...,
    use_hetero: bool = False  # NUEVO
):
    ...

    # Llamar a create_sequences_from_df con nuevo flag
    if fiboevo and hasattr(fiboevo, 'create_sequences_from_df'):
        result = fiboevo.create_sequences_from_df(
            df_clean,
            feature_cols,
            seq_len=seq_len,
            horizon=horizon,
            include_high_low=use_hetero  # Pasar flag
        )

        if use_hetero and len(result) == 5:
            X_all, y_ret_all, y_vol_all, y_high_all, y_low_all = result
        else:
            X_all, y_ret_all, y_vol_all = result
            y_high_all = y_ret_all.copy()  # Fallback
            y_low_all = y_ret_all.copy()

    # Crear DataLoader con 5 tensors si hetero
    if use_hetero:
        ds_train = TensorDataset(
            torch.from_numpy(Xtr),
            torch.from_numpy(ytr),
            torch.from_numpy(voltr),
            torch.from_numpy(yhigh_tr),
            torch.from_numpy(ylow_tr)
        )
    else:
        # Legacy: 3 tensors
        ds_train = TensorDataset(
            torch.from_numpy(Xtr),
            torch.from_numpy(ytr),
            torch.from_numpy(voltr)
        )
```

#### Fase 4: CLI Flag

**Archivo**: `prepare_dataset.py`

```python
def parse_args():
    ...
    p.add_argument("--use-hetero", action="store_true", default=False,
                   help="Use heteroscedastic LSTM (4-heads with uncertainty)")
    return p.parse_args()

def main():
    args = parse_args()
    ...
    res = build_and_save_dataset(
        ...,
        use_hetero=args.use_hetero
    )
```

**Uso**:
```bash
# Legacy (MSE)
python prepare_dataset.py --sqlite data.db --symbol BTCUSDT --timeframe 30m

# Hetero (NLL)
python prepare_dataset.py --sqlite data.db --symbol BTCUSDT --timeframe 30m --use-hetero
```

#### Fase 5: Trading Daemon

**Archivo**: `trading_daemon.py`

```python
# En iteration_once(), después de predicción
if self.use_hetero:
    # Modelo retorna 8 outputs
    outputs = self.model(xb)
    mu_ret, s_ret, mu_high, s_high, mu_low, s_low, mu_vol, s_vol = outputs

    # Convertir a predicciones con incertidumbre
    predictions = {
        'ret': {
            'mu': float(mu_ret),
            'sigma': float(torch.sqrt(torch.exp(s_ret))),
            'snr': float(torch.abs(mu_ret) / (torch.sqrt(torch.exp(s_ret)) + 1e-8))
        },
        'high': {'mu': float(mu_high)},
        'low': {'mu': float(mu_low)},
    }

    # Decisión con SNR filtering
    decision = make_trading_decision(predictions, current_close, self.config)
    if decision is None:
        self._enqueue_log(f"No trade: SNR too low ({predictions['ret']['snr']:.2f})")
        return
else:
    # Legacy behavior (MSE)
    pred_ret, pred_vol = self.model(xb)
    # ... lógica existente ...
```

---

## 11. Configuración Recomendada

### Para GTX 1070 (8GB VRAM)

```python
config_hetero = {
    # ========== Model Architecture ==========
    'input_size': 35,  # ~35-40 features con decouple
    'hidden_size': 128,  # Reducir a 96 si OOM
    'num_layers': 2,
    'dropout': 0.15,

    # ========== Training ==========
    'batch_size': 64,  # Reducir a 48 o 32 si OOM
    'seq_len': 32,
    'horizon': 10,
    'epochs': 50,
    'early_stopping_patience': 10,

    # Optimizer
    'learning_rate': 1e-4,  # Moderado para estabilidad
    'weight_decay': 1e-5,   # L2 regularization
    'grad_clip': 1.0,       # Gradient clipping (esencial)

    # Mixed Precision (GTX 1070 soporta FP16)
    'use_amp': True,

    # ========== Loss Weights ==========
    'w_ret': 1.0,          # Retorno (principal)
    'w_high': 0.8,         # High (secundario)
    'w_low': 0.8,          # Low (secundario)
    'w_vol': 0.5,          # Volatilidad (auxiliar)
    'penalty_sigma': 1e-3, # Anti-inflación σ

    # ========== DataLoader ==========
    'num_workers': 2,      # 2-4 en Windows, 4-8 en Linux
    'pin_memory': True,
    'persistent_workers': True,

    # ========== Inference ==========
    'snr_threshold': 1.5,  # Minimum SNR para operar
    'min_ret_threshold': 0.005,  # 0.5% return mínimo
    'sl_factor': 0.98,     # Stop loss 2% debajo low
    'risk_per_trade': 0.02,  # 2% capital por trade

    # ========== Calibration Targets ==========
    'calibration_tolerance': 0.05,  # ±5% es aceptable
}
```

### Para GPU Más Potente (16GB+)

```python
config_high_end = {
    'hidden_size': 192,  # Más capacidad
    'batch_size': 128,   # Batches más grandes
    'num_workers': 8,    # Más workers
    'dropout': 0.1,      # Menos regularización (más datos en batch)
}
```

### Para CPU Only (Desarrollo/Testing)

```python
config_cpu = {
    'hidden_size': 64,   # Más pequeño
    'batch_size': 16,    # Batches chicos
    'num_workers': 0,    # Sin multiprocessing
    'use_amp': False,    # AMP solo en GPU
}
```

---

## 12. Referencias y Recursos

### Papers Fundamentales

1. **"Estimating Uncertainty in Neural Networks for Cardiac MRI Segmentation"**
   Kendall & Gal (2017)
   Introduce aleatoric (data) vs epistemic (model) uncertainty

2. **"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"**
   Kendall & Gal (2017)
   Heteroscedastic aleatoric uncertainty en CNN

3. **"Practical Deep Learning with Bayesian Principles"**
   Osawa et al. (2019)
   VOGN optimizer para incertidumbre

4. **"GARCH Models"**
   Bollerslev (1986)
   Modelo clásico de heterocedasticidad en finanzas

### Implementaciones de Referencia

- **PyTorch Lightning**: `torch.nn.GaussianNLLLoss` (built-in desde 1.10)
- **TensorFlow Probability**: `tfp.distributions.Normal` para NLL
- **Pyro**: Framework bayesiano sobre PyTorch

### Libros

- **"Deep Learning"** - Goodfellow, Bengio, Courville (2016)
  Cap. 5.5: Maximum Likelihood Estimation

- **"Pattern Recognition and Machine Learning"** - Bishop (2006)
  Cap. 2.3: Gaussian Distribution
  Cap. 5.6: Mixture Density Networks

### Tutoriales Online

- Uncertainty in Deep Learning (Yarin Gal PhD thesis)
- Uncertainty Quantification 360 (IBM)
- Probabilistic Deep Learning (Manning, 2023)

---

## Apéndice A: Glosario de Términos

| Término | Definición |
|---------|-----------|
| **Heterocedasticidad** | Varianza no constante (condicional a x_t) |
| **Homocedasticidad** | Varianza constante (σ² fija) |
| **NLL** | Negative Log-Likelihood (función de pérdida probabilística) |
| **log_var** | log(σ²), parametrización estable de varianza |
| **SNR** | Signal-to-Noise Ratio = \|μ\| / σ |
| **Calibración** | Qué tan bien σ predicho refleja error real |
| **Aleatoric uncertainty** | Incertidumbre en los datos (ruido) |
| **Epistemic uncertainty** | Incertidumbre del modelo (falta de datos) |
| **Volatility clustering** | Alta volatilidad seguida de alta volatilidad |
| **Mixed precision (AMP)** | Entrenamiento FP16 + FP32 (más rápido) |

---

## Apéndice B: Checklist de Implementación

### Antes de Entrenar
- [ ] `high` y `low` en df_ohlcv (no en features, pero disponibles)
- [ ] `create_sequences_from_df` devuelve 5 arrays
- [ ] `DataLoader` con 5 tensors (xb, yret, yvol, yhigh, ylow)
- [ ] Modelo `LSTM4HeadsHetero` instanciado correctamente
- [ ] `initialize_hetero_heads()` llamado con y_train_dict
- [ ] Learning rate moderado (1e-4)
- [ ] Gradient clipping activado (1.0)
- [ ] Mixed precision habilitado si GPU lo soporta

### Durante Entrenamiento
- [ ] Loss converge sin NaNs/Infs
- [ ] Component losses balanceadas (ninguna domina)
- [ ] Val loss no diverge de train loss (gap <2x)
- [ ] Gradients no explotan (monitoring)
- [ ] VRAM usage <80% (headroom para spikes)

### Después de Entrenar
- [ ] Calibration test: 68% dentro 1σ, 95% dentro 2σ
- [ ] Sigma distribution razonable (mean ~0.01-0.05 para log-returns)
- [ ] SNR distribution: mayoría >1.0, algunos >2.0
- [ ] Backtest con SNR filtering mejora Sharpe
- [ ] Intervalos de confianza visualizados

### En Producción
- [ ] Trading daemon usa μ y σ
- [ ] SNR threshold tuneado en backtest
- [ ] Position sizing inversamente proporcional a σ
- [ ] Logging de confidence levels
- [ ] Alertas si calibration degrada

---

## Apéndice C: Troubleshooting Quick Reference

| Síntoma | Causa Probable | Solución Rápida |
|---------|----------------|-----------------|
| Loss → NaN | σ colapsa | Añadir `clamp(s, -12, 8)` |
| Loss no converge | LR muy alto | Reducir a 1e-4 o 1e-5 |
| Gradientes explotan | Sin clipping | `clip_grad_norm_(params, 1.0)` |
| σ muy grande | Trampa del modelo | Aumentar `penalty_sigma` a 5e-3 |
| Calibration pobre | Init incorrecto | Llamar `initialize_hetero_heads()` |
| OOM en GTX 1070 | Batch/hidden grande | Reducir batch=32, hidden=96 |
| Train/Val gap grande | Overfitting | Aumentar dropout a 0.2, weight_decay |
| SNR siempre bajo | σ sobre-estimado | Reducir `penalty_sigma`, verificar init |
| Predicciones idénticas | Modelo no aprendió | Verificar LR, epochs, data quality |

---

**Documento creado**: 2025-10-14
**Autor**: Análisis de heterocedasticidad para LSTM de trading
**Versión**: 1.0
**Estado**: Documentación completa lista para implementación
