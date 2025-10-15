# Diseño: Predicción de supremo/infimo y reglas de trading (fiboevo)

Cubriremos:

- Confirmar que **bb** usa `std` (sí).
- Cómo construir targets de `high` / `low` (y opcionalmente su desviación / probabilidad).
- Qué cambiar en el modelo (añadir cabezas) y en `train`/`eval`.
- Cómo usar las predicciones para reglas de entrada/salida tal y como propones.
- Opciones de pérdida (MSE, quantile, NLL heterocedástica) y planteamiento matemático como problema de optimización.
- Estrategia de fine-tuning / reciclabilidad vs reentrenar desde cero.
- Sugerencias concretas de configuración para tu **GTX1070 (8 GB VRAM)**.
- Snippets de código listos (patches) para integrar en `fiboevo.py`.

---

## 1) Confirmación rápida: Bollinger + std

Sí — tu `bb_std` es la desviación estándar móvil (`.rolling(20).std()`), así que Bollinger ya incorpora información de dispersión/volatilidad local. Correcto.

---

## 2) Targets: cómo crear `y_high` / `y_low` (y opción de incertidumbre)

Te propongo definir (consistente con tu actual `y_ret` que es log-return del close):

\[
y_{\text{high}} = \log\big(\max_{t \in (t_0+1..t_h)} \text{high}_t\big) - \log(\text{close}_{t_0})
\]

\[
y_{\text{low}} = \log\big(\min_{t \in (t_0+1..t_h)} \text{low}_t\big) - \log(\text{close}_{t_0})
\]

Con esto predices el **retorno máximo posible** y **retorno mínimo posible** en la ventana `horizon` respecto al `close` en `t0`.

Si además quieres una estimación de desviación/incertidumbre, puedes computar dentro de esa ventana la `std` de los returns relativos a `close_{t0}` (o la `std` de retornos intrawindow) y usarlo como target `y_high_std` / `y_low_std` o entrenar la red para que prediga una *sigma* (heterocedasticidad).

---

## 3) Formulación matemática (optimización)

Planteo tres alternativas principales:

### A) Regresión determinista (MSE)

Minimizar:
\[
\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \Big( (y^{(i)}_{\text{ret}} - \hat y^{(i)}_{\text{ret}})^2 + w_h (y^{(i)}_{\text{high}} - \hat y^{(i)}_{\text{high}})^2 + w_l (y^{(i)}_{\text{low}} - \hat y^{(i)}_{\text{low}})^2 + \alpha \cdot \text{vol\_loss}\Big)
\]
con pesos \(w_h, w_l\).

### B) Regresión probabilística (NLL heterocedástico)

Para cada cabeza predecir media \(\mu\) y escala \(\sigma>0\) y minimizar la NLL normal por muestra:
\[
\mathcal{L}_{\text{NLL}} = \sum \Big( \frac{(y-\mu)^2}{2\sigma^2} + \log \sigma \Big)
\]

Esto produce intervalos y probabilidades (útil para gestionar riesgo / decidir si abrir una operación).

### C) Quantile loss

Si te interesa cubrir quantiles (p.ej. predecir el 90% supremo), optimizar `quantile loss` para el cuantil objetivo.

**Recomendación práctica**: empieza con A (MSE) para estabilidad y diagnóstico; luego explora B o C si necesitas incertidumbres o cobertura de cola.

---

## 4) Reglas de entrada/salida (cómo usar las predicciones)

Dado `pred_ret` (predicción de retorno del close), `pred_high`, `pred_low` (todos en **log-returns** respecto a `close_t0`), conviértelos a precios:

```python
p0 = close_t0
pred_close = p0 * np.exp(pred_ret)      # precio objetivo del close
pred_high_price = p0 * np.exp(pred_high)
pred_low_price  = p0 * np.exp(pred_low)
```

Reglas ejemplo (implementables):

- Si `pred_ret > 0` (alcista):
  - **Entrada 1** = `(precio_actual + precio_low) / 2`  
    `entry1 = (p_now + pred_low_price)/2`
  - **Entrada 2** = `0.6*p_now + 0.4*pred_low_price` (más agresiva, entra por debajo)
  - **Salida objetivo** = `(pred_close + pred_high_price) / 2`
- Si `pred_ret < 0` (bajista): simetría usando `pred_high_price` para cortos (o entradas por encima para abrir short).

Añade spreads, slippage y verifica todo con backtest; la entrada2 suele ser más arriesgada pero puede mejorar ejecución si el supremo/infimo se cumple.

---

## 5) Cambios de código — patches listos

A continuación tienes los cambios principales para integrar en `fiboevo.py`. Pega los fragmentos donde corresponda (respeta imports `np, torch, nn, DataLoader, TensorDataset`, etc.).

### 5.1 `create_sequences_from_df` → ahora devuelve `X, y_ret, y_vol, y_high, y_low`

```python
def create_sequences_from_df(
    df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int = 32, horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X, y_ret, y_vol, y_high, y_low
    y_high and y_low are log-returns of max(high) and min(low) in the (t0+1..t_h) window
    relative to close[t0].
    """
    assert seq_len >= 1 and horizon >= 1
    dfc = df.reset_index(drop=True).copy()

    if "close" not in dfc.columns:
        raise RuntimeError("create_sequences_from_df expects column 'close' in df")

    present = [c for c in feature_cols if c in dfc.columns]
    if len(present) == 0:
        raise RuntimeError("No feature columns present after filtering.")

    arr = dfc[present].astype(np.float32).values
    close = dfc["close"].astype(np.float64).values  # use float64 for log stability
    high = dfc["high"].astype(np.float64).values if "high" in dfc.columns else close.copy()
    low  = dfc["low"].astype(np.float64).values  if "low" in dfc.columns else close.copy()

    M = len(dfc)
    N = M - seq_len - horizon + 1
    if N <= 0:
        raise ValueError(f"Not enough rows: need >= seq_len + horizon ({seq_len + horizon}), got {M}")

    F = arr.shape[1]
    X = np.zeros((N, seq_len, F), dtype=np.float32)
    y_ret = np.zeros((N,), dtype=np.float32)
    y_vol = np.zeros((N,), dtype=np.float32)
    y_high = np.zeros((N,), dtype=np.float32)
    y_low  = np.zeros((N,), dtype=np.float32)

    if np.any(close <= 0) or np.any(high <= 0) or np.any(low <= 0):
        raise RuntimeError("Non-positive close/high/low values present; cannot compute log-returns. Clean data first.")

    logc = np.log(close)
    logh = np.log(high)
    logl = np.log(low)

    for i in range(N):
        X[i] = arr[i : i + seq_len]
        t0 = i + seq_len - 1
        t_h = t0 + horizon

        # target: log-return of close between t0 and t_h
        y_ret[i] = float(logc[t_h] - logc[t0])

        # high / low within future window (t0+1 .. t_h inclusive)
        future_high_max = float(np.max(logh[t0 + 1 : t_h + 1]))
        future_low_min   = float(np.min(logl[t0 + 1 : t_h + 1]))

        # log-return of supremo/infimo vs close at t0
        y_high[i] = future_high_max - logc[t0]
        y_low[i]  = future_low_min  - logc[t0]

        # volatility within window (std of intra-window log-returns)
        if horizon >= 1:
            rets = logc[t0 + 1 : t_h + 1] - logc[t0 : t_h]
            y_vol[i] = float(np.std(rets, ddof=0)) if rets.size > 0 else 0.0
        else:
            y_vol[i] = 0.0

    return (
        X.astype(np.float32),
        y_ret.astype(np.float32),
        y_vol.astype(np.float32),
        y_high.astype(np.float32),
        y_low.astype(np.float32),
    )
```

> Nota: si quieres `y_high_std` / `y_low_std` lo añadimos, pero lo dejo opcional para mantener simplicidad.

---

### 5.2 Modelo con 4 cabezas (ret, high, low, vol)

```python
if torch is not None:
    class LSTM4Heads(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers, batch_first=True,
                dropout=(dropout if num_layers > 1 else 0.0)
            )
            def make_head(out_activation=None):
                layers = [nn.Linear(hidden_size, max(4, hidden_size // 2)), nn.ReLU(), nn.Linear(max(4, hidden_size // 2), 1)]
                if out_activation is not None:
                    layers.append(out_activation)
                return nn.Sequential(*layers)

            self.head_ret  = make_head()
            self.head_high = make_head()
            self.head_low  = make_head()
            # vol head predicts positive value
            self.head_vol  = make_head(nn.Softplus())

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            ret  = self.head_ret(last).squeeze(-1)
            high = self.head_high(last).squeeze(-1)
            low  = self.head_low(last).squeeze(-1)
            vol  = self.head_vol(last).squeeze(-1)
            return ret, high, low, vol
else:
    LSTM4Heads = None
```

---

### 5.3 `train_epoch` y `eval_epoch` adaptados (MSE multi-head, con pesos)

```python
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha_vol_loss: float = 0.5,
    w_high: float = 1.0,
    w_low: float = 1.0,
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch in loader:
        # Expect batch = (xb, yret, yvol, yhigh, ylow)
        if len(batch) == 5:
            xb, yret, yvol, yhigh, ylow = batch
        else:
            xb = batch[0]; yret = batch[1]; yvol = batch[2]; yhigh = batch[3]; ylow = batch[4]

        xb = xb.to(device)
        yret = yret.to(device); yvol = yvol.to(device); yhigh = yhigh.to(device); ylow = ylow.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                pred_ret, pred_high, pred_low, pred_vol = model(xb)
                loss_ret = mse(pred_ret, yret)
                loss_high = mse(pred_high, yhigh)
                loss_low  = mse(pred_low, ylow)
                loss_vol  = mse(pred_vol, yvol)
                loss = loss_ret + w_high * loss_high + w_low * loss_low + alpha_vol_loss * loss_vol
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_ret, pred_high, pred_low, pred_vol = model(xb)
            loss_ret = mse(pred_ret, yret)
            loss_high = mse(pred_high, yhigh)
            loss_low  = mse(pred_low, ylow)
            loss_vol  = mse(pred_vol, yvol)
            loss = loss_ret + w_high * loss_high + w_low * loss_low + alpha_vol_loss * loss_vol
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(1, n)
```

```python
def eval_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, alpha_vol_loss: float = 0.5, w_high: float = 1.0, w_low: float = 1.0
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    mse = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                xb, yret, yvol, yhigh, ylow = batch
            else:
                xb = batch[0]; yret = batch[1]; yvol = batch[2]; yhigh = batch[3]; ylow = batch[4]
            xb = xb.to(device)
            yret = yret.to(device); yvol = yvol.to(device); yhigh = yhigh.to(device); ylow = ylow.to(device)

            pred_ret, pred_high, pred_low, pred_vol = model(xb)
            loss_ret = mse(pred_ret, yret)
            loss_high = mse(pred_high, yhigh)
            loss_low  = mse(pred_low, ylow)
            loss_vol  = mse(pred_vol, yvol)
            loss = loss_ret + w_high * loss_high + w_low * loss_low + alpha_vol_loss * loss_vol

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
    return total_loss / max(1, n)
```

**Compatibilidad:** si aún quieres el pipeline antiguo (X, y_ret, y_vol), puedes envolver y crear `y_high=y_low=y_ret` temporalmente para retrocompatibilidad, pero la migración a 5-tupla es preferible.

---

### 5.4 `DataLoader` / `TensorDataset`

```python
# Después de X, y_ret, y_vol, y_high, y_low = create_sequences_from_df(...)
ds = TensorDataset(
    torch.from_numpy(X),
    torch.from_numpy(y_ret),
    torch.from_numpy(y_vol),
    torch.from_numpy(y_high),
    torch.from_numpy(y_low),
)
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
```

**Batch size**: para tu GTX1070 (8GB VRAM) recomiendo `batch_size=64` o `128` dependiendo de `hidden` y `seq_len`. Si OOM, baja a `32` o `64`.

---

## 6) Pérdidas alternativas (quantile y NLL)

**Quantile loss** (ejemplo: cuantíl 0.9):
```python
def quantile_loss(pred, target, q=0.9):
    err = target - pred
    return torch.max(q*err, (q-1.0)*err).mean()
```

**NLL heterocedástico** (predice `mu` y `sigma>0`):
```python
# ejemplo conceptual:
mu = head_mu(last)
sigma = nn.Softplus()(head_sigma(last)) + 1e-6
nll = 0.5 * (((target - mu) / sigma) ** 2).mean() + torch.log(sigma).mean()
```

Esto te da incertidumbre por muestra; útil para decidir si abrir o no posición (si `sigma` grande => evitar).

---

## 7) Fine-tuning / reciclabilidad

**Warm-start (preferible):** guarda checkpoints cada N epochs. Para actualizar con nuevos datos (diarios/intradiarios):

1. Carga modelo + scaler.  
2. Concatena los últimos `K` minutos/histórico con los nuevos datos (sliding window).  
3. Entrena *pocos* epochs con `lr_new = 0.1 * lr_original` y `shuffle=True`.  
4. Usa replay: buffer aleatorio de muestras pasadas para evitar olvido (rehearsal).

**Retrain completo:** solo si cambias arquitectura o features. Para drift gradual, warm-start suele valer.

**Scalers:** importante: actualizar `StandardScaler` con `partial_fit` o recalcula sobre ventana reciente y mantener estadísticas históricas. Si cambias scaler drásticamente, el modelo puede perder rendimiento.

**Frecuencia:** en `1m` timeframe, prueba fine-tune diario y reentrenado completo semanal. Ajusta según drift del activo.

---

## 8) Hardware (GTX1070 / 8GB VRAM) — recomendaciones concretas

- `batch_size`: empieza en `64`; probar `128` si cabe. Si OOM reduce a `32`.
- `seq_len=224` + `hidden=192` + `batch=128` puede ser alto — monitoriza VRAM. Para prototipo baja a `hidden=128` y `batch=64`.
- `torch.backends.cudnn.benchmark = True`.
- `num_workers=4` en `DataLoader` y `pin_memory=True`.
- Habilita **mixed precision (AMP)** para acelerar y reducir VRAM (`use_amp=True`).
- Usa GPU siempre; CPU será lento.

---

## 9) Métricas a reportar (buenas prácticas)

- **MAE precio**: convierte predicciones log→precio y calcula MAE/RMSE en unidades monetarias.  
- **Directional accuracy**: % de veces que `pred_ret` tiene el mismo signo que `y_ret`.  
- **Peak hit-rate**: % en el que `true_high` >= umbral de ejecución (ej. `pred_high_price > entry_price + spread`).  
- **Coverage** (si usas NLL/quantiles): % en que el `true_high` está dentro del intervalo previsto.  

Snippet (MAE precio + accuracy):
```python
def evaluate_price_metrics(pred_ret, y_ret, close_t0):
    pred_price = close_t0 * np.exp(pred_ret)
    true_price = close_t0 * np.exp(y_ret)
    mae_price = np.mean(np.abs(pred_price - true_price))
    rmse_price = np.sqrt(np.mean((pred_price - true_price)**2))
    dir_acc = np.mean((np.sign(pred_ret) == np.sign(y_ret)).astype(float))
    return {"mae_price": mae_price, "rmse_price": rmse_price, "dir_acc": dir_acc}
```

---

## 10) Resumen de acciones concretas (prioridad)

1. Reemplaza `create_sequences_from_df` por la versión que devuelve `y_high` y `y_low`.  
2. Cambia la creación del dataset / `DataLoader` para incluir esos targets.  
3. Sustituye `LSTM2Head` por `LSTM4Heads` y ajusta `train`/`eval` como te pasé.  
4. Entrena inicialmente con **MSE**, `w_high=0.7`, `w_low=0.7`, `alpha_vol_loss=0.5` (ajusta).  
5. Habilita **AMP** (mixed precision) y `num_workers=4`, `batch_size=64` para tu 1070.  
6. Añade evaluación en unidades de precio cada epoch para tener números interpretables.  
7. Implementa fine-tune warm-start para actualizar con datos recientes.

---

Si quieres, puedo ahora generar **un diff** o el archivo `fiboevo.py` completo con las modificaciones ya aplicadas, o **un script de backtest** que pruebe las reglas de entrada/salida y compare usar `pred_high`/`pred_low` vs solo `pred_ret`.
