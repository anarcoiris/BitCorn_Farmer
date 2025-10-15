#--------#
# NOTAS: #
#--------#


"""
# notas 14 de octubre:

# Creating the markdown file with the requested content and code snippets.
content = r'''# Diseño: Predicción de supremo/infimo y reglas de trading (fiboevo)

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
Reglas ejemplo (implementables):

Si pred_ret > 0 (alcista):

Entrada 1 = (precio_actual + precio_low) / 2
entry1 = (p_now + pred_low_price)/2

Entrada 2 = 0.6*p_now + 0.4*pred_low_price (más agresiva, entra por debajo)

Salida objetivo = (pred_close + pred_high_price) / 2

Si pred_ret < 0 (bajista): simetría usando pred_high_price para cortos (o entradas por encima para abrir short).

Añade spreads, slippage y verifica todo con backtest; la entrada2 suele ser más arriesgada pero puede mejorar ejecución si el supremo/infimo se cumple.

5) Cambios de código — patches listos
A continuación tienes los cambios principales para integrar en fiboevo.py. Pega los fragmentos donde corresponda (respeta imports np, torch, nn, DataLoader, TensorDataset, etc.).

5.1 create_sequences_from_df → ahora devuelve X, y_ret, y_vol, y_high, y_low
python
Mostrar siempre los detalles

Copiar código
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
Nota: si quieres y_high_std / y_low_std lo añadimos, pero lo dejo opcional para mantener simplicidad.

5.2 Modelo con 4 cabezas (ret, high, low, vol)
python
Mostrar siempre los detalles

Copiar código
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
5.3 train_epoch y eval_epoch adaptados (MSE multi-head, con pesos)
python
Mostrar siempre los detalles

Copiar código
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
python
Mostrar siempre los detalles

Copiar código
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
Compatibilidad: si aún quieres el pipeline antiguo (X, y_ret, y_vol), puedes envolver y crear y_high=y_low=y_ret temporalmente para retrocompatibilidad, pero la migración a 5-tupla es preferible.

5.4 DataLoader / TensorDataset
python
Mostrar siempre los detalles

Copiar código
# Después de X, y_ret, y_vol, y_high, y_low = create_sequences_from_df(...)
ds = TensorDataset(
    torch.from_numpy(X),
    torch.from_numpy(y_ret),
    torch.from_numpy(y_vol),
    torch.from_numpy(y_high),
    torch.from_numpy(y_low),
)
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
Batch size: para tu GTX1070 (8GB VRAM) recomiendo batch_size=64 o 128 según hidden y seq_len. Si OOM, baja a 32 o 64.

6) Pérdidas alternativas (quantile y NLL)
Quantile loss (ejemplo: cuantíl 0.9):

python
Mostrar siempre los detalles

Copiar código
def quantile_loss(pred, target, q=0.9):
    err = target - pred
    return torch.max(q*err, (q-1.0)*err).mean()
NLL heterocedástico (predice mu y sigma>0):

python
Mostrar siempre los detalles

Copiar código
# ejemplo conceptual:
mu = head_mu(last)
sigma = nn.Softplus()(head_sigma(last)) + 1e-6
nll = 0.5 * (((target - mu) / sigma) ** 2).mean() + torch.log(sigma).mean()
Esto te da incertidumbre por muestra; útil para decidir si abrir o no posición (si sigma grande => evitar).

7) Fine-tuning / reciclabilidad
Warm-start (preferible): guarda checkpoints cada N epochs. Para actualizar con nuevos datos (diarios/intradiarios):

Carga modelo + scaler.

Concatena los últimos K minutos/histórico con los nuevos datos (sliding window).

Entrena pocos epochs con lr_new = 0.1 * lr_original y shuffle=True.

Usa replay: buffer aleatorio de muestras pasadas para evitar olvido (rehearsal).

Retrain completo: solo si cambias arquitectura o features. Para drift gradual, warm-start suele valer.

Scalers: importante: actualizar StandardScaler con partial_fit o recalcula sobre ventana reciente y mantener estadísticas históricas. Si cambias scaler drásticamente, el modelo puede perder rendimiento.

Frecuencia: en 1m timeframe, prueba fine-tune diario y reentrenado completo semanal. Ajusta según drift del activo.

8) Hardware (GTX1070 / 8GB VRAM) — recomendaciones concretas
batch_size: empieza en 64; probar 128 si cabe. Si OOM reduce a 32.

seq_len=224 + hidden=192 + batch=128 puede ser alto — monitoriza VRAM. Para prototipo baja a hidden=128 y batch=64.

torch.backends.cudnn.benchmark = True.

num_workers=4 en DataLoader y pin_memory=True.

Habilita mixed precision (AMP) para acelerar y reducir VRAM (use_amp=True).

Usa GPU siempre; CPU será lento.

9) Métricas a reportar (buenas prácticas)
MAE precio: convierte predicciones log→precio y calcula MAE/RMSE en unidades monetarias.

Directional accuracy: % de veces que pred_ret tiene el mismo signo que y_ret.

Peak hit-rate: % en el que true_high >= umbral de ejecución (ej. pred_high_price > entry_price + spread).

Coverage (si usas NLL/quantiles): % en que el true_high está dentro del intervalo previsto.

Snippet (MAE precio + accuracy):

python
Mostrar siempre los detalles

Copiar código
def evaluate_price_metrics(pred_ret, y_ret, close_t0):
    pred_price = close_t0 * np.exp(pred_ret)
    true_price = close_t0 * np.exp(y_ret)
    mae_price = np.mean(np.abs(pred_price - true_price))
    rmse_price = np.sqrt(np.mean((pred_price - true_price)**2))
    dir_acc = np.mean((np.sign(pred_ret) == np.sign(y_ret)).astype(float))
    return {"mae_price": mae_price, "rmse_price": rmse_price, "dir_acc": dir_acc}
10) Resumen de acciones concretas (prioridad)
Reemplaza create_sequences_from_df por la versión que devuelve y_high y y_low.

Cambia la creación del dataset / DataLoader para incluir esos targets.

Sustituye LSTM2Head por LSTM4Heads y ajusta train/eval como te pasé.

Entrena inicialmente con MSE, w_high=0.7, w_low=0.7, alpha_vol_loss=0.5 (ajusta).

Habilita AMP (mixed precision) y num_workers=4, batch_size=64 para tu 1070.

Añade evaluación en unidades de precio cada epoch para tener números interpretables.

Implementa fine-tune warm-start para actualizar con datos recientes.


"""


"""
# notas 2 de octubre

Primero analiza el conjunto, sin los fiboevo.py anteriores al que acabas de actualizar, y revisa:
1: Qué cambios en el resto de la logica requiere el nuevo fiboevo.py
2: Qué conflictos puede haber y cómo resolverlos.

Analiza esta informacion y elabora, razonadamente, un plan o lista de cambios que habría que aplicar al resto de scripts involucrados (principalmente trading_daemon, trading_gui_extended y prepare_dataset)
Si hay conflictos, analiza los pros y los contras de cada uno, evalua cual debería ser el merge más indicado para corregir problemas, expandir funcionalidad y no romper otras partes que dependan de estas salvo que sea completamente necesario (en cuyo caso indicarás qué modificaciones requiere en los callers)
Haz todos los preparativos necesarios para una transición fina a la actualización de trading_gui_extended.py (y tomando las notas que sean necesarias para el TradingDaemon u otros)
ChatGPT Plus
Pensó durante 13s

Perfecto — voy a analizar razonada y planificadamente el conjunto sin los fiboevo.py anteriores y centrarme en las implicaciones de sustituirlos por la versión modificada que te entregué. Te doy (1) un diagnóstico de impacto, (2) conflictos potenciales y cómo resolverlos, y (3) un plan de cambios concreto y priorizado con snippets de parche para trading_daemon, trading_gui_extended, y prepare_dataset + pasos de verificación.

Lo entrego en español, con secciones claras para que puedas aplicarlo paso a paso.

Resumen ejecutivo (1 línea)

La nueva fiboevo.py introduce: prepare_input_for_model, ensure_scaler_feature_names, load_model → (model, meta), load_scaler(..., meta), y una simulate_fill unificada; todo ello mejora robustez (alineamiento columnas, evitar doble-scaling, modo de liquidez). Esto requiere cambios localizados en los callers que cargan modelos/scalers y en los puntos de inferencia/backtest para usar prepare_input_for_model y validar meta.

1) Cambios en el resto de la lógica que requiere la nueva fiboevo.py

A continuación enumero las áreas del código que deben adaptarse y por qué:

A. Carga de modelos

Qué cambió: load_model ahora devuelve (model, meta) (antes devolvía sólo model o model + sidecar implícito).
Impacto: Todos los callers que hacen model = load_model(path) fallarán o ignorarán meta.
Acción: cambiar por model, model_meta = load_model(path) y usar model_meta para validar feature_cols, seq_len, horizon, input_size.

B. Carga de scaler

Qué cambió: load_scaler(path, meta) inyecta feature_names_in_ cuando falta; existe además ensure_scaler_feature_names.
Impacto: Callers que hacían scaler = joblib.load("scaler.pkl") deberían usar ahora scaler = load_scaler("scaler.pkl", meta) o ensure_scaler_feature_names(scaler, meta) para prevenir desalineamiento de columnas.
Acción: actualizar cargas de scaler en CLI, GUI y daemon.

C. Inferencia — preparación de la entrada

Qué cambió: introducción de prepare_input_for_model(df, feature_cols, seq_len, scaler, method) y recomendaciones de usarlo en vez de construir arrays manualmente (.values.astype(np.float32)).
Impacto: backtest_market_maker, trading_daemon.iteration_once() (o similar) y GUI predictores deben sustituir la construcción manual de seq_x por prepare_input_for_model(...).to(device) para evitar:

doble scaling (si df ya estaba escalado)

reordenamientos de columnas que provoquen features mal alineadas
Acción: pasar scaler a backtest_market_maker, y usar prepare_input_for_model siempre.

D. Backtester simulate_fill

Qué cambió: simulate_fill ahora soporta modo simple (compatible) y modo liquidity-aware que puede devolver filled_fraction en tercer valor.
Impacto: Callers que interpreten siempre el tercer retorno como índice label deben comprobar comportamiento. En el backtester que te entregué he adaptado la lógica pero otros callers externos deben revisar/ajustar.
Acción: revisar usos de simulate_fill y adaptar (o forzar use_liquidity_model=False si se espera label).

E. Metadata y validaciones

Qué hay ahora: meta puede contener feature_cols, input_size, seq_len, horizon. prepare_input_for_model valida scaler.feature_names_in_ vs feature_cols.
Impacto: scripts que suponen auto-detección libre de features ahora deben preferir meta['feature_cols'].
Acción: añadir rutina de validación en daemon/gui que:

si meta.feature_cols existe, usarlo y avisar si columnas faltan;

si no existe, auto-detectar pero loggear advertencia.

F. Guardar artefactos en prepare_dataset

Qué hay que hacer: garantizar que prepare_dataset/training guarden meta completo y que scaler se guarde con feature_names_in_.
Impacto: si no se guardan, load_scaler(..., meta) no podrá inyectar nombres y puede producir desalineamientos.
Acción: añadir scaler.feature_names_in_ = np.array(feature_cols) al guardar y guardar meta.json con feature_cols, seq_len, horizon, input_size.

G. Indices / timestamps / IDs

Qué hay: create_sequences_from_df espera df.dropna().reset_index(drop=True) — por tanto utiliza posiciones. simulate_fill usa index label para obtener posición mediante df.index.get_loc(label).
Impacto: si el código del daemon o del GUI mezcla DataFrames con índices temporales y luego resetea índices de manera inconsistente con el uso en simulate_fill, los fill_label y get_loc pueden devolver errores.
Acción: definir una convención clara:

para backtest: usar DataFrame con índice temporal (DatetimeIndex) y no resetear indices hasta después de simular fills; cuando sea necesario, usar df.reset_index() localmente en create_sequences_from_df (ya hace reset_index internamente).

trading_daemon: al recibir datos por websocket, conservar timestamps como índice y sólo pasar subsecciones con reset_index(drop=True) a prepare_input_for_model.

2) Conflictos posibles y recomendaciones de resolución

A continuación describo conflictos probables, pros/cons y la recomendación para el merge.

Conflicto 1 — load_model firma antigua vs nueva

Problema: muchos sitios hacen model = fiboevo.load_model(path). Nueva firma (model, meta) romperá esos callers.

Opciones:

Cambiar todos los callers para usar la nueva firma (recomendado).

Pro: usa meta de modelo, mejor seguridad/diagnóstico.

Contra: trabajo de refactor.

Crear wrapper retrocompatible: def load_model_compat(path): m = load_model(path); if isinstance(m, tuple): return m[0]; return m

Pro: mínimo cambio visible.

Contra: oculta meta y perpetúa malas prácticas.

Recomendación: aplicar cambio en callers (opción 1) + añadir en fiboevo.py una función load_model_legacy(path) que devuelva sólo model para compatibilidad temporal. Documentar y planificar remover legacy en siguiente release.

Conflicto 2 — escala doble / mezcla de datos escalados y no escalados

Problema: algunos módulos quizá pre-escalaban df antes de llamar a backtester o al daemon. Si además se pasa scaler a prepare_input_for_model, se corre el riesgo de doble-scaling.

Opciones:

Estandarizar: definir que nunca se escale a nivel global; el escalado se hace justo antes de crear secuencias (training) y prepare_input_for_model siempre aplica scaler si se le pasa.

Pro: un único lugar de escalado, evita duplicados.

Contra: hay que refactorizar callers que actualmente pasan df escalado.

Detección automática: introducir meta flag scaler_applied o un campo meta['scaled_by'] = "scaler.pkl"; prepare_input_for_model puede comprobar consistencia mediante la media/var en meta y decidir no volver a escalar.

Pro: más robusto frente a diferentes pipelines.

Contra: requiere guardar más metadata y aplica heurísticas.

Recomendación: combinar 1 y 2: normalizar el pipeline (preferible) y además añadir defensas heurísticas en prepare_input_for_model (p.ej. comprobar si los valores están ya estandarizados usando percentiles y comparar con scaler.mean_ si está disponible). Para transición, aceptar ambos y emitir warning si detecta posible doble-scaling.

Conflicto 3 — retorno de simulate_fill variable (label vs fraction)

Problema: algunos callers esperan el tercer elemento sea fill_label; la nueva función puede devolver filled_fraction.

Opciones:

Mantener retrocompatibilidad por defecto use_liquidity_model=False (ya hecho).

Forzar contrato consistente: devolver siempre (filled, executed_price, info_dict) donde info_dict tiene keys label o filled_fraction.

Recomendación: cambiar el contrato hacia (filled, executed_price, info) donde info es dict (más explícito). Pero esto rompe compatibilidad. Mientras tanto, no romper: mantener comportamiento actual (simple: label) y documentar que callers deben revisar type(fill_info) (datetime/int vs float). En una segunda versión, migrar a dict.

Conflicto 4 — orden de columnas / feature_cols ausentes

Problema: modelo meta indica feature_cols que pueden faltar en runtime (por ejemplo si websocket envía menos features).

Opciones:

Abort inference si faltan columnas (estricto).

Fallback: rellenar columnas faltantes con NaN/0 y loggear (más flexible).

Recomendación: por defecto abort con advertencia (evita inferencias con inputs inválidos). Ofrecer una opción allow_missing_features=True que rellene con np.nan/impute y proseguir (para escenarios exploratorios).

Conflicto 5 — indices y posiciones vs timestamps

Problema: tras reset_index y operaciones, simulate_fill puede devolver indices que get_loc no encuentre.

Recomendación: adoptar convención y documentarla: mantener DatetimeIndex para backtests y usar reset_index(drop=True) sólo en funciones que lo requieran; simulate_fill ya documenta que devuelve label compatible con df.index.get_loc. Añadir defensiva: si get_loc falla, usar fallback a nearest integer index.

3) Plan de cambios (priorizado, con snippets y verificación)

Voy a listar los cambios concretos a aplicar en trading_daemon.py, trading_gui_extended.py y prepare_dataset.py — en orden de prioridad. Para cada cambio incluyo un snippet propuesto.

Prioridad Alta — mantener compatibilidad e integridad (urgente)
1. Actualizar cargas de modelo y scaler (todo caller)

Objetivo: usar model, meta = load_model(path) y scaler = load_scaler(path, meta).

Snippet a aplicar (ejemplo en trading_daemon.py)

Reemplazar:

model = fiboevo.load_model(model_path)
scaler = joblib.load("scaler.pkl")


por:

model, model_meta = fiboevo.load_model(model_path)
# Prefer fiboevo.load_scaler which inyecta feature_names_in_ desde meta
try:
    scaler = fiboevo.load_scaler("scaler.pkl", meta=model_meta)
except Exception:
    # fallback: try joblib directly and ensure feature names
    scaler = joblib.load("scaler.pkl")
    scaler = fiboevo.ensure_scaler_feature_names(scaler, model_meta)


Por qué: garantiza que scaler.feature_names_in_ está presente y que disponemos de model_meta para validaciones posteriores.

Verificación:

Reiniciar daemon, comprueba logs: meta cargado y scaler.feature_names_in_ inyectado o existente.

Prueba: llamar prepare_input_for_model con últimas filas y comprobar que no lanza error de mismatch.

2. Sustituir inferencia manual por prepare_input_for_model

Objetivo: eliminar construcción manual de arrays (evita doble-scaling y desalineamiento).

En backtest_market_maker (ya modificado en la versión entregada) — si tienes una versión antigua en tu repo, sustituir:

seq_x = df_loc[feature_cols].iloc[t - seq_len : t].values.astype(np.float32)
x_t = torch.from_numpy(seq_x).unsqueeze(0).to(device)
pred_ret, pred_vol = model(x_t)


por:

seq_df = df_loc.iloc[t - seq_len : t].reset_index(drop=True)
x_t = fiboevo.prepare_input_for_model(seq_df, feature_cols, seq_len, scaler=scaler, method="per_row").to(device)
pred_ret, pred_vol = model(x_t)


En trading_daemon.iteration_once() (o similar):

Reemplazar el bloque que construye X manualmente por:

# antes de predecir, verifica meta
if hasattr(self, "model_meta") and self.model_meta:
    feature_cols = self.model_meta.get("feature_cols", feature_cols_auto)
# prepare df segment (last seq_len rows)
seq_df = feats_df.iloc[-seq_len:].reset_index(drop=True)
x_t = fiboevo.prepare_input_for_model(seq_df, feature_cols, seq_len, scaler=self.model_scaler).to(self.model_device)
pred_ret, pred_vol = self.model(x_t)


Verificación:

Ejecutar un ciclo de predicción con datos reales; comparar outputs con runs anteriores (espera que la diferencia provenga sólo de corrección de alineamiento y no por bugs).

3. Validación post-load del modelo (en daemon/gui)

Objetivo: evitar inferencias con features faltantes o input_size inconsistent.

Snippet ejemplo:

def validate_model_loaded(model_meta, scaler, current_feats):
    if model_meta and isinstance(model_meta.get("feature_cols"), (list,tuple)):
        feat_meta = model_meta["feature_cols"]
        missing = [c for c in feat_meta if c not in current_feats.columns]
        if missing:
            logger.warning("Missing features: %s", missing)
            # decide: raise or continue with fallback
            raise RuntimeError("Cannot infer: required features missing.")
    # check input_size
    declared = int(model_meta.get("input_size", len(feat_meta)))
    if declared != len(feat_meta):
        logger.warning("meta.input_size (%s) != len(feature_cols) (%s)", declared, len(feat_meta))


Dónde: llamar justo después de load_model.

Verificación: cargar modelos con meta incompleta y comprobar que la función lanza advertencias/errores controlados.

Prioridad Media — coherencia de datos y preservación de indices
4. Normalizar convención de índices (timestamps)

Objetivo: definir y aplicar una convención para índices en backtest y daemon (DatetimeIndex preferido).

Regla sugerida:

Los DataFrame de OHLCV que provienen de DB (Influx/SQLite) deben conservar DatetimeIndex.

create_sequences_from_df seguirá reset_index(drop=True) internamente (ya lo hace), así que no requiere cambio.

simulate_fill espera etiquetas compatibles con df.index.get_loc(label) — no resetear índices antes de llamar a simulate_fill.

Cambios concretos: revisar puntos donde se hace df.reset_index() antes de pasar segmentos a simulate_fill y evitar hacerlo.

Verificación: backtest que utilice timestamps; verificar que fill_label devuelto por simulate_fill se transforma correctamente a position con df.index.get_loc(fill_label).

5. Preparar transición del daemon: pasar scaler y model_meta a estado del daemon

Objetivo: almacenar self.model, self.model_meta, self.model_scaler, self.model_device en TradingDaemon para uso en cada iteración.
Por qué: evita recargas repetidas y facilita validación.

Snippet (inicialización tras cargar modelos):

self.model, self.model_meta = fiboevo.load_model(model_path)
self.model_scaler = fiboevo.load_scaler(scaler_path, meta=self.model_meta)  # or ensure_scaler_feature_names
self.model_device = torch.device("cuda"

"""
