 Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№
  Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№
   Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№
    Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№
     Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№

    def _connect_websocket(self):
        if websocket is None:
            self._enqueue_log("websocket-client library missing. Run: pip install websocket-client")
            return
        url = self.websocket_url_var.get().strip()
        if not url:
            self._enqueue_log("Websocket URL empty.")
            self.websocket_status_var.set("Disconnected")
            return

        # If already connected, ignore
        if getattr(self, "ws_connected", False):
            self._enqueue_log("Already connected.")
            return

        # Create WebSocketApp with callbacks
        def on_open(ws):
            self.ws_queue.put({"type":"_meta","event":"open"})
            self._enqueue_log("WS open (callback)")

        def on_close(ws, close_status_code, close_msg):
            self.ws_queue.put({"type":"_meta","event":"close", "code": close_status_code, "msg": close_msg})
            self._enqueue_log(f"WS closed: {close_status_code} {close_msg}")

        def on_error(ws, err):
            self.ws_queue.put({"type":"_meta","event":"error", "error": str(err)})
            self._enqueue_log(f"WS error: {err}")

        def on_message(ws, message):
            # Put raw message in queue; actual parsing done on main thread to avoid race conditions with Tk.
            self.ws_queue.put({"type":"message","raw": message})

        self.ws_app = websocket.WebSocketApp(url,
                                             on_open=on_open,
                                             on_message=on_message,
                                             on_error=on_error,
                                             on_close=on_close)
        # Run in thread
        self.ws_thread = threading.Thread(target=lambda: self.ws_app.run_forever(ping_interval=30, ping_timeout=10), daemon=True)
        self.ws_thread.start()
        self.websocket_status_var.set("Connecting...")
        # schedule queue processor
        try:
            self.nb.after(150, self._process_ws_queue)
        except Exception:
            # fallback to top-level window if nb missing
            self.after(150, self._process_ws_queue)


- Uses fiboevo for features and model (expects fiboevo.add_technical_features and LSTM2Head)
- Thread-safe logging via queue
- Separate Prepare(Data) and Train(Model) workers; also combined Prepare+Train
- Saves artifacts in ./artifacts/

 Asuntos a comprobar:

Implementar websocket real (streaming ticker/push) — necesito URL y protocolo; se puede usar websocket-client o asyncio websockets.

Guardado seguro de api_key/api_secret (usar keyring o cifrado).

Indicador visual de carga/estado del modelo (por ejemplo, icono o texto "Model loaded" en Status).

Posibilidad de reiniciar/recargar el daemon loop tras un load_model_and_scaler si lo deseas.

Añadir opción para que Get Latest Prediction use exactamente la misma preprocessing que el daemon (si cambian feature_cols) — ahora intento preferir daemon.model_meta['feature_cols'] si existe.

Mostrar histórico de predicciones / mini-gráfica (requiere matplotlib embedding).

Mejor manejo de timeframes complejos (semana/mes) — ampliar timeframe_to_seconds.

Automatizar validación de modelo/compatibilidad antes de cargar (uso de meta.json).

Soporte para múltiples tickers/exchanges simultáneos en la pestaña Status (si necesitas monitorizar varios).