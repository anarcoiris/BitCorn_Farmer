#!/usr/bin/env python3
"""
csv_to_influx_annotated.py

Lee CSVs con formato:
Unix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount

y genera un CSV anotado compatible con Influx (Flux annotated CSV),
listo para `influx write --bucket ... --file annotated.csv --format csv`.

Uso:
    py -3.10 csv_to_influx_annotated.py --input-dir ../exports/binance_btcusdt/1h/ --out ../exports/annotated/Binance_btcusdt_1h.csv --exchange binance --timeframe 1h

csv_to_influx_annotated_long.py

Convierte CSVs en la carpeta --input-dir (formato esperado:
 Unix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount)
a un CSV anotado compatible con Influx en formato *long* con cabecera:

,result,table,_start,_stop,_time,_value,_field,_measurement,exchange,symbol,timeframe

Uso:
    python csv_to_influx_annotated_long.py --input-dir /path/to/csvs --out out_influx_long.csv \
        --exchange binance --timeframe 1h --measurement prices
"""

from pathlib import Path
import argparse
import pandas as pd
from datetime import datetime, timezone
import csv
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Carpeta con CSVs de velas")
    p.add_argument("--out", required=True, help="Fichero de salida anotado")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--measurement", default="prices")
    return p.parse_args()

def to_rfc3339_from_epoch_ms(ms):
    try:
        msf = float(ms)
    except Exception:
        raise ValueError(f"Invalid epoch ms: {ms}")
    return datetime.utcfromtimestamp(msf / 1000.0).replace(tzinfo=timezone.utc).isoformat()

def read_and_normalize_csv(p: Path):
    # lee CSV y devuelve DataFrame normalizado con columnas:
    # _time (RFC3339), open, high, low, close, volume (opt), symbol
    df = pd.read_csv(str(p))
    # tolerant column mapping
    cols = {c.strip(): c for c in df.columns}
    required = ["Unix", "Symbol", "Open", "High", "Low", "Close"]
    for r in required:
        if r not in cols:
            raise RuntimeError(f"Missing column {r} in {p}")

    out = pd.DataFrame()
    out["_time"] = df[cols["Unix"]].apply(to_rfc3339_from_epoch_ms)
    out["open"]  = pd.to_numeric(df[cols["Open"]], errors="coerce")
    out["high"]  = pd.to_numeric(df[cols["High"]], errors="coerce")
    out["low"]   = pd.to_numeric(df[cols["Low"]], errors="coerce")
    out["close"] = pd.to_numeric(df[cols["Close"]], errors="coerce")

    # volumen: intenta Volume BTC entonces Volume USDT
    if "Volume BTC" in cols:
        out["volume"] = pd.to_numeric(df[cols["Volume BTC"]], errors="coerce")
    elif "Volume USDT" in cols:
        out["volume"] = pd.to_numeric(df[cols["Volume USDT"]], errors="coerce")
    else:
        # si no existe, crear columna NaN (para mantener esquema)
        out["volume"] = pd.NA

    out["symbol"] = df[cols["Symbol"]].astype(str)
    return out

def expand_to_long(df: pd.DataFrame, measurement: str, exchange: str, timeframe: str, start_iso: str, stop_iso: str, table_id: int = 0):
    """
    Convierte df (cada fila = vela) a formato long rows:
    columns: result,table,_start,_stop,_time,_value,_field,_measurement,exchange,symbol,timeframe
    Retorna lista de filas (listas) listas listas (ready para escribir).
    """
    fields = ["open", "high", "low", "close", "volume"]
    rows = []
    for _, r in df.iterrows():
        t_iso = r["_time"]
        symbol = r.get("symbol", "")
        for f in fields:
            v = r.get(f)
            # skip missing values (optional): dejamos filas con value vacío si NaN -> Influx lo ignora
            if pd.isna(v):
                val = ""
            else:
                # ensure numeric formatting plain
                val = f"{float(v):.12g}"
            # produce row: leading empty col for annotations as in Flux examples
            # row = ["", result, table, _start, _stop, _time, _value, _field, _measurement, exchange, symbol, timeframe]
            row = [
                "",            # annotation leading empty
                "",            # result (empty)
                str(table_id), # table id as long (Flux expects integer in #datatype)
                start_iso,     # _start
                stop_iso,      # _stop
                t_iso,         # _time (RFC3339)
                val,           # _value
                f,             # _field
                measurement,   # _measurement
                exchange,      # exchange (tag)
                symbol,        # symbol (tag)
                timeframe      # timeframe (tag)
            ]
            rows.append(row)
    return rows

def write_annotated_long_csv(out_path: Path, all_rows, start_iso, stop_iso):
    """
    Escribe anotaciones y filas en fichero CSV compatible con Flux annotated CSV (long).
    Annotations chosen:
      #group, #datatype, #default
    """
    # columns we will emit (matching expand_to_long)
    cols = ["result","table","_start","_stop","_time","_value","_field","_measurement","exchange","symbol","timeframe"]
    # Build annotation rows
    # group: choose which columns belong to group key (tags + time-related). We'll mark _start/_stop/_time/exchange/symbol/timeframe as group=true
    group_flags = []
    for c in cols:
        if c in ("_start","_stop","_time","exchange","symbol","timeframe"):
            group_flags.append("true")
        else:
            group_flags.append("false")
    # datatype row: align types:
    # result -> string
    # table -> long (int)
    # _start,_stop,_time -> dateTime:RFC3339
    # _value -> double
    # _field,_measurement,exchange,symbol,timeframe -> string
    datatype = [
        "string", "long", "dateTime:RFC3339", "dateTime:RFC3339",
        "dateTime:RFC3339", "double", "string", "string", "string", "string", "string"
    ]
    default_row = ["_result"] + [""] * (len(cols)-1)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # write annotation rows (note leading annotation cell omitted in writer rows in examples; we follow the same style)
        writer.writerow(["#group"] + group_flags)
        writer.writerow(["#datatype"] + datatype)
        writer.writerow(["#default"] + default_row)
        # header row must start with empty annotation col
        writer.writerow([""] + cols)
        # write data rows (each row already includes leading empty annotation cell)
        for r in all_rows:
            writer.writerow(r)

def main():
    args = parse_args()
    p = Path(args.input_dir)
    files = sorted([x for x in p.glob("*.csv")])
    if not files:
        print("No CSV files found in", p, file=sys.stderr)
        sys.exit(1)

    all_dfs = []
    for f in files:
        print("Processing", f)
        try:
            df = read_and_normalize_csv(f)
            all_dfs.append(df)
        except Exception as e:
            print("Error processing", f, e, file=sys.stderr)

    # concat all (keep ordering by time)
    big = pd.concat(all_dfs, ignore_index=True)
    # ensure sort by time (convert to datetime for sorting)
    big["_dt"] = pd.to_datetime(big["_time"])
    big = big.sort_values("_dt").reset_index(drop=True)
    start_iso = big["_time"].iloc[0] if len(big) > 0 else ""
    stop_iso = big["_time"].iloc[-1] if len(big) > 0 else ""
    # Expand to long rows
    rows = expand_to_long(big, measurement=args.measurement, exchange=args.exchange, timeframe=args.timeframe, start_iso=start_iso, stop_iso=stop_iso, table_id=0)
    outp = Path(args.out)
    write_annotated_long_csv(outp, rows, start_iso, stop_iso)
    print("Wrote annotated long CSV to", outp)

if __name__ == "__main__":
    main()
