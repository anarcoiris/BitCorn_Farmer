#!/usr/bin/env python3
"""
csv_to_kafka_producer.py

Lee CSVs en una carpeta y publica mensajes JSON en Kafka con clave (symbol|timeframe)
para mantener orden por partición. Este producer usa kafka-python.

Uso:
    py -3.10 csv_to_kafka_producer.py --input-dir Uso:

    py -3.10 csv_to_kafka_producer.py --input-dir ..\exports\binance_btcusdt\1h --topic prices --bootstrap kafka:9092 --exchange binance --timeframe 1h --speed 0.0

speed: delay (s) entre mensajes; 0.0 (to) enviar lo más rápido posible

"""

import argparse
from pathlib import Path
import json
import time
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime
from typing import Optional

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="prices")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--speed", type=float, default=0.0, help="delay between messages in seconds")
    p.add_argument("--key-by-symbol", action="store_true", help="use symbol as kafka key (ensures ordering per symbol)")
    p.add_argument("--batch-size", type=int, default=1000)
    return p.parse_args()

def to_epoch_ms(val) -> int:
    # Accept numeric epoch ms or parseable date string
    try:
        return int(float(val))
    except Exception:
        # try parse as datetime string
        dt = pd.to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        return int(dt.timestamp() * 1000)

def message_from_row(row, exchange, timeframe):
    # mapping tolerant
    out = {
        "exchange": exchange,
        "symbol": str(row.get("Symbol") or row.get("symbol")),
        "timeframe": timeframe,
    }
    # timestamp prefer 'Unix' numeric epoch ms
    if "Unix" in row:
        out["ts_candle"] = int(row["Unix"])
    elif "timestamp" in row:
        out["ts_candle"] = to_epoch_ms(row["timestamp"])
    else:
        out["ts_candle"] = to_epoch_ms(row.get("Date") or row.get("date") or row.get("time"))

    # numeric fields
    for k in ("Open","High","Low","Close","Volume BTC","Volume USDT","tradecount"):
        if k in row:
            kk = k.lower().replace(" ", "_")
            try:
                out_field = float(row[k]) if k != "tradecount" else int(row[k])
            except Exception:
                out_field = None
            out[kk] = out_field
    # also provide deterministic field names consumer expects
    # normalize to 'open','high','low','close','volume'
    if "open" not in out and "Open" in row:
        out["open"] = float(row["Open"])
    # mapping volume: prefer 'Volume BTC' as main volume
    if "volume" not in out:
        if "Volume BTC" in row:
            out["volume"] = float(row["Volume BTC"])
        elif "volume_btc" in row:
            out["volume"] = float(row["volume_btc"])
        elif "Volume USDT" in row:
            out["volume"] = float(row["Volume USDT"])
    # add extras
    out["raw_row"] = None  # optional
    return out

def main():
    args = parse_args()
    p = Path(args.input_dir)
    files = sorted(p.glob("*.csv"))
    if not files:
        print("No files", p)
        return

    producer = KafkaProducer(
        bootstrap_servers=[args.bootstrap],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if isinstance(k, str) else k,
        linger_ms=10,
        acks='all',
    )

    total = 0
    for f in files:
        print("Reading", f)
        df = pd.read_csv(f)
        for idx, r in df.iterrows():
            msg = message_from_row(r, args.exchange, args.timeframe)
            key = None
            if args.key_by_symbol:
                key = f"{msg['symbol']}|{args['timeframe']}" if False else f"{msg['symbol']}|{args['timeframe']}"
                # safer:
                key = f"{msg['symbol']}|{args.timeframe}"
            # send
            try:
                producer.send(args.topic, value=msg, key=key)
                total += 1
            except Exception as e:
                print("Send failed", e)
            if args.speed > 0:
                time.sleep(args.speed)
            # flush periodically
            if total % args.batch_size == 0:
                producer.flush()
        # flush after file
        producer.flush()
    print("Published", total, "messages")
    producer.close()

if __name__ == "__main__":
    main()
