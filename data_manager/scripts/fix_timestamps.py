#!/usr/bin/env python3
"""
fix_timestamps.py

Detecta y corrige timestamps en CSV que están en unidades equivocadas
(extra ceros). Convierte la columna de timestamp al formato milisegundos
desde epoch (13 dígitos) y actualiza la columna de fecha legible.

Uso:
    python fix_timestamps.py input.csv output_fixed.csv
Opciones:
    --ts-col INDEX    (índice 0 por defecto)
    --date-col INDEX  (índice 1 por defecto)
    --sep SEP         (delimitador, por defecto ',')
    --tz TZ           (timezone para mostrar fechas; por defecto 'UTC')
    --min-year YEAR   (mínimo año plausible, default 2000)
    --max-year YEAR   (máximo año plausible, default 2035)
    --keep-ms         (mantener milisegundos en la columna fecha)
"""

from __future__ import annotations
import argparse
import csv
import math
from datetime import datetime, timezone
from typing import Optional, Tuple, List

def detect_and_fix_ts(ts_orig: str, min_year=2000, max_year=2035) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Intenta corregir ts_orig (string) devolviendo:
        (ts_millis (int), iso_datetime_str (str), reason (str))
    Si no se puede interpretar, devuelve (None, None, motivo).
    """
    ts_orig_str = ts_orig.strip()
    if ts_orig_str == "":
        return None, None, "empty"
    # Remove possible non-digit chars
    neg = False
    s = ts_orig_str
    if s.startswith("-"):
        neg = True
        s = s[1:]
    if not s.isdigit():
        # may contain decimal point => try float seconds
        try:
            f = float(ts_orig_str)
            # assume this is seconds (float)
            ms = int(round(f * 1000.0))
            dt = datetime.utcfromtimestamp(ms / 1000.0)
            if min_year <= dt.year <= max_year:
                return ms, dt.isoformat(sep=" "), "interpreted-as-float-seconds"
        except Exception:
            return None, None, "non-digit"
    # Now s is digits
    # original integer
    try:
        val = int(s)
    except Exception:
        return None, None, "int-parse-failed"
    if neg:
        val = -val

    # If already 13 digits (milliseconds), test plausibility
    def ms_to_year(ms: int) -> int:
        try:
            return datetime.utcfromtimestamp(ms/1000.0).year
        except Exception:
            return -9999

    cand_ms = None
    reason = None

    # Try divisors by powers of 10 (k = 0..6): ms_candidate = val // (10**k)
    # (this covers cases: val in ms (k=0), microseconds (k=3), nanoseconds (k=6), or other)
    for k in range(0, 7):
        divisor = 10 ** k
        ms_candidate = val // divisor
        # sanity: ms_candidate must be > 0 and not absurdly small
        if ms_candidate <= 0:
            continue
        year = ms_to_year(ms_candidate)
        if min_year <= year <= max_year:
            cand_ms = ms_candidate
            reason = f"divided_by_10^{k} (year={year})"
            break

    # If not found, also try multipliers (rare): e.g., val is seconds (10 digits) and want ms multiply by 1000
    if cand_ms is None:
        # consider multipliers 10^k for k 1..3
        for k in range(1, 4):
            ms_candidate = val * (10 ** k)
            year = ms_to_year(ms_candidate)
            if min_year <= year <= max_year:
                cand_ms = ms_candidate
                reason = f"multiplied_by_10^{k} (year={year})"
                break

    if cand_ms is None:
        # As last resort, if val already looks like ms length (13 digits) accept it if close
        if 1_000_000_000_000 <= abs(val) <= 9999999999999:
            year = ms_to_year(val)
            if min_year-5 <= year <= max_year+5:
                cand_ms = val
                reason = f"assume_as_is (year={year})"

    if cand_ms is None:
        return None, None, "no-plausible-conversion"

    # Build ISO datetime (UTC), include seconds; optionally include ms
    try:
        dt = datetime.utcfromtimestamp(cand_ms/1000.0).replace(tzinfo=timezone.utc)
        iso_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        # append milliseconds if present
        ms_part = cand_ms % 1000
        if ms_part != 0:
            iso_str = f"{iso_str}.{ms_part:03d}"
        return int(cand_ms), iso_str, reason
    except Exception as e:
        return None, None, f"datetime-conversion-failed:{e}"

def process_csv(input_path: str, output_path: str, ts_col:int=0, date_col:int=1,
                sep: str=',', min_year=2000, max_year=2035, keep_ms=False) -> None:
    """
    Procesa CSV línea a línea, corrige timestamps y escribe CSV nuevo.
    Crea un archivo de log con las filas modificadas: output_path + '.fixlog.txt'
    """
    log_lines: List[str] = []
    total = 0
    fixed = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin, delimiter=sep)
        writer = csv.writer(fout, delimiter=sep)
        for row in reader:
            total += 1
            # guard: skip empty lines
            if not row:
                writer.writerow(row)
                continue
            # if row shorter than ts_col/date_col, just copy
            if len(row) <= ts_col:
                writer.writerow(row)
                continue
            ts_orig = row[ts_col]
            ts_ms, iso_str, reason = detect_and_fix_ts(ts_orig, min_year=min_year, max_year=max_year)
            if ts_ms is not None:
                # update timestamp column to canonical ms (int)
                row[ts_col] = str(ts_ms)
                # update or create date column
                # if date_col index exists in row, replace; else append
                if len(row) > date_col:
                    # optionally preserve ms in iso string or trunc
                    if not keep_ms and '.' in iso_str:
                        # drop ms fractional part
                        base_dt = iso_str.split('.')[0]
                        row[date_col] = base_dt
                    else:
                        row[date_col] = iso_str
                else:
                    # append up to date_col index with empty values
                    while len(row) < date_col:
                        row.append('')
                    row.append(iso_str if keep_ms or '.' in iso_str else iso_str.split('.')[0])
                if reason:
                    log_lines.append(f"Row {total}: ts_orig={ts_orig} -> {ts_ms} ({iso_str}) reason={reason}")
                fixed += 1
            else:
                log_lines.append(f"Row {total}: ts_orig={ts_orig} SKIPPED reason={reason}")

            writer.writerow(row)

    # write log
    log_path = output_path + ".fixlog.txt"
    with open(log_path, 'w', encoding='utf-8') as fh:
        fh.write(f"Processed {total} rows, fixed {fixed} rows\n")
        fh.write("\n".join(log_lines))
    print(f"Done. Processed {total} rows, fixed {fixed}.")
    print(f"Output saved to: {output_path}")
    print(f"Log saved to: {log_path}")

def main():
    parser = argparse.ArgumentParser(description="Fix timestamps in CSV (convert to milliseconds + update date column).")
    parser.add_argument("input", help="Input CSV path")
    parser.add_argument("output", help="Output CSV path (fixed)")
    parser.add_argument("--ts-col", type=int, default=0, help="Zero-based index of timestamp column (default 0)")
    parser.add_argument("--date-col", type=int, default=1, help="Zero-based index of date column to update (default 1)")
    parser.add_argument("--sep", default=",", help="CSV separator (default ',')")
    parser.add_argument("--min-year", type=int, default=2000, help="Minimum plausible year (default 2000)")
    parser.add_argument("--max-year", type=int, default=2035, help="Maximum plausible year (default 2035)")
    parser.add_argument("--keep-ms", action="store_true", help="Keep milliseconds in date column if present")
    args = parser.parse_args()
    process_csv(args.input, args.output, ts_col=args.ts_col, date_col=args.date_col,
                sep=args.sep, min_year=args.min_year, max_year=args.max_year,
                keep_ms=args.keep_ms)

if __name__ == "__main__":
    main()
