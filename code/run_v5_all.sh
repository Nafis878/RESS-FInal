#!/bin/sh
# Amendment v5 runs (protocol_v5.md): four workers; existing outputs are skipped. Usage: sh code/run_v5_all.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
L=../results/logs
worker() { for d in "$@"; do python3 run5.py "$d" > "$L/v5_$d.log" 2>&1; echo "$d $?" >> "$L/v5_done.txt"; done; }
worker FD002 &
worker FD004 &
worker PHM08 FD001 &
worker BATTERY NCMAPSS FD003 &
wait
