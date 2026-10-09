#!/bin/sh
# Amendment v6 Part 1 runs (protocol_v6.md): four workers; existing outputs are skipped. Usage: sh code/run_v6_lb_all.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
L=../results/logs
worker() { for d in "$@"; do python3 run6_lb.py "$d" 0,1,2 > "$L/v6lb_$d.log" 2>&1; echo "$d $?" >> "$L/v6lb_done.txt"; done; }
worker FD002 &
worker FD004 PHM08 &
worker BATTERY FD001 &
worker NCMAPSS FD003 &
wait
