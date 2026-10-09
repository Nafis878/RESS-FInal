#!/bin/sh
# Amendment v6 Part 3 (protocol_v6.md), three workers. Usage: sh code/run_v6_cens_all.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
L=../results/logs
cens() { for d in "$@"; do python3 run6_cens.py "$d" > "$L/v6cens_$d.log" 2>&1; echo "cens $d $?" >> "$L/v6cens_done.txt"; done; }
cens FD002 FD001 BATTERY &
cens FD004 FD003 &
cens PHM08 NCMAPSS &
wait
