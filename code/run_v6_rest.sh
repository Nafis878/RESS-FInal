#!/bin/sh
# Amendment v6 Parts 3 and 4 (protocol_v6.md). Usage: sh code/run_v6_rest.sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
L=../results/logs
cens() { for d in "$@"; do python3 run6_cens.py "$d" > "$L/v6cens_$d.log" 2>&1; echo "cens $d $?" >> "$L/v6rest_done.txt"; done; }
shift_() { for d in "$@"; do python3 run6_shift.py "$d" 0,1,2 > "$L/v6shift_$d.log" 2>&1; echo "shift $d $?" >> "$L/v6rest_done.txt"; done; }
(shift_ BATTERY NCMAPSS; cens FD001 FD003 PHM08) &
cens BATTERY NCMAPSS FD002 FD004 &
wait
