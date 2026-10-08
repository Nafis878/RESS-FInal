#!/bin/sh
# equal-information comparators (protocol_v3.md); repetition 0 first
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
for rep in 0 1 2; do for d in FD002 FD004 PHM08 NCMAPSS BATTERY FD001 FD003; do
  python3 run3.py $d $rep >> ../results/logs/v3_$d.log 2>&1; echo "$d rep $rep exit $?" >> ../results/logs/v3_done.txt
done; done
