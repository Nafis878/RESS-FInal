#!/bin/sh
# amendment experiments: repetition 0 (primary) for every dataset first, then repetitions 1 and 2; four processes at a time
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
run() { python3 run2.py "$1" "$2" >> ../results/logs/v2_$1.log 2>&1; echo "$1 rep $2 exit $?" >> ../results/logs/v2_done.txt; }
for rep in 0 1 2; do
  run FD002 $rep & run FD004 $rep & run PHM08 $rep & run NCMAPSS $rep & wait
  run BATTERY $rep & run FD001 $rep & run FD003 $rep & wait
done
