#!/bin/bash
# run combination jobs.  Before running.
# need jaspy active:
# module load jaspy
# and need current dir in python search path:
# export PYTHONPATH=$PYTHONPATH:$PWD
dirs="summary_5km_1h"
#dirs="summary_5km_15Min"
Q='high-mem --mem=100000' # high mem q with a 100 GB of RAM
time="4:00:00" # four hours should be enough
for dir in $dirs
do

  cmd="./combine_summary.py --verbose ${dir}  ${dir}.nc"
  cmd="sbatch -p $Q --time=$time -o output/${dir}.out ${cmd}"
  echo "$cmd"
  output=$($cmd)
  echo "Status: $? Output: $output"
done
