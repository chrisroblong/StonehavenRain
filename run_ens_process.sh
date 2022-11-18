#!/bin/bash
module load jaspy #load jaspy environment
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p output # where output is going
# run processing jobs for CPM ensemble
rgn="358.5 360.5 2.5 4.5" # region = +/- 1 degree of Edinburgh
TIME='2:00:00' # time job will be in the Q
Q='short-serial' # Q for job
for dir in /badc/ukcp18/data/land-cpm/uk/2.2km/rcp85/*/pr/1hr/latest
do echo "Processing $dir"
   ens=$(echo $dir | sed -E  s'!^.*rcp85/([0-9][0-9]*)/.*!\1!')
   cmd="./ens_seas_max.py  $dir/pr_*_1hr_*.nc --region $rgn -o CPM_seas_max$ens --monitor --verbose --rolling 2 4 8"
   sbatch -p $Q --time=$TIME -o "output/ens_seas_max_$ens.out" $cmd
done
