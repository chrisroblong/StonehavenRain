#!/bin/bash
module load jaspy #load jaspy environment
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p output # where output is going
# run processing jobs for CPM ensemble
rgn="359 361 3.5 5.5" # region = +/- 1 degree of Stonehaven
TIME='12:00:00' # time job will be in the Q
Q='short-serial' # Q for job
for dir in /badc/ukcp18/data/land-cpm/uk/2.2km/rcp85/*/pr/1hr/latest
do echo "Processing $dir"
   ens=$(echo $dir | sed -E  s'!^.*rcp85/([0-9][0-9]*)/.*!\1!')
   outdir=CPM$ens
   mkdir -p $outdir
   for range in '1980 2020' '2020 2040' '2060 2080'
   do 
       echo $range
       r1=$(echo $range| awk '{print $1}')
       cmd="./ens_seas_max.py  $dir/pr_*_1hr_*.nc --region $rgn -o $outdir/CPM_pr_seas_max$ens --monitor --verbose --rolling 2 4 8 --range $range"
       echo "./ens_seas_max.py  $dir/pr_\*_1hr_\*.nc --region $rgn -o $outdir/CPM_pr_seas_max$ens --monitor --verbose --rolling 2 4 8 --range $range"
       outfile=output/"ens_seas_max_"$ens"_"$r1".out"
       echo sbatch -p $Q -t $TIME -o $outfile
       sbatch -p $Q -t $TIME -o  $outfile $cmd
   done
done
