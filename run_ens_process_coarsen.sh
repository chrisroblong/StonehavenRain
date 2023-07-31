#!/bin/bash
module load jaspy #load jaspy environment
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p output # where output is going
# run processing jobs for CPM ensemble
rgn="359 361 3.5 5.5" # region = +/- 1 degree of Stonehaven
TIME='12:00:00' # time job will be in the Q
Q='high-mem --mem=100000' # Q for job
for dir in /badc/ukcp18/data/land-cpm/uk/2.2km/rcp85/*/pr/1hr/latest
do echo "Processing $dir"
   ens=$(echo $dir | sed -E  s'!^.*rcp85/([0-9][0-9]*)/.*!\1!')
   outdir=data/CPM/CPM$ens
   mkdir -p $outdir
   for range in '1981 2020' '2021 2040' '2041 2060' '2061 2080'
    do
    for coarsen in '1' '2' '3'
     do
         echo $range
         r1=$(echo $range| awk '{print $1}')
         cmd="./ens_seas_max_coarsen.py  $dir/pr_*_1hr_*.nc --region $rgn -o $outdir/coarsen_${coarsen}_CPM_pr_seas_max$ens --monitor --verbose --rolling 2 4 8 --range $range --coarsen $coarsen"
         echo "./ens_seas_max_coarsen.py  $dir/pr_\*_1hr_\*.nc --region $rgn -o $outdir/coarsen_${coarsen}_CPM_pr_seas_max$ens --monitor --verbose --rolling 2 4 8 --range $range --coarsen $coarsen"
         outfile=output/"ens_seas_max_coarsen_"$ens"_"$r1".out"
         echo sbatch -p $Q -t $TIME -o $outfile
         sbatch -p $Q -t $TIME -o  $outfile $cmd
     done
    done
done
