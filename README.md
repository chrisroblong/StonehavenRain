# Code used in Tett et al,2023 

## Code used to process CPM and radar data on JASMIN

* process_radar_data -- processes the radar data to generate datasets of monthly max, monthly mean and time of max at each cell.
* combine_summary -- combines all the radar summary data into one file 
* run_combine.sh -- shell script to run combine_summary using slurm (as need lots of memory)
* run_jobs -- run multiple parallel jobs to process the radar data. Uses SLURM.

* ens_seas_max -- processes CPM data to find seasonal max, mean and time of max at each cell. Runs on one ensemble at a time.
* run_ens_process.sh -- run lots of parallel jobs to process the CPM data using ens_seas_max and SLURM.

* comp_cpm_ts -- compute monthly mean temperatures & precipitation from CPM data. 
    Computes CET, CPM domain mean and Edinburgh region means for both (though CET precipitation is not very useful). 


## Libraries 

* edinburghRainLib -- main library. Contains bits of code that appear general + various configuration info. 
    You will likely need to edit this file if you want to use this software.
* gev_r -- code to support doing GEV fits using R + some derived calculations. 
You need to install R on your computer and  modify this library to point to R. 

## Processing & plots for SI

* analyse_cpm_data_rolling -- fit GEV's to CPM maxRain data and bootstrap the uncertainties to get samples of parameters. 
* comp_radar_fits -- fit GEV distributions to regional extreme radar data. Also does bootstrap to get samples of parameters.
* radar_qc -- plot figure showing radar data is mostly OK.

## Main figures

* plot_edinburgh_geography -- Plot edinburgh image, plan of castle and map of SE Scotland. 
* plot_edinburgh_precip_dist -- Plot precipitation for 4th july 2021 and distributions of radar regional extremes data.
* plot_intensity_risk_ratios -- Plot intensity and probability risk ratios 

------
## Additional Code
* comp_rotated_coords --compute co-ordinates for CET locations. Done as JASMIN does not have the library. 
    Values then got inserted (manually) into edinburghRainLib
* create_counties -- Create county shape files from OS data. Used to generate high enough resolution  county data. 
* radar_station_metadata.xlxs -- Radar station metadata