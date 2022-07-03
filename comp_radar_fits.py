"""
Compute gev fits to radar data and bootstrap estimates of the GEV fits

"""
import edinburghRainLib
import gev_r
import xarray
import numpy as np



file = edinburghRainLib.dataDir / 'radar_precip/summary_1km_15min.nc'
radar_data_all= edinburghRainLib.gen_radar_data(file=file)
radar_fit = gev_r.xarray_gev(radar_data_all.radar, dim='time_index')

# now to sample...
mc_dist=[]
n_monte_carlo=1000
nRadar = radar_data_all.radar.shape[0]
print(f"Monte Carlo Sampling {n_monte_carlo} times")
rng = np.random.default_rng()
for indx in range(0, n_monte_carlo):
    if ((indx+1)%20 ==0):
        print(f"{indx}",end=' ')
    indx_rdr = rng.integers(0,nRadar,size=nRadar) # random sample (with replacement)
    mc_radar_data = radar_data_all.radar.isel(time_index=indx_rdr)  # data > 0 < 200
    # compute empirical fit
    radar_fit_mc = gev_r.xarray_gev(mc_radar_data, dim='time_index')
    mc_dist.append( radar_fit_mc)
mc_dist = xarray.concat(mc_dist, dim='sample')
#save the data
mc_dist.to_netcdf(edinburghRainLib.dataDir/'radar_precip'/'bootstrap_reg_radar_params.nc')
radar_fit.to_netcdf(edinburghRainLib.dataDir/'radar_precip'/'reg_radar_params.nc')
radar_data_all.to_netcdf(edinburghRainLib/'radar_precip'/'reg_radar_rain.nc')
print("\nRadar fit and bootstrapped")

# generate