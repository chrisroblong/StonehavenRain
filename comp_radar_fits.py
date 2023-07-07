"""
Compute gev fits to radar data and bootstrap estimates of the GEV fits.
Also compute empirical fit (without bootstrapping)

"""
import stonehavenRainLib
import gev_r
import xarray
import numpy as np
from old_code import emp_dist

time=slice('2005','2020')
file = stonehavenRainLib.dataDir / 'radar_precip/summary_1km_15min.nc'
radar_data_all= stonehavenRainLib.gen_radar_data(file=file).sel(time=time)

radar_fit = gev_r.xarray_gev(radar_data_all.radar, dim='time_index')
rn,mx = stonehavenRainLib.get_radar_data(file=file)
emp_rn = emp_dist.empDist(rn.sel(time=time).values.flatten())
# print out edinburgh rain value and where it is in the empirical distribution
ed_v = float(rn.sel(**stonehavenRainLib.stonehaven_crash,method='nearest').sel(time='2021'))
ed_rp = 1.0/emp_rn.sf(ed_v)

print(f"Castle Rain  {ed_v:3.1f} (mm/hr) Emp. RP {ed_rp:3.0f} Years")
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
mc_dist.to_netcdf(stonehavenRainLib.dataDir/'radar_precip'/'bootstrap_reg_radar_params.nc')
radar_fit.to_netcdf(stonehavenRainLib.dataDir/'radar_precip'/'reg_radar_params.nc')
radar_data_all.to_netcdf(stonehavenRainLib.dataDir/'radar_precip'/'reg_radar_rain.nc')
print("\nRadar fit and bootstrapped")

# generate