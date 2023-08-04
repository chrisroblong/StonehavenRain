"""
Compute gev fits to radar data and bootstrap estimates of the GEV fits.
Also compute empirical fit (without bootstrapping)

"""
import stonehavenRainLib
import gev_r
import xarray
import numpy as np
from old_code import emp_dist


def radar_fit(radar_data_all):
    return gev_r.xarray_gev(radar_data_all.radar, dim='time_index')


def emp_rn(rn):
    return emp_dist.empDist(rn.sel(time=time).values.flatten())


# get where stonehaven rain is in the empirical distribution
def print_emp_val(file):
    rn, _ = stonehavenRainLib.get_radar_data(file=file)
    ed_v = float(rn.sel(**stonehavenRainLib.stonehaven_crash, method='nearest').sel(time='2020'))
    ed_rp = 1.0/emp_rn(rn).sf(ed_v)
    print(f"Crash Rain  {ed_v:3.1f} (mm/hr) Emp. RP {ed_rp:3.0f} Years")


# now to sample...
def mc_dist(radar_data_all, n_monte_carlo):
    mc_dist=[]
    nRadar = radar_data_all.radar.shape[0]
    print(f"Monte Carlo Sampling {n_monte_carlo} times")
    rng = np.random.default_rng()
    for indx in range(0, n_monte_carlo):
        if (indx + 1) % 20 == 0:
            print(f"{indx}",end=' ')
        indx_rdr = rng.integers(0,nRadar,size=nRadar) # random sample (with replacement)
        mc_radar_data = radar_data_all.radar.isel(time_index=indx_rdr)  # data > 0 < 200
        # compute empirical fit
        radar_fit_mc = gev_r.xarray_gev(mc_radar_data, dim='time_index')
        mc_dist.append(radar_fit_mc)
    ds = xarray.concat(mc_dist, dim='sample').dropna(dim='sample')  # removed NaNs
    ds['sample'] = range(len(ds['sample']))  # coord label to be able to remove bad samples
    bad_samples = ds.Parameters.where((ds.Parameters.min(dim='time_quant').sel(parameter='location') < 0)).dropna(dim='sample')['sample']
    ds = ds.drop_sel(sample=bad_samples)
    bad_samples = ds.Parameters.where((ds.Parameters.min(dim='time_quant').sel(parameter='scale') < 0)).dropna(dim='sample')['sample']
    ds = ds.drop_sel(sample=bad_samples)
    ds = ds.drop_vars("sample")
    print(ds)
    return ds


def save(radar_data_all, prefix=""):
    mc_dist(radar_data_all, 1000).to_netcdf(stonehavenRainLib.dataDir / 'radar_fits' / f'{prefix}bootstrap_reg_radar_params.nc')
    radar_fit(radar_data_all).to_netcdf(stonehavenRainLib.dataDir / 'radar_fits' / f'{prefix}reg_radar_params.nc')
    radar_data_all.to_netcdf(stonehavenRainLib.dataDir / 'radar_fits' / f'{prefix}reg_radar_rain.nc')
    print("\nRadar fit and bootstrapped")


def no2020(radar_data_all):
    time_to_remove = np.datetime64('2020-06-01')
    return radar_data_all.sel(time=radar_data_all.time != time_to_remove)


time = slice('2005','2022')
file = stonehavenRainLib.dataDir / 'transfer_dir/summary_5km_1h.nc'
radar_data_all = stonehavenRainLib.gen_radar_data(file=file).sel(time=time)
#no2020radar_data_all = no2020(radar_data_all)
print(radar_data_all)
#print(no2020radar_data_all)
save(radar_data_all)
#save(no2020radar_data_all, "no2020")
print_emp_val(file)

