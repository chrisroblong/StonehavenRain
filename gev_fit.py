"""
Do a GEV fit
"""
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import xarray
import pathlib
#import dask
import seaborn as sns
import pandas as pd
import edinburghRainLib

def gev_fit(x,**kwargs):
    """
    Do GEV fit and return numpy array
    :param x: Data to be fit
    :return: array of parameters
    """
    d=x[~np.isnan(x)]# remove nan from data
    fit = scipy.stats.genextreme.fit(d,**kwargs)
    return np.array(fit)


def xarray_gev(ds,**kwargs):
    ds_fit = xarray.apply_ufunc(gev_fit, ds, input_core_dims=[['time_ensemble']], output_core_dims=[['parameter']],
                                #dask_gufunc_kwargs=dict(output_sizes=dict(parameter=3)),dask='allowed',
                                vectorize=True,kwargs=kwargs)
    return ds_fit
#if __name__ == '__main__':
#    client = dask.distributed.Client(n_workers=8, threads_per_worker=2, memory_limit='1GB')
#    print(client)
generate =False
files = list((edinburghRainLib.dataDir/"FullSlices/").glob("full_summer_slice*.nc"))
fits = dict()
data_set = dict()
for file in files:
    ds = xarray.load_dataset(file).sel(grid_longitude=slice(358.61, 360.61), grid_latitude=slice(2, 5))
    # make ensemble_member and time a coord
    idx = pd.MultiIndex.from_arrays([ds.time.values, ds.ensemble_member.values], names=['time2', 'ensemble_member'])
    # generate multi-index with time and ensemble member
    ds=ds.drop_vars(['ensemble_member']).\
        assign_coords(time=idx).unstack().rename_dims(time2='time').rename_vars(time2='time')
    # rest time to the multi-index, unstack the multi-time and then rename time2 (as dim and var) to time
    # and then store it.
    data_set[file] = ds
    gevFile = edinburghRainLib.outdir / (file.stem + '_gev_fit.nc')
    if generate: # generate the fit -- very slow.
        print("Processing ",file)
        print("ds.pr.shape",ds.pr.shape)
        fit = xarray_gev(ds.pr.stack(time_ensemble=['time','ensemble']),loc=8,scale=5)
        fits[file]=fit
        fit.to_netcdf(gevFile)
        print(f"Generated {gevFile}")
    else:
     # just read cached data.
        fits[file]=xarray.load_dataset(gevFile).pr

grid_coords=['grid_longitude','grid_latitude']
for k,v in fits.items():
    print(k.name,v.mean(grid_coords).values,v.std(grid_coords).values/np.sqrt(v.count(grid_coords).values))

for file in files:
    print((fits[file]/fits[files[0]]).mean(grid_coords).values)

## let's do some scatter plots on the distributions
# in rotated co-ords Edinburgh is 3.5 lat, 359.61 long
sel=dict(grid_longitude=slice(359.11,360.11),grid_latitude=slice(3,4))
delta=0.25
sel=dict(grid_longitude=slice(359.61-delta,359.61+delta),grid_latitude=slice(3.5-delta,3.5+delta))
alpha=1
fig,axis = plt.subplots(nrows=1,ncols=3,num='scatter_params',clear=True,figsize=[10,5])
colors=['red','blue']
pname=['shape','location','scale']
for param,ax in enumerate(axis):
    f2=fits[files[2]].sel(parameter=param,**sel)
    f1=fits[files[1]].sel(parameter=param,**sel)
    f0=fits[files[0]].sel(parameter=param,**sel)
    for f,l,c in zip([f1,f2],['2020-2040','2060-2080'],colors):
        ax.scatter(f0,f,s=3,marker='.',label=l,color=c,alpha=alpha)
        ax.set_title(pname[param])
    # compute fit and plot
    x = np.linspace(*ax.get_xlim(), 100)
    for f,c in zip([f1,f2],colors):
        x1,x0 = np.polyfit(f0.values.flatten(), f.values.flatten(), deg=1)
        y=x0 + x1*x
        print(x0,x1)
        #ax.plot(x,y,color=c)
    # plot 1-1 lin
    #ax.plot(x,x,color='black',linewidth=2,linestyle='dashed')
    for sc,c in zip([1.15,1.3],colors):
        ax.plot(x,x*sc,color=c,linewidth=2,linestyle='dashed')

    ax.set_aspect('equal', adjustable='box')
    print("----------------------")

ax.legend()
fig.tight_layout()
fig.show()

## get sim_cet and convert to seasonal-mean
def qsat(temperature):
    """

    :param T:
    :return:
    """
    es=6.112*np.exp(17.6*temperature/(temperature+243.5))
    return es


sim_cet=xarray.load_dataset(edinburghRainLib.dataDir/'cet_cpm.nc')
sim_cet = sim_cet.tas.resample(time='QS-DEC').mean().dropna('time') # seasonal mean
sim_cet = sim_cet.sel(time=(sim_cet.time.dt.month==6)) # summer
max_precip= xarray.concat(data_set.values(),dim='time').pr.drop_vars('ensemble_member_id') # get rid of the string id
sat_hum= qsat(sim_cet)
plt.figure(num='cet_mn_extreme_scatter',clear=True)
for time in [1,2,5]:
    mn_sat_hum=sat_hum.coarsen(time=time).mean()
    mn_sat_hum /= mn_sat_hum.mean()
    for space in [1,2,5,10]:
        mn_extreme = max_precip.coarsen(grid_longitude=space,grid_latitude=space,time=time,boundary='trim').max().mean(grid_coords).T
        mn_extreme /= mn_extreme.mean()
        if time in [1] and space in [1,10]:
            plt.scatter(mn_sat_hum,mn_extreme,color='k',marker='o',s=5)
        reg=scipy.stats.linregress(mn_sat_hum.values.flatten(),mn_extreme.values.flatten())
        reg_robust = scipy.stats.theilslopes(mn_extreme.values.flatten(),x=mn_sat_hum.values.flatten())
        print(f"time {time} space {space}")
        print(
            f"{reg.slope:4.2f} ({ (reg.slope - 2 * reg.stderr):4.2f} - { (reg.slope + 2 * reg.stderr):4.2f}) R**2 {(reg.rvalue ** 2) * 100:3.1f}")
        print(f"Thiel: {reg_robust[0]:4.2f} ({ (reg_robust[2]):4.2f} - {(reg_robust[3]):4.2f}) ")
# x=np.linspace(*plt.xlim())
# es=qsat(x)
# es=es/np.mean(es)
# plt.plot(x,es,color='blue')


plt.xlabel('QSAT(CET) (g/kg)')
plt.ylabel("Mn Hourly Extreme (mm/hr) ")
plt.show()



