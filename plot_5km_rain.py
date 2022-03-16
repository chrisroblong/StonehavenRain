"""
plot some diagnosticvs from processed radar data
"""
import pandas as pd
import xarray
import matplotlib.pyplot as plt
import commonLib
import cartopy.crs as ccrs
import scipy.stats
import numpy as np
read_data = True
gen_data = False
if read_data:
    print("Reading data")
    if gen_data:
        print("Generating data -- slow...")
        ds=xarray.open_mfdataset("summary_5km/*monthly.nc",chunks=dict(time=24,projection_y_coordinate=40,projection_x_coordinate=40),combine='nested').sel(**commonLib.edinburgh_region).load()
        ds.to_netcdf("Edinburgh_extremes.nc")
    else:
        ds = xarray.load_dataset("Edinburgh_extremes.nc")
    print("Read monthly summary data")
    seas_max=ds.monthlyMax.sel(**commonLib.edinburgh_region).resample(time='QS-Dec').max().load()
    timeseries=dict()
    sites = commonLib.sites.copy()
    for name,loc in sites.items():
        timeseries[name]=ds.monthlyMax.sel(method='nearest',**loc).resample(time='QS-Dec').max().load()


## make some plots for summer. 
fig,ax = plt.subplots(nrows=1,ncols=1,num='ed_ts',clear=True,figsize=[8,5])
for name, ts in timeseries.items():
    col=commonLib.colors[name]
    ts.sel(time=(ts.time.dt.month==6)).plot.line(color=col,ax=ax,label=name)
ax.legend()
ax.set_title("Seasonal Max Hourly Rainfall (mm/hr)")
fig.tight_layout()
fig.show()
commonLib.saveFig(fig)

proj=ccrs.OSGB()
fig,ax=plt.subplots(nrows=1,ncols=1,clear=True,num='mean_summer_max',figsize=[7,8],subplot_kw=dict(projection=proj))
cmap='Blues'
mn = seas_max.sel(time=(seas_max.time.dt.month == 6))

cbar_kwargs=dict(orientation='horizontal',pad=0.1,label='mm/hr')
mn.mean('time').plot(ax=ax,robust=True,cmap=cmap, cbar_kwargs=cbar_kwargs)
ax.coastlines()
commonLib.std_decorators(ax)
ext = [commonLib.edinburgh_region['projection_x_coordinate'].start,commonLib.edinburgh_region['projection_x_coordinate'].stop,
       commonLib.edinburgh_region['projection_y_coordinate'].start,commonLib.edinburgh_region['projection_y_coordinate'].stop]
ax.set_extent(ext,crs=proj)

fig.tight_layout()
ax.set_title("Mean Summer Max Rain (mm/hr)")
fig.show()
commonLib.saveFig(fig)

## plot reginally aggregated stats.. 
fig,axes = plt.subplots(nrows=1,ncols=2,num='Reg_Clim_max',clear=True,figsize=[8,6])
seas_max.groupby('time.month').mean().mean(['projection_x_coordinate','projection_y_coordinate']).to_pandas().plot.bar(ax=axes[0])
axes[0].set_title("Mean seasonal max")
axes[0].set_ylabel("mm/hr")
x=np.linspace(5,105,100)
seas_max.sel(time=(seas_max.time.dt.month==6)).plot.hist(bins=x,ax=axes[1],density=True)
# let's fit a gev to this
distp=scipy.stats.genextreme.fit(seas_max.sel(time=(seas_max.time.dt.month==6)).values.flatten())
dist=scipy.stats.genextreme(*distp)
axes[1].plot(x,dist.pdf(x),linewidth=2,color='green')
#axes[1]

axes[1].set_yscale('log')
axes[1].set_title("Regional Dist of Summer Maxes")
axes[1].set_ylabel("PDF")
axes[1].set_xlabel("Max Summer rain (mm/hr)")
# plot the return time vs rainfall.
axes[1].plot(x,dist.sf(x),linewidth=2,color='red')
for xx in [1e-2,1e-3]:
    axes[1].axhline(xx,color='black',linestyle='dashed')
for yy in [25,40]:
    axes[1].axvline(yy,color='black',linestyle='dashed')

fig.show()
fig.tight_layout()
commonLib.saveFig(fig)
