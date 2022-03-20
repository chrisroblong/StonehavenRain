"""
Plot distributions of extremes from 1km and 5km data for summer.

"""

import xarray
import matplotlib.pyplot as plt
import commonLib
import cartopy.crs as ccrs
import scipy.stats
import numpy as np
import pandas as pd

def rgn(dist=50.):
    edinburgh_region = dict()
    for k, v in commonLib.edinburgh_castle.items():  # 50km around edinburgh
        edinburgh_region[k] = slice(v - dist*1000, v + dist*1000.)
    return edinburgh_region

def rgn_dist(ds,dist=50.):
    distance = 0.0
    for k, v in commonLib.edinburgh_castle.items():  # iterate over coords.
        distance = distance+(ds[k]-v)**2
    distance = np.sqrt(distance)
    L = (distance <= dist*1e3)
    return ds.where(L)
def comp_seas_max(file,time=None):
    """
    REad in file and compute seasonal max
    :param file: file path to dataset
    :return: the seasonal max value for the monthlyMax in Edinburgh region.
    """
    ds = xarray.load_dataset(file)
    seas_max = ds.monthlyMax.resample(time='QS-Dec').max().load()
    if time is not None:
        seas_max = seas_max.sel(time=time)
    return seas_max

read_data = True
if read_data:
    seas_max_5km = comp_seas_max("Edinburgh_extremes.nc",time=slice('2000','2019'))
    seas_max_1km = comp_seas_max("Edinburgh_extremes_1km.nc",time=slice('2000','2019'))
    seas_max_1km_15min = comp_seas_max("Edinburgh_extremes_1km_15min.nc",time=slice('2000','2019'))
    print("Read monthly summary data")

fig,axes = plt.subplots(nrows=1,ncols=2,num='Reg_Clim_max_1km',clear=True,figsize=[8,6])
sm = []
spatial_coords  = ['projection_x_coordinate', 'projection_y_coordinate']
color = {'1km_50km':'blue','5km_50km':'skyblue','1km_25km':'black',
         '5km_25km':'grey','1km_15min_50km':'purple','1km_15min_25km':'fuchsia'}

rgns = {'50km':50.,'25km':25.}
for rname,region in rgns.items():
    for seas_max,name in zip([seas_max_1km,seas_max_5km,seas_max_1km_15min],['1km','5km','1km_15min']):
        ds = seas_max.groupby('time.month').mean()
        ds = rgn_dist(ds,region)
        sm.append(
            ds.mean(spatial_coords).to_pandas().rename(name+'_'+rname)
        )
sm= pd.DataFrame(sm).T
sm.plot.bar(ax=axes[0],color=color)
axes[0].set_title("Mean seasonal max 1km")
axes[0].set_ylabel("mm/hr")

x=np.linspace(5,150,150)
##
for rname,region in rgns.items():
    for ds,t in zip([seas_max_1km,seas_max_5km,seas_max_1km_15min],['1km','5km','1km_15min']):
        name = t+'_'+rname
        data = rgn_dist(ds.sel(time=(ds.time.dt.month==6)),region)
        data = data.values.flatten() #
        data = data[~np.isnan(data)]
        distp=scipy.stats.genextreme.fit(data)
        dist=scipy.stats.genextreme(*distp)
        # plot the return time vs rainfall.
        axes[1].plot(x, 1.0/dist.sf(x), linewidth=2, color=color[name],label=name)

axes[1].legend()
axes[1].set_yscale('log')
axes[1].set_title("Regional Dist Summer Maxes")
axes[1].set_ylabel("Return Period")
axes[1].set_xlabel("Max Summer rain (mm/hr)")

for xx in [100,1000.]:
    axes[1].axhline(xx,color='black',linestyle='dashed')
for yy in [35,50]:
    axes[1].axvline(yy,color='black',linestyle='dashed')

fig.show()
fig.tight_layout()
commonLib.saveFig(fig)