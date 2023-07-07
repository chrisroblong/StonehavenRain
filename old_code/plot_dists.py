"""
Plot distributions of extremes from 1km and 5km data for summer.

"""

import xarray
import matplotlib.pyplot as plt
import stonehavenRainLib
import commonLib
import cartopy.crs as ccrs
import scipy.stats
import numpy as np
import pandas as pd

def rgn(dist=50.):
    stonehaven_region = dict()
    for k, v in stonehavenRainLib.stonehaven_crash.items():  # dist  around edinburgh
        stonehaven_region[k] = slice(v - dist*1000, v + dist*1000.)
    return stonehaven_region

def rgn_dist(ds,dist=50.):
    distance = 0.0
    for k, v in  stonehavenRainLib.stonehaven_crash.items():  # iterate over coords.
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
prd_dists = slice('2000','2019')
if read_data:
    # read in data, convert to seasonal data then fix the block max in a 25x25 km block.
    seas_max_5km = comp_seas_max("Edinburgh_extremes.nc").coarsen(projection_x_coordinate=5,projection_y_coordinate=5).max()
    seas_max_1km = comp_seas_max("Edinburgh_extremes_1km.nc").coarsen(projection_x_coordinate=25,projection_y_coordinate=25).max()
    seas_max_1km_15min = comp_seas_max("Edinburgh_extremes_1km_15min.nc").coarsen(projection_x_coordinate=25,projection_y_coordinate=25).max()
    print("Read monthly summary data")

def comp_crit_value(ds):
    return ds.sel(method='nearest',time='2021-06',**stonehavenRainLib.stonehaven_crash)
crit_values = {'5km':comp_crit_value(seas_max_5km),
                '1km_15min':comp_crit_value(seas_max_1km_15min)}
fig,axes = plt.subplots(nrows=1,ncols=2,num='Reg_Clim_max',clear=True,figsize=[11,5])
sm = []
spatial_coords  = ['projection_x_coordinate', 'projection_y_coordinate']
color = {'1km_50km':'blue','5km_50km':'skyblue','1km_150km':'black',
         '5km_150km':'grey','1km_15min_50km':'purple','1km_15min_150km':'fuchsia'}

rgns = {'50km':50.,'150km':150.}
for rname,region in rgns.items():
    for seas_max,name in zip([seas_max_5km,seas_max_1km_15min],['5km','1km_15min']):
        ds = seas_max.sel(time=prd_dists).groupby('time.month').mean()
        ds = rgn_dist(ds,region)
        sm.append(
            ds.mean(spatial_coords).to_pandas().rename(name+'_'+rname)
        )
sm= pd.DataFrame(sm).T
sm.plot.bar(ax=axes[0],color=color)
axes[0].set_title("Mean seasonal max")
axes[0].set_ylabel("mm/hr")

x=np.linspace(5,200,100)
##
distributions = dict()
params=dict()
linestyles={'1km_15min':'solid','5km':'dashed'}
figpp,axpp = plt.subplots(nrows=1,ncols=2,clear=True,num='pp_plot')
d=scipy.stats.genextreme
for rname,region in rgns.items():
    for ds,t,steps in zip([seas_max_5km,seas_max_1km_15min],['5km','1km_15min'],[4,20]):
        #sel_cond = dict(projection_x_coordinate=slice(steps//2, -1, steps), projection_y_coordinate=slice(steps//2, -1, steps))
        sel_cond = dict(projection_x_coordinate=slice(0, -1, steps), projection_y_coordinate=slice(0, -1, steps))
        name = t+'_'+rname
        #datax = rgn_dist(ds.sel(time=(ds.time.dt.month==6)).sel(time=prd_dists),region)
        datax=rgn_dist(ds.sel(time=(ds.time.dt.month==6)).sel(time=prd_dists),region)
        #data = datax.isel(**sel_cond)
        data = datax.values.flatten() #
        data = data[~np.isnan(data)]
        distp=d.fit(data)
        params[name]=distp
        dist=d(*distp)
        distributions[name]=dist
        # plot the return time vs rainfall.
        axes[1].plot(x, 1.0/dist.sf(x), linewidth=2, color=color[name],label=name,linestyle=linestyles[t])
        (xx,yy),fit = scipy.stats.probplot(data,dist=dist)
        yfit= fit[1]+fit[0]*xx
        print(name,fit[0])
        if t.startswith('5km'):
            axpp[0].plot(xx,yfit,color=color[name],linewidth=2)
            axpp[0].scatter(xx,yy,color=color[name],label=name,marker='o',s=2)
        else:
            axpp[1].plot(xx,yfit,color=color[name],linewidth=2)
            axpp[1].scatter(xx, yy, color=color[name], label=name, marker='o', s=2)
axes[1].legend()
for ax in axpp:
    ax.legend()
axes[1].set_yscale('log')
axes[1].set_title("Regional Dist Summer Maxes")
axes[1].set_ylabel("Return Period")
axes[1].set_xlabel("Max Summer rain (mm/hr)")

for xx in [10,100.]:
    axes[1].axhline(xx,color='black',linestyle='dashed')
for key,yy in crit_values.items():
    axes[1].axvline(yy,color='black',linestyle=linestyles[key])

fig.show()
fig.tight_layout()
commonLib.saveFig(fig)

figpp.show()
figpp.tight_layout()
commonLib.saveFig(figpp)

# add in prob-prob plots so we can see if the fits any good. Stealing idea from Shane O'Neil.

## plot the t/s of 1 hr 5 km & 15 min 1km.
fig_ts,ax=plt.subplots(nrows=1,ncols=1,num='edinburgh_ts',figsize=[10,4],clear=True)
fig_scatter,ax_scatter=plt.subplots(nrows=1,ncols=1,num='edinburgh_scatter',figsize=[6,6],clear=True) # plot scatter plot
ax2=ax.twinx()
for name,site in  stonehavenRainLib.sites.items():
    color=  stonehavenRainLib.colors[name]
    ts_1hr = seas_max_5km.sel(method='nearest',**site).sel(time=(seas_max_5km.time.dt.month==6))
    ts_15min = seas_max_1km_15min.sel(method='nearest', **site).sel(time=(seas_max_5km.time.dt.month == 6))
    ts_1hr.plot(ax=ax,color=color,label=name,linestyle='dashed')
    ts_15min.plot(ax=ax2, color=color, label=name)
    ax2.axhline(ts_15min.mean(),color='black',linestyle='dashed')
    print(name,ts_15min.mean().values)
    ax.axhline(ts_1hr.mean(),color='black')
    print(name,ts_1hr.mean().values)
    print(name,np.corrcoef(ts_15min,ts_1hr)[0,1])
    ax_scatter.scatter(ts_15min,ts_1hr,color=color,label=name)

ax.set_title('Summer time maximum precipitation')
ax2.set_title('')
ax2.set_ylabel('15min Precip (mm/hr)')
ax.set_ylabel('Hourly Precip (mm/hr)')

ax2.legend()
ax.set_xlabel("Year")

fig_ts.tight_layout()
fig_ts.show()
commonLib.saveFig(fig_ts)

# labels etc for ax_scatter
ax_scatter.set_ylabel('Hourly Max Precip (mm/hr)')
ax_scatter.set_xlabel('15 Min Max Precip (mm/hr)')
fig_scatter.tight_layout()
fig_scatter.show()
commonLib.saveFig(fig_scatter)