"""
Plot the mean summer hourly extreme from the radar data.
Actually plot median to get rid of what looks like artifacts.
Need to plot radar stations & edinburgh...
and inset plot for region around Edinburgh -- or just second plot.

Some dodgy data that will need finding and (ignoring)
see https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-climate-extremes which should suggest some reasonable values
"""

import edinburghRainLib
import xarray
import matplotlib.pyplot as plt
import numpy
import pathlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pandas as pd

#annualDir = commonLib.dataDir/'radar_max/annual_max'
annualDir = pathlib.Path('annual_max')
ds = xarray.open_mfdataset(annualDir.glob("*.nc")).load()
proj=ccrs.OSGB()
levels=[]



rgn=[]
sel_rgn=dict()
for key,value in edinburghRainLib.edinburgh_castle.items():
    r=[value-50e3,value+50e3]
    rgn.append(r)
    sel_rgn[key]=slice(r[0],r[1])
fig,axes = plt.subplots(nrows=2,ncols=2,clear=True,num='UKradar_extreme',figsize=[12,8],subplot_kw=dict(projection=proj))
for ax,var,title in zip(axes.flatten(),
                        [ds.maxV.mean('time'),ds.maxV.max('time'),ds.maxV.sel(time='2021'),ds.maxVTime.sel(time='2021').dt.dayofyear],
                        ['Mean','Max','2021','2021 time']):
    var.sel(**sel_rgn).plot(ax=ax, robust=True, transform=proj)
    ax.set_extent(rgn,crs=proj)
    ax.set_title(f"{title} Summer Max Hourly Rainfall")
    edinburghRainLib.std_decorators(ax)

fig.show()

## quality control

fig,axes = plt.subplots(nrows=1,ncols=2,clear=True,num='UKradar_QC',figsize=[12,8],subplot_kw=dict(projection=proj))
indx=ds.maxV.argmax('time',skipna=False)
maxTime = ds.maxVTime.isel(time=indx)
for ax,var,title in zip(axes.flatten(),
                        [ds.maxV.max('time'),maxTime.dt.year],
                        ['Max','Yr of max']):
    var.sel(**rgn).plot(ax=ax, robust=True, transform=proj)

    #ax.set_extent([5e4,5e5,5e5,1.1e6],crs=proj)
    ax.set_extent([-4,-2.5,55.5,56.5],crs=ccrs.PlateCarree())
    ax.set_title(f"{title} Summer Max Hourly Rainfall")
    edinburghRainLib.std_decorators(ax)

fig.show()

