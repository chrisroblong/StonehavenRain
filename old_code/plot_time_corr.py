"""
Plot the fraction of times at a cell where the extreme occurs more than 6 hours in time away from extremes at Edinburgh Castle
Do for all 4 seasons.
"""
import xarray

import edinburghRainLib
import commonLib
import numpy as np
import  matplotlib.pyplot as plt
import cartopy.crs as ccrs

def prob_same_event(file,time_sep=6.,time=None):
    """

    :param file: file to read in
    :param time_sep: Time separation for event to be independent.
    :param time: If not None the times to be used.
    :return:  fraction of times where seasonal extremes within 6 hours of Edinburgh Castle extreme.
    """
    monthlyDS = xarray.load_dataset(file)
    if time is not None:
        monthlyDS=monthlyDS.sel(time=time)
    ddseas = monthlyDS.resample(time='QS-Dec').map(commonLib.time_process,
                                                   varPrefix='monthly')
    ed = ddseas.MaxTime.sel(method='nearest', **commonLib.edinburgh_castle)
    dt = np.abs(ed - ddseas.MaxTime) / np.timedelta64(1,'h')
    p=(dt < time_sep).groupby('time.month').mean()
    return  p

time_prd=slice('2004','2020')
#p_1km  = prob_same_event('Edinburgh_extremes_1km.nc',time=time_prd)
p_1km_15min  = prob_same_event('Edinburgh_extremes_1km_15min.nc',time=time_prd)
p_5km  = prob_same_event('Edinburgh_extremes.nc',time=time_prd)
proj=ccrs.OSGB()
levels=np.array([0.0,0.05,0.1,0.2,0.4,0.6,0.8])

kw_cbar=dict(orientation='horizontal',fraction=0.05,pad=0.1)
cmap='Blues'
spatial_coords  = ['projection_x_coordinate', 'projection_y_coordinate']
fig, axes = plt.subplots(nrows=1, ncols=2, clear=True, sharex=True, sharey=True,
                         num=f'Edinburgh_p_same_event', figsize=[11, 7],
                         subplot_kw=dict(projection=proj))



ext = [edinburghRainLib.edinburgh_region['projection_x_coordinate'].start,edinburghRainLib.edinburgh_region['projection_x_coordinate'].stop,
       edinburghRainLib.edinburgh_region['projection_y_coordinate'].start,edinburghRainLib.edinburgh_region['projection_y_coordinate'].stop]
dist = [50,100,150]
for ax,title,var in zip(axes.flatten(),['1km_15min','5km'],[p_1km_15min,p_5km]):


    ax.set_extent(ext, crs=proj)

    ##cm = var.sel(month=6).plot.contourf(ax=ax, cmap=cmap, levels=levels, transform=proj, add_colorbar=False)
    cm = var.sel(month=6).plot(ax=ax, cmap=cmap, levels=levels, transform=proj, add_colorbar=False)
    ##cl = var.sel(month=6).plot.contour(ax=ax, colors='black', levels=levels, transform=proj, add_colorbar=False)
    ax.set_title(f"{title}")

    edinburghRainLib.std_decorators(ax)  # put std stuff on axis
fig.colorbar(cm,ax=axes,**kw_cbar)
fig.show()
commonLib.saveFig(fig)
