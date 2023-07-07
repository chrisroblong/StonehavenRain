"""
Plot radar and ERA-5 MSLP for 4th of July 2021 in Southern Scotland/Northern England.
Will do for 15:00 & 16:00 Z

Needs matplotlib >=3.5.1(well 3.4 has a bug with alpha as an array which 3.5.1 does not have)

"""

import matplotlib.pyplot as plt
import stonehavenRainLib
import pathlib
import iris.cube
import iris.fileformats
import iris.quickplot
import matplotlib.colors as mcol
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray
import commonLib
import numpy as np
stonehaven_region = dict()
for k,v in stonehavenRainLib.stonehaven_crash.items(): # 100km around edinburgh
    stonehaven_region[k]=slice(v-100e3,v+100e3)

read_data = False
if read_data:
    file= stonehavenRainLib.nimrodRootDir/'uk-1km/2021/metoffice-c-band-rain-radar_uk_20210704_1km-composite.dat.gz.tar'
    rainAll=stonehavenRainLib.extract_nimrod_day(file,QCmax=400.)
    #era_5 = xarray.open_dataset(stonehavenRainLib.dataDir/'UK_ERA5_data_2021_07_03to05.nc')
    time_sel = ['2021-07-04T14','2021-07-04T15','2021-07-04T16']
    #msl = era_5.msl.sel(time=time_sel,longitude=slice(-15,0),latitude=slice(59,53))/100.
    #era_5_rain = era_5.tp.sel(time=time_sel, longitude=slice(-15,0),latitude=slice(57,53))*1000
    rain = rainAll.sel(time=time_sel,**stonehaven_region)
    # get bathymetry/topography data
    topog = xarray.load_dataset(stonehavenRainLib.dataDir/'GEBCO_topog_bathy_scotland.nc')


## make plots

projection = ccrs.OSGB()
cmap='Blues'
figure_scot, ax_scot = plt.subplots(nrows=1, ncols=3, num='Sth_Scot', clear=True, figsize=[11, 5],
                                    subplot_kw=dict(projection=projection))

label_scot = commonLib.plotLabel()
rn_levels=[1,2,5,10,20,50,100]
slp_levels=np.linspace(1003,1006,7)
label = commonLib.plotLabel()
rgn = [stonehaven_region['projection_x_coordinate'].start,stonehaven_region['projection_x_coordinate'].stop,
              stonehaven_region['projection_y_coordinate'].start,stonehaven_region['projection_y_coordinate'].stop]
for indx,ax in enumerate(ax_scot.flatten()):
    ax.set_extent(rgn,crs=projection)

    topog.elevation.plot(ax=ax,transform=ccrs.PlateCarree(),add_colorbar=False,cmap='YlOrBr',levels=[100,200,300,400,500,600])

    #
    r = rain.isel(time=indx)
    alpha = np.where(r > 1, 1.0, 0.0) # this makes the plotting very slow.
    #alpha=0.7
    cm = r.plot.pcolormesh(ax=ax, levels=rn_levels, cmap=cmap,add_colorbar=False,alpha=alpha)
    #cm = ax.pcolormesh(r.projection_x_coordinate,r.projection_y_coordinate,r.values,alpha=alpha)
    #cmE=era_5_rain.isel(time=indx).plot(ax=ax,levels=rn_levels,cmap=cmap,transform=ccrs.PlateCarree(),alpha=0.2,add_colorbar=False)

    label.plot(ax)


    #c=msl.isel(time=indx).plot.contour(ax=ax,levels=slp_levels,transform=ccrs.PlateCarree(),colors='grey',linestyles=['solid','dashed'])
    #ax.clabel(c)
    stonehavenRainLib.std_decorators(ax)
    ax.plot(*list(stonehavenRainLib.edinburgh_botanics.values()), marker='o', ms=6, color=stonehavenRainLib.colors['botanics'],
            transform=projection,alpha=0.7)

figure_scot.colorbar(cm,orientation='horizontal',ax=ax_scot,fraction=0.08,pad=0.05,aspect=40)
#figure_scot.tight_layout()
figure_scot.show()
commonLib.saveFig(figure_scot)
