# plot the july 4th 2021 extremes near Edinburgh from 1km and 5km data
import matplotlib.pyplot as plt
import xarray
import commonLib
import pathlib
import cartopy.crs as ccrs
import numpy as np
import itertools

month=7
day =4
hr_start=14 #-- hour (UTC) to start time plots
hr_end = 20 # Hour (UTC) to end time plots
year=2021

read_data = False # If true read the data...

rgn = dict(projection_x_coordinate=slice(5e4,5e5),
           projection_y_coordinate=slice(5e5,1.1e6))


edinburgh = dict(projection_x_coordinate=325847,
           projection_y_coordinate=674007)

edinburgh_castle = dict(projection_x_coordinate=325166,
           projection_y_coordinate=673477)


edinburgh_botanics = dict(projection_x_coordinate=324836,
                          projection_y_coordinate=675483)

edinburgh_KB = dict(projection_x_coordinate=326495,
                          projection_y_coordinate=670628)

# sites we want to plot and the colors they are.


time_str=f"{year:04d}{month:02d}{day:02d}"
# get the data
if read_data:
    rain2hr=dict() # 2 hour running
    rain1h=dict()
    rain15m=dict()
    for resoln in ['1km','5km']:
        nimrod_dir=commonLib.nimrodRootDir/f"uk-{resoln}/{year:04d}"
        file_name = f'metoffice-c-band-rain-radar_uk_{time_str}_{resoln}-composite.dat.gz.tar'
        path = nimrod_dir/file_name
        rain = commonLib.extract_nimrod_day(path,region=rgn,QCmax=400.0)
        rain1h[resoln+' hr'] = rain.resample(time='1h',label='right').mean() # average rain in hour in mm/hr.
        rain15m[resoln+' 15min'] = rain.resample(time='15min',label='right').mean() # average rain in 15min period in mm/hr.
        rain2hr[resoln+' 2hr_r']=rain1h[resoln+' hr'].rolling(time=2).mean() # 2 hr running avg (mm/hr)


## Plot the timeseries from the day for the two sites. 
time_sel=slice(f'{year}-{month:02d}-{day:02d}T{hr_start:02d}',f'{year}-{month:02d}-{day:02d}T{hr_end:02d}')


sites = commonLib.sites.copy() # remove KB
figts,axes = plt.subplots(nrows=1,ncols=2,clear=True,sharex='all',sharey='row',
                          num=f'UKradarEDts{year}_{month:02d}_{day:02d}',figsize=[11,7])

for ax,key,var in zip(axes.flatten(),['1km 15min','5km hr'],[rain15m,rain1h]):
    for label,loc in sites.items():
        scale =1
        if '15min' in key:
            scale=4
        ts = var[key].sel(method='nearest',**loc).sel(time=time_sel)/scale
        ts.plot.step(ax=ax,color=commonLib.colors[label],linewidth=4,label=f"{label} {int(ts.sum()):3d} mm")
    ax.set_title(key)
    ax.set_ylabel("Rain (mm/period)")
    ax.grid(visible=True,axis='x')
    ax.legend()



figts.suptitle(f"{year}-{month:02d}-{day:02d}")

figts.tight_layout()
figts.show()
commonLib.saveFig(figts)
#breakpoint()
## plot radar data as map
proj=ccrs.OSGB()
levels_1h=np.array([2,5,10,20,30,40,50])
levels_2h_r=levels_1h/2
levels_15m=levels_1h*2

kw_cbar=dict(orientation='horizontal',fraction=0.05,pad=0.1)
cmap='Blues'


fig, axes = plt.subplots(nrows=1, ncols=2, clear=True, sharex=True, sharey=True,
                         num=f'UKradar{year}_{month:02d}_{day:02d}', figsize=[11, 7],
                         subplot_kw=dict(projection=proj))

ext = [commonLib.edinburgh_region['projection_x_coordinate'].start,commonLib.edinburgh_region['projection_x_coordinate'].stop,
       commonLib.edinburgh_region['projection_y_coordinate'].start,commonLib.edinburgh_region['projection_y_coordinate'].stop]

for ax,key,var,levels in zip(axes.flatten(),['1km 15min','5km hr'],[rain15m,rain1h],
                                [levels_15m,levels_1h]):

    ax.set_extent(ext, crs=proj)

    maxV = var[key].max('time')
    cm = maxV.plot(ax=ax, cmap=cmap, levels=levels, transform=proj, add_colorbar=True,cbar_kwargs=kw_cbar)
    ax.set_title(f"Max {key}")

    commonLib.std_decorators(ax)  # put std stuff on axis
# add on botanics & castle
# end looping over vars
for ax in axes.flatten():  # put the sites on
    for key in sites.keys():
        c = commonLib.colors[key]
        loc = commonLib.sites[key]
        ax.plot(*(loc.values()), marker='o', ms=7, color=c)

fig.suptitle(f"Max Rainfall rates (mm/hr) {year}-{month:02d}-{day:02d}")

fig.tight_layout()
fig.show()
commonLib.saveFig(fig)

