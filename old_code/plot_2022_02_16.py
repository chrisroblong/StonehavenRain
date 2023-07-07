# plot the 2022-02-16 1km data for the UK
import matplotlib.pyplot as plt
import xarray
import stonehavenRainLib
import pathlib
import cartopy.crs as ccrs
import numpy as np
import itertools

month=2
day =16
hr_start=0 #-- hour (UTC) to start time plots
hr_end = 23 # Hour (UTC) to end time plots
year=2022

read_data = True # If true read the data...

rgn = dict(projection_x_coordinate=slice(5e4,5e5),
           projection_y_coordinate=slice(5e5,1.1e6))
rgn=None


# sites we want to plot and the colors they are.
sites=dict(edinburgh=commonLib.stonehaven_crash)
colors=dict(edinburgh='black')
#nimrodDir= pathlib.Path(f"/badc/ukmo-nimrod/data/composite/uk-{resoln}")
nimrodDir = commonLib.dataDir/'raw_data'
time_str=f"{year:04d}{month:02d}{day:02d}"
# get the data
resoln='1km'
if read_data:
    nimrod_dir=nimrodDir/f"{year:04d}"
    file_name = f'metoffice-c-band-rain-radar_uk_{time_str}_{resoln}-composite.dat.gz.tar'
    path = nimrod_dir/file_name
    rain = commonLib.extract_nimrod_day(path,region=rgn,QCmax=400.0)



## Plot the timeseries from the day for the two sites. 
time_sel=slice(f'{year}-{month:02d}-{day:02d}T{hr_start:02d}',f'{year}-{month:02d}-{day:02d}T{hr_end:02d}')

# setup for landscape. 
figts,ax = plt.subplots(nrows=1,ncols=1,clear=True,sharex='all',sharey='row',
                          num=f'UKradarEDts{year}_{month:02d}_{day:02d}',figsize=[11,7])

# plot not quite overlapping

for label,loc in sites.items():
    ts = rain.sel(method='nearest',**loc).sel(time=time_sel)
    ts.plot.step(ax=ax,color=colors[label],linewidth=2,label=label)
ax.set_title('Edinburgh Rain (mm/hr)')
ax.grid(visible=True,axis='x')



figts.suptitle(f"Rainfall rates (mm/hr) {year}-{month:02d}-{day:02d}")

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
levels=[0,1,2,4,8,16]
# Setup for portrait
fig,axes = plt.subplots(nrows=1,ncols=3,clear=True,sharex=True,sharey=True,
                        num=f'UKradar{year}_{month:02d}_{day:02d}',figsize=[11,7],
                        subplot_kw=dict(projection=proj))

for ax,time in zip(axes.flatten(),['2022-02-16T11:00','2022-02-16T12:00','2022-02-16T13:00']):
    ax.set_extent([-7,1,54,59],crs=ccrs.PlateCarree())
    cm=rain.sel(time=time).plot(ax=ax,cmap=cmap,levels=levels,transform=proj,add_colorbar=False)
    ax.set_title(f"Rain  {time}")
    commonLib.std_decorators(ax) # put std stuff on axis
    # end looping over vars
fig.colorbar(cm,ax=axes,**kw_cbar) # colorbar for the releant plots


#fig.tight_layout()
fig.show()
commonLib.saveFig(fig)    

