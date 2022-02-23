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

# sites we want to plot and the colors they are.
sites=dict(castle=edinburgh_castle,botanics=edinburgh_botanics)
colors = dict(castle='purple',botanics='brown')

time_str=f"{year:04d}{month:02d}{day:02d}"
# get the data
if read_data:
    rain2hr=dict() # 2 hour running
    rain1h=dict()
    rain15m=dict()
    for resoln in ['1km','5km']:
        nimrod_dir=pathlib.Path(f"/badc/ukmo-nimrod/data/composite/uk-{resoln}")/f"{year:04d}"
        file_name = f'metoffice-c-band-rain-radar_uk_{time_str}_{resoln}-composite.dat.gz.tar'
        path = nimrod_dir/file_name
        rain = commonLib.extract_nimrod_day(path,region=rgn,QCmax=400.0)
        rain1h[resoln+' hr'] = rain.resample(time='1h',label='right').mean() # average rain in hour in mm/hr.
        rain15m[resoln+' 15min'] = rain.resample(time='15min',label='right').mean() # average rain in 15min period in mm/hr.
        rain2hr[resoln+' 2hr_r']=rain1h[resoln+' hr'].rolling(time=2).mean() # 2 hr running avg (mm/hr)


## Plot the timeseries from the day for the two sites. 
time_sel=slice(f'{year}-{month:02d}-{day:02d}T{hr_start:02d}',f'{year}-{month:02d}-{day:02d}T{hr_end:02d}')

# setup for landscape. 
figts,axes = plt.subplots(nrows=2,ncols=3,clear=True,sharex='all',sharey='row',
                          num=f'UKradarEDts{year}_{month:02d}_{day:02d}',figsize=[11,7])
axes=axes.T
# plot not quite overlapping
for ax,(title,var) in zip(axes.flatten(),itertools.chain(rain2hr.items(),rain1h.items(),rain15m.items())):
    for label,loc in sites.items():
        ts = var.sel(method='nearest',**loc).sel(time=time_sel)
        ts.plot.step(ax=ax,color=colors[label],linewidth=2,label=label)
    ax.set_title(title)
    ax.grid(visible=True,axis='x')



figts.suptitle(f"Rainfall rates (mm/hr) {year}-{month:02d}-{day:02d}")
axes[0][0].legend()
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

# Setup for portrait
fig,axes = plt.subplots(nrows=2,ncols=3,clear=True,sharex=True,sharey=True,
                        num=f'UKradar{year}_{month:02d}_{day:02d}',figsize=[11,7],
                        subplot_kw=dict(projection=proj))
axes=axes.T 

for rain,axe,levels in zip([rain2hr,rain1h,rain15m],
                           axes[:],
                           [levels_2h_r,levels_1h,levels_15m]):
    
    for ax,(title,var) in zip(axe,rain.items()):

        ax.set_extent([-3.5,-2.5,55.75,56.25],crs=ccrs.PlateCarree())
        maxV = var.max('time')
        cm=maxV.plot(ax=ax,cmap=cmap,levels=levels,transform=proj,add_colorbar=False)
        ax.set_title(f"Max {title}")
        # add on Edinburgh (centre) & Edinburgh castle.
        commonLib.std_decorators(ax) # put std stuff on axis

    fig.colorbar(cm,ax=axe,**kw_cbar) # colorbar for the releant plots
# end looping over vars
for ax in axes.flatten(): # put the sites on
    for key in sites.keys():
        c = colors[key]
        loc = sites[key]
        ax.plot(*(loc.values()),marker='o',ms=4,color=c)


fig.suptitle(f"Max Rainfall rates (mm/hr) {year}-{month:02d}-{day:02d}")


fig.tight_layout()
fig.show()
commonLib.saveFig(fig)    

