"""
Plot 3x2 figure. -- "figure 1"
Top row: Rain/topog @ 15:00 4th July; Rain/topog @ 16:00 on 4th July; Rain/Topog at 17:00 on 4th July
    d(and 5km 15 minute)
Bottom Row:  Max 1km rain for summer 2021; distribution (from broader region)
"""
import matplotlib.pyplot as plt
import stonehavenRainLib
import commonLib
import gev_r
import cartopy.crs as ccrs
import numpy as np
import xarray
from old_code import emp_dist
from matplotlib_scalebar.scalebar import ScaleBar
# std plotting stuff
scalebarprops=dict(frameon=False,length_fraction=0.4)
kw_colorbar=dict(orientation='horizontal',fraction=0.1,pad=0.1,aspect=30) # keywords for colorbars.
# first get the data.
readData = True ## needed first time code is run.
bootstrap=False # if want the bootstraping done for uncertainties in the radar data fit. If False then data will be read in from cached files
ed_rgn = stonehavenRainLib.stonehaven_region
rgn= []
for k,v in ed_rgn.items():
    rgn.extend([v.start,v.stop])

qt=0.95
rain = np.linspace(5, 250.) # rain values wanted for rp's etc
if readData:
    # get the following datasets
    # 600m topography.
    # rainfall for 4th July 2021 -- 5 minute 1km radar & 15 minute 5km radar.
    topog = xarray.load_dataset(stonehavenRainLib.dataDir / 'GEBCO_topog_bathy_scotland.nc')
    file1km= stonehavenRainLib.nimrodRootDir/'uk-1km/2021/'/\
          'metoffice-c-band-rain-radar_uk_20210704_1km-composite.dat.gz.tar'
    print("Loading radar data for 2021-07-04")
    rain1km=stonehavenRainLib.extract_nimrod_day(file1km,QCmax=400.,region=ed_rgn)
    print("Extracting radar data")

    mc_dist = xarray.load_dataset(stonehavenRainLib.dataDir/'radar_precip'/'bootstrap_reg_radar_params.nc')
    radar_fit = xarray.load_dataset(stonehavenRainLib.dataDir/'radar_precip'/'reg_radar_params.nc')
    radar_data_all= xarray.load_dataset(stonehavenRainLib.dataDir/'radar_precip'/'reg_radar_rain.nc')
    e_dist = emp_dist.empDist(radar_data_all.radar.sel(time_quant=qt))
    emp_rp = 1.0/e_dist.sf(rain)
    # get in the monthly max data and work out the rain at the Castle in July 2021 and Aug 2022
    ds = xarray.load_dataset(stonehavenRainLib.dataDir / 'radar_precip' / 'summary_1km_15Min.nc')
    castle_rain = ds.monthlyMax.sel(time=(ds.monthlyMax.time.dt.season == 'JJA')).\
        sel(time=['2020-08-31','2021-07-31'],**stonehavenRainLib.stonehaven_crash,method='nearest').load()

    print("read data")

## plot the data now.
q = [0.05, 0.95]  # quantiles for fit between
rp = 1. / gev_r.xarray_sf(rain,radar_fit.Parameters, 'Rx15min', )
mc_rp = (1. / gev_r.xarray_sf(rain,mc_dist.Parameters, 'Rx15min')).quantile(q, 'sample')
projLand = ccrs.epsg(3035)  # for land cover
projGB = ccrs.OSGB() # OS GB grid.
projRot = ccrs.RotatedPole(pole_longitude=177.5,pole_latitude=37.5) # for rotated grid (cpm)
projPC = ccrs.PlateCarree() # long/lat grid
topog_lev = [-50, -25, 0, 25, 50, 75, 100, 150, 200, 300, 400, 500]

rn_levels=np.array([5,10,20,30,40,50,100,200])
fig= plt.figure(clear=True,num='geography_dists',figsize=[10,7])
# first the rain and the topog.
# inset plot of W Europe included in first plot to show where Edinburgh is.
axes=[] # keeping all the axes we have!
for indx,time in enumerate(['2021-07-04T15:00','2021-07-04T16:00','2021-07-04T17:00']):
    ax=fig.add_subplot(2,3,indx+1,projection=projGB)
    axes.append(ax)
    ax.set_extent(rgn,crs=projGB)
    cmt=topog.elevation.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, cmap='terrain',
                         levels=topog_lev)
    ax.plot(*stonehavenRainLib.stonehaven_crash.values(), transform=projGB,
            marker='o', color='purple', ms=9, alpha=0.7)
    r=rain1km.sel(time=time)
    r=xarray.where(r > 1,r,np.nan) # remove light  rain.
    cmr=r.plot(ax=ax,transform=projGB,add_colorbar=False,cmap='Blues',levels=rn_levels)
    stonehavenRainLib.std_decorators(ax)
    ax.set_title(f"Rain at {time[-5:]} ")
    scalebar = ScaleBar(1, "m", **scalebarprops)
    ax.add_artist(scalebar)

# add on colorbars
fig.colorbar(cmr,ax=axes[2],**kw_colorbar,label='Rain (mm/hr)')
fig.colorbar(cmt,ax=axes[2],label='Topog. (m)')
## plot t/s
ax=fig.add_subplot(2,3,4)
axes.append(ax)
time_rng=slice('2021-07-04T15:00','2021-07-04T19:00')
trans=dict(botanics='RBGE',castle='Ed. Castle')
for name,site in stonehavenRainLib.sites.items():
    ts=rain1km.sel(**site,method='nearest').sel(time=time_rng)
    color = stonehavenRainLib.colors[name]
    total_rain = float(ts.resample(time='1h').mean().sum()) # total rain over the period.
    label = trans[name]+f" {total_rain:3.0f} mm"
    ts.plot.step(ax=ax,color=color,linewidth=2,label=label)

ax.set_title("Edinburgh Rainfall")
ax.set_ylabel("mm/hr")
ax.legend(fontsize='small')

ed_narrow= dict()
rr=[]
for k,v in stonehavenRainLib.stonehaven_crash.items(): # 20x20km around edinburgh
    ed_narrow[k]=slice(v-15e3,v+15e3)
    rr.extend([v-10e3,v+10e3])
# subplot for max.
ax=fig.add_subplot(2,3,5,projection=projGB)
axes.append(ax)
ax.plot(stonehavenRainLib.stonehaven_crash.values())
ax.set_extent(rr,crs=projGB)
topog.elevation.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, cmap='terrain',
                         levels=topog_lev)
rn15min =rain1km.sel(**ed_narrow).rolling(time=3).mean().max('time') #max 15 minute average.
rn15min = xarray.where(rn15min > 10,rn15min,np.nan)
cr15=rn15min.plot(ax=ax,levels=rn_levels,cmap='Blues',add_colorbar=False)
ax.set_title('15 Min Max (mm/hr)')
stonehavenRainLib.std_decorators(ax)
for name, site in stonehavenRainLib.sites.items():
    color=stonehavenRainLib.colors[name]
    ax.plot(*site.values(), transform=projGB,
            marker='o', color=color, ms=8, alpha=0.7)
scalebar = ScaleBar(1, "m", **scalebarprops)
ax.add_artist(scalebar)



ax = fig.add_subplot(2,3,6)
axes.append(ax)

## plot the radar fit and uncertainties.
#fig,ax=plt.subplots(num='test',clear=True) # for testing
radar_data = radar_data_all.sel(time_quant=qt) # want media
mc_rp_median=mc_rp.sel(time_quant=qt)
ax.fill_between(mc_rp_median.Rx15min,mc_rp_median.isel(quantile=0),y2=mc_rp_median.isel(quantile=1),color='grey')
rp.sel(time_quant=qt).plot(color='red',ax=ax,linewidth=3)
#ax.plot(rain,emp_rp,linewidth=3,color='green')
ax.set_xlabel("Rx15min (mm/hr)")
ax.set_yscale('log')
ax.set_ylabel("Return Period (years)")
ax.set_title("Regional Return Period")
for v in [10,100]:
    ax.axhline(v,linestyle='dashed')
# add on the appropriate value for 2021  & 2020
color=stonehavenRainLib.colors['castle']
value = float(radar_data.critical2021)
value2020 = float(radar_data.critical2020)
ax.axvline(value,color=color,linestyle='solid')
ax.axvline(value2020,color=color,linestyle='dashed')
# plot the actual castle rainfall for 2020 and 2021.
for yr,linestyle in zip(['2020','2021'],['dashed','solid']):
    value = castle_rain.sel(time=yr)
    ax.axvline(value,color='red',linestyle=linestyle)
ax.set_xlim(30,150.)
ax.set_ylim(5,1000)
label = commonLib.plotLabel()
for ax in axes:
    label.plot(ax)
fig.tight_layout()
fig.show()
commonLib.saveFig(fig)
##

