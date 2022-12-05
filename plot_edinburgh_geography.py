"""
Plot Geography for Edinburgh extreme rainfall.
Plots As follows:
SE Scotland -- using 600m DEM data & showing counties; Edinburgh using 90m DEM data
Plan of castle. + scalebar
"""
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import rioxarray
import edinburghRainLib
import commonLib
import cartopy.crs as ccrs
import numpy as np
import xarray
import cartopy.mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ed_rgn = edinburghRainLib.edinburgh_region
rgn= []
for k,v in ed_rgn.items():
    rgn.extend([v.start,v.stop])

ed_detail = []
ed_detail_sel = dict()
for k, v in edinburghRainLib.edinburgh_castle.items():
    ed_detail.extend([v - 1e3, v + 1e3])
    ed_detail_sel[k] = slice(v - 1e3, v + 1e3)

labels={'Edinburgh':[330e3,677e3],
        'Firth of Forth':[329500 , 688e3],
        'Fife': [310999 , 690000],
        'Scotish Borders':[315000,630000.],
        'F':[286000,676000.], # Falkirk
        "WL":[302e3,674e3], # West Lothian
        "ML":[331e3,659e3], # Mid Lothian
        "EL": [350e3,667e3], # East Lothian
        "P&K":[285e3,713e3], # Perth & Kinross
        "C":[292e3,695e3],# Clackmannnanshire
        "NL":[277e3,662e3],# North Lanarkshire
        "S":[276e3,697e3]
        }
#castle_img = plt.imread(edinburghRainLib.dataDir/'images'/'Castle_from_Princes_Street,_Edinburgh.jpg') # from https://commons.wikimedia.org/wiki/File:Castle_from_Princes_Street,_Edinburgh.JPG

castle_img = plt.imread(edinburghRainLib.dataDir/'images'/'Edinburgh_castle_jessica.jpg') # from Jessica.
castle_plan = plt.imread(edinburghRainLib.dataDir/'images'/'Edinburgh_Castle_Plan.png')
topog_lev = [-50, -25, 0, 25, 50, 75, 100, 150, 200, 300, 400, 500,750]
kw_colorbar=dict(orientation='vertical',fraction=0.2,pad=0.05,aspect=30,label='Height (m)',ticks=topog_lev) # keywords for colorbars.
textprops = dict(backgroundcolor='grey',alpha=0.5,bbox=dict(pad=0)) # text props for plotting
scalebarprops=dict(frameon=False)
topog = xarray.load_dataset(edinburghRainLib.dataDir / 'GEBCO_topog_bathy_scotland.nc')
land =rioxarray.open_rasterio(
        edinburghRainLib.dataDir / r"u2018_clc2018_v2020_20u1_raster100m\DATA\U2018_CLC2018_V2020_20u1.tif",
        chunks='auto', masked=True)  # get in the Corraine Land Cover dataset
land = land.reindex(y=land.y[::-1]).sel(x=slice(3.4e6, 3.6e6), y=slice(3.6e6, 3.8e6))
land = land.where(land > 0).load()
# urb=xarray.where(land.isin([1,2,3,4,5,6]),1,np.nan).load()#.coarsen(x=10,y=10).mean()
urb = xarray.where(land.isin([1, 2, 3, 4, 5, 6]), 1,
                   np.nan)  # cts Urban Fabric, discont urban fabric, ports & airports. Excluding Industrial & Commerical units as pulls out windfarms...
noUrb = xarray.where((land > 6) & (land < 44),1, np.nan)
topog90m = edinburghRainLib.read_90m_topog(region=ed_detail_sel)
projLand = ccrs.epsg(3035)  # for land cover
projGB = ccrs.OSGB() # OS GB grid.
projRot = ccrs.RotatedPole(pole_longitude=177.5,pole_latitude=37.5) # for rotated grid (cpm)
projPC = ccrs.PlateCarree() # long/lat grid

fig = plt.figure('edinburgh_geography',figsize=[7,6],clear=True)
gs = fig.add_gridspec(2, 2,height_ratios=[1,1.3],width_ratios=[1,1.3])
ax_seScot = fig.add_subplot(gs[1,:], projection=projGB)
ax_seScot.set_extent(rgn, crs=projGB)
cmt = topog.elevation.plot(ax=ax_seScot, transform=ccrs.PlateCarree(), cmap='terrain',
                           levels=topog_lev,add_colorbar=True,cbar_kwargs=kw_colorbar)
scalebar = ScaleBar(1,"m",**scalebarprops)
ax_seScot.add_artist(scalebar)
# if time == '2021-07-04T15:00':
urb.plot(ax=ax_seScot, add_colorbar=False, transform=projLand, levels=[0.99, 1.01], colors='black',alpha=1)
ax_seScot.plot(*edinburghRainLib.edinburgh_castle.values(), transform=projGB,
               marker='o', color='purple', ms=9, alpha=0.7)
edinburghRainLib.std_decorators(ax_seScot,radarNames=True)
ax_seScot.set_title(f"SE Scotland")
# put labels on
for name,coords in labels.items():
    ax_seScot.annotate(name, coords, transform=cartopy.crs.OSGB(approx=True),weight='bold')#,**textprops)
# add an inset plot of the BI
axBI = inset_axes(ax_seScot, width="39%", height="33%", loc="lower left",
                  axes_class=cartopy.mpl.geoaxes.GeoAxes, borderpad=0.,
                  axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
axBI.set_extent((-11, 2, 50, 61))
axBI.tick_params(labelleft=False, labelbottom=False)
axBI.coastlines()
axBI.plot(*edinburghRainLib.edinburgh_castle.values(),
          marker='o',color=edinburghRainLib.colors['castle'],ms=5,transform=projGB)




axes=[]
for g,img,title in zip([gs[0,0],gs[0,1]],[castle_img,castle_plan],['Edinburgh Castle','Castle Plan']):
    ax=fig.add_subplot(g)
    axes.append(ax)
    ax.imshow(img)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_title(title)
axes.append(ax_seScot)

label = commonLib.plotLabel()
for ax in axes:
    label.plot(ax)
fig.show()
##
commonLib.saveFig(fig)