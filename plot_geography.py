"""
Plot the topography and land cover features for the region around edinburgh
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray
import rioxarray
import cartopy.crs as ccrs
import edinburghRainLib
import commonLib
import GSHHS_WDBII

gshhs = GSHHS_WDBII.GSHHS_WDBII()
coastline = gshhs.coastlines(scale='full')

topog =xarray.load_dataset(edinburghRainLib.dataDir/'GEBCO_topog_bathy_scotland.nc')
land=rioxarray.open_rasterio(edinburghRainLib.dataDir/r"u2018_clc2018_v2020_20u1_raster100m\DATA\U2018_CLC2018_V2020_20u1.tif",
                             chunks='auto',masked=True)# get in the Corraine Land Cover dataset
land = land.sel(x=slice(3.4e6,3.6e6),y=slice(3.8e6,3.6e6))
land = land.where(land >0)
#urb=xarray.where(land.isin([1,2,3,4,5,6]),1,np.nan).load()#.coarsen(x=10,y=10).mean()
urb=xarray.where(land.isin([1,2,3,4,5,6]),1,np.nan).load()# cts Urban Fabric, discont urban fabric, ports & airports. Excluding Industrial & Commerical units as pulls out windfarms...
proj = ccrs.epsg(3035) # for land cover
projectionOS = ccrs.OSGB() # for GB grid
rgn_far=[]
rgn_close=[]
for r in edinburghRainLib.edinburgh_castle.values():
    rgn_far.extend([r-50e3,r+50e3]) # within 50 km
    rgn_close.extend([r - 7.5e3, r + 7.5e3]) # within 7.5km

topog_lev=[-50,-25,0,25,50,75,100,150,200,300,400,500]
fig,axes = plt.subplots(nrows=1,ncols=2,num='top_urb',clear=True,subplot_kw=dict(projection=projectionOS),figsize=[8,5])
dl=dict(bottom='x')
for ax,rgn,title in zip(axes.flatten(),[rgn_far,rgn_close],['SE Scotland','City of Edinburgh']):
    ax.set_extent(rgn,crs=projectionOS)
    cm=topog.elevation.plot(robust=True,ax=ax,transform=ccrs.PlateCarree(),levels=topog_lev,cmap='terrain',add_colorbar=False)
    urb.plot(ax=ax,add_colorbar=False,transform=proj,levels=[0.99,1.01],colors='black',alpha=0.5)
    edinburghRainLib.std_decorators(ax)
    ax.plot(*edinburghRainLib.edinburgh_castle.values(),transform=projectionOS,marker='o',color='purple',ms=6,alpha=0.7)
    ax.set_title(title)
    if len(dl) == 1:
        dl.update(left='y')
    else:
        del(dl['left'])
        dl.update(right='y')
    ax.gridlines(draw_labels=dl,dms=True)
fig.colorbar(cm,ax=axes,orientation='horizontal',label='Height/Baythmetry from ASL (m)',fraction=0.1,pad=0.1,aspect=30,ticks=topog_lev,spacing='proportional')
fig.show()
commonLib.saveFig(fig)
