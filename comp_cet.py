"""
Compute seasonal-mean CET from monthly-mean temperatures

"""
import edinburghRainLib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray

wts = dict(
    Rothampsted=1.0/3,
    Malvern=1.0/3.,
    Squires_Gate=1.0/6,
    Ringway=1.0/6
)




projRot=ccrs.RotatedPole(pole_longitude=177.5,pole_latitude=37.5)
proj=ccrs.PlateCarree()

fig=plt.figure(num='test_cet',clear=True)
ax=fig.add_subplot(111,projection=proj)
ax.set_extent([-5,2, 50,57],crs=proj)
# plot in rotated coords
for key,coords in edinburghRainLib.rotated_coords.items():
    ax.plot(coords[0],coords[1],marker='x',color='red',transform=projRot)

ax.coastlines()
fig.show()

## now to read the ensemble data.

ed=edinburghRainLib.rotated_coords['Edinburgh']
edinburgh_region = dict(grid_longitude=slice(ed[0]-0.25,ed[0]+0.25),
                        grid_latitude=slice(ed[1]-0.25,ed[1]+0.25))
cet_list=[]
ed_list=[]
for p in edinburghRainLib.cpmDir.glob('[0-9][0-9]'):
    pth=p/'tas/mon/latest'
    ncfiles=list(pth.glob('*.nc'))
    ds=xarray.open_mfdataset(ncfiles,chunks=dict(grid_longitude=10,grid_latitude=10),parallel=True,concat_dim='time',combine='nested',
                             data_vars='minimal',coords='minimal',compat='override') # not too bad! The extra args come from the xarray doc.
    cet=0.0
    for key,wt in wts.items():
        coords=edinburghRainLib.rotated_coords[key]
        ts = ds.tas.sel(method='nearest',tolerance=0.1,grid_longitude=coords[0],
                    grid_latitude=coords[1]).load()
        cet += (ts*wt)
    
    cet_list.append(cet)
    ed_ts=ds.tas.sel(**edinburgh_region).mean(['grid_longitude','grid_latitude'])
    ed_list.append(ed_ts)
    print(f"Done with {p}")

cet=xarray.concat(cet_list,dim='ensemble_member')
ed=xarray.concat(ed_list,dim='ensemble_member')
cet.to_netcdf('cet_cpm.nc')
ed.to_netcdf('ed_reg_ts.nc')
    
    
                             
                    
    
