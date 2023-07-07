"""
Compute seasonal-mean CET, edinburgh region & CPM average  from
  monthly-mean temperatures and precipitation. (No CET for precip!)

"""
import stonehavenRainLib
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
for key,coords in stonehavenRainLib.rotated_coords.items():
    ax.plot(coords[0],coords[1],marker='x',color='red',transform=projRot)

ax.coastlines()
fig.show()

## now to read the ensemble data.

ed=stonehavenRainLib.rotated_coords['Edinburgh']
stonehaven_region = dict(grid_longitude=slice(ed[0]-0.5,ed[0]+0.5),
                        grid_latitude=slice(ed[1]-0.5,ed[1]+0.5))


chunks=dict(grid_longitude=10,grid_latitude=10)
for var in ['tas','pr']:
    cpm_list=[]
    ed_list=[]
    cet_list=[]
    for p in stonehavenRainLib.cpmDir.glob('[0-9][0-9]'):
        pth=p/var/'mon/latest'
        pth_rain = p/'pr/mon/latest'
        ncfiles=list(pth.glob('*.nc'))
        da=xarray.open_mfdataset(ncfiles,
                                 chunks=chunks,parallel=True,
                                 concat_dim='time',combine='nested',
                                 data_vars='minimal',coords='minimal',
                                 compat='override')[var]
        # not too bad performance! The extra args come from the xarray doc.
        cet=0.0
        for key,wt in wts.items():
            coords=stonehavenRainLib.rotated_coords[key]
            ts = da.sel(method='nearest',tolerance=0.1,
                             grid_longitude=coords[0],
                             grid_latitude=coords[1]).load()
            cet += (ts*wt)
        
        
        cet_list.append(cet)
        ed_ts=da.sel(**stonehaven_region).mean(stonehavenRainLib.cpm_horizontal_coords).load()
        ed_list.append(ed_ts)
        cpm_ts=da.mean(stonehavenRainLib.cpm_horizontal_coords).load()
        cpm_list.append(cpm_ts)
        print(f"Done with {p} for {var}") # end loop over ensemble members
        




    cet=xarray.concat(cet_list,dim='ensemble_member')
    ed=xarray.concat(ed_list,dim='ensemble_member')
    cpm=xarray.concat(cpm_list,dim='ensemble_member')
    cet.to_netcdf(f'cet_{var}.nc')
    ed.to_netcdf(f'ed_reg_{var}.nc')
    cpm.to_netcdf(f'cpm_reg_{var}.nc')
    print(f"Done with {var}")
    # end loop over variables
    
    
                             
                    
    
