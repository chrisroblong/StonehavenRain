"""
Compute seasonal-mean CET, edinburgh region & CPM average  from
  monthly-mean temperatures and precipitation. (No CET for precip!)

"""
import stonehavenRainLib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray

wts = dict(
    Rothampsted=1.0 / 3,
    Malvern=1.0 / 3.,
    Squires_Gate=1.0 / 6,
    Ringway=1.0 / 6
)


'''
plot not working
projRot = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
proj = ccrs.PlateCarree()

fig = plt.figure(num='test_cet', clear=True)
ax = fig.add_subplot(111, projection=proj)
ax.set_extent([-5, 2, 50, 57], crs=proj)
# plot in rotated coords
for key, coords in stonehavenRainLib.rotated_coords.items():
    ax.plot(coords[0], coords[1], marker='x', color='red', transform=projRot)

ax.coastlines()
fig.show()
'''

# now to read the ensemble data.

st = stonehavenRainLib.rotated_coords['Stonehaven']
stonehaven_region = dict(grid_longitude=slice(st[0] - 0.5, st[0] + 0.5),
                         grid_latitude=slice(st[1] - 0.5, st[1] + 0.5))

chunks = dict(grid_longitude=10, grid_latitude=10)
for var in ['tas', 'pr']:
    cpm_list = []
    st_list = []
    cet_list = []
    for p in stonehavenRainLib.cpmDir.glob('[0-9][0-9]'):
        pth = p / var / 'mon/latest'
        pth_rain = p / 'pr/mon/latest'
        ncfiles = []
        for year in range(1980, 2041):
            yearstr = str(year)
            ncfiles = ncfiles + list(pth.glob('*'+yearstr+'11.nc'))
        for year in range(2060, 2081):
            yearstr = str(year)
            ncfiles = ncfiles + list(pth.glob('*'+yearstr+'11.nc'))
        print("loading" + str(len(ncfiles)) + "files")
        da = xarray.open_mfdataset(ncfiles,
                                   chunks=chunks, parallel=True,
                                   concat_dim='time', combine='nested',
                                   data_vars='minimal', coords='minimal',
                                   compat='override')[var]
        # not too bad performance! The extra args come from the xarray doc.
        cet = 0.0
        for key, wt in wts.items():
            coords = stonehavenRainLib.rotated_coords[key]
            ts = da.sel(method='nearest', tolerance=0.1,
                        grid_longitude=coords[0],
                        grid_latitude=coords[1]).load()
            cet += (ts * wt)

        cet_list.append(cet)
        st_ts = da.sel(**stonehaven_region).mean(stonehavenRainLib.cpm_horizontal_coords).load()
        st_list.append(st_ts)
        cpm_ts = da.mean(stonehavenRainLib.cpm_horizontal_coords).load()
        cpm_list.append(cpm_ts)
        print(f"Done with {p} for {var}")  # end loop over ensemble members

    cet = xarray.concat(cet_list, dim='ensemble_member')
    st = xarray.concat(st_list, dim='ensemble_member')
    cpm = xarray.concat(cpm_list, dim='ensemble_member')
    cet.to_netcdf(f'cet_{var}.nc')
    st.to_netcdf(f'st_reg_{var}.nc')
    cpm.to_netcdf(f'cpm_reg_{var}.nc')
    print(f"Done with {var}")
    # end loop over variables
