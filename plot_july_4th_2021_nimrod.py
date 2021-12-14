"""
Plot data summaries for 4th of July 2021 in Edinburgh
Use iris to load in a day of data then convert to xarray
"""

import matplotlib.pyplot as plt
import pathlib
import iris.cube
import iris.fileformats
import iris.quickplot
import matplotlib.colors as mcol
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray
import commonLib


def select_nimrod(cube, field, filename):
    """
    Process nimrod cube so we can merge it.
    Does by removing recursive_filter_iterations
    :param cube: cube being processed
    :param field: field
    :param filename: filename
    :return:
    """
    filter = cube.attributes.pop('recursive_filter_iterations')  # pop gets rid of this which should make merging work.
    return cube


d = pathlib.Path(r"C:\Users\stett2\data\Edinburgh_rain")
file = d / '202107041300_nimrod_ng_radar_rainrate_composite_1km_UK'
files = [str(f) for f in d.glob('20210704*UK')]
# use iris to read nimrod cubes but then convert to xarray for subsequent work.
c = iris.cube.CubeList(iris.fileformats.nimrod.load_cubes(files, callback=select_nimrod)).merge('time')[0]
# and convert to xarray as that is much easier to work with.
radar_rain = xarray.DataArray.from_iris(c)
# co-ords of data are UK National grid.
# Edinburgh castle is  324990 Easting & 672876 Northing
ed_botanics = dict(projection_x_coordinate=324990, projection_y_coordinate=672876, method='nearest')
ed_castle = dict(projection_x_coordinate=325204, projection_y_coordinate=673447, method='nearest')
rgn = dict(projection_y_coordinate=slice(500e3, 750e3), projection_x_coordinate=slice(200e3, 450e3))
ed_rgn = dict(projection_y_coordinate=slice(645e3, 695e3), projection_x_coordinate=slice(280e3, 370e3))
radar_rain_rgn = radar_rain.sel(**rgn)  # and extract the rgn we want.
radar_rain_rgn = radar_rain_rgn.resample(time='h').mean()  # hourly data
# then pullout Edinburgh...
radar_rain_ed = radar_rain_rgn.sel(**ed_rgn)

## make plots
regions = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')
nations = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_countries',
    scale='10m',
    facecolor='none')
projection = ccrs.OSGB()
times = ['13', '14', '15', '16', '17']
figure_scot, ax_scot = plt.subplots(nrows=3, ncols=2, num='Sth_Scot', clear=True, figsize=[8, 12],
                                    subplot_kw=dict(projection=projection))
ax_scot = ax_scot.flat
label_scot = commonLib.plotLabel()

figure, ax = plt.subplots(nrows=3, ncols=2, num='Edinburgh', clear=True, figsize=[8, 12],
                          subplot_kw=dict(projection=projection))
label = commonLib.plotLabel()
ax = ax.flat
norm = mcol.LogNorm()
cbar = dict(orientation='horizontal')
for a, dataArray in zip([ax[0], ax_scot[0]], [radar_rain_ed, radar_rain_rgn]):
    dataArray.max('time').plot.pcolormesh(rasterized=True, vmin=1, vmax=64, norm=norm, ax=a,cbar_kwargs=cbar)
    # Edinburgh Castle is roughly easting 325000 Northing 673000
    a.plot(325204, 673447, marker='*', ms=10, color='black', transform=ccrs.OSGB(), alpha=0.5)
    a.add_feature(regions, edgecolor='red')
    a.add_feature(cfeature.BORDERS, edgecolor='red', resolution='10m')
    a.coastlines(resolution='10m')
    a.set_title("Max hourly rain")


for a, a_scot, time in zip(ax[1:], ax_scot[1:], times):
    radar_rain.sel(time=f'2021-07-04T{time}:00', **ed_rgn).plot.pcolormesh(rasterized=True, vmin=1, vmax=64, norm=norm,
                                                                           ax=a, add_colorbar=False)

    radar_rain.sel(time=f'2021-07-04T{time}:00', **rgn).plot.pcolormesh(rasterized=True, vmin=1, vmax=64, norm=norm,
                                                                        ax=a_scot, add_colorbar=False)
    for aa in [a, a_scot]:  # common axis stuff
        aa.set_title(f'5min {time}00Z')
        a.plot(325204, 673447, marker='*', ms=10, color='black', transform=ccrs.OSGB(), alpha=0.5)
        aa.add_feature(regions, edgecolor='red')
        aa.coastlines(resolution='10m')

# add labels to all plots.

for a in ax_scot:
    label_scot.plot(a)
for a in ax:
    label.plot(a)

for fig, title in zip([figure, figure_scot], ['Edinburgh', 'Southern Scotland']):
    fig.suptitle(f"{title} rain (mm/hour) 2021-07-04")
    fig.tight_layout()
    fig.show()
    commonLib.saveFig(fig)


## plot t/s of rain at Castle and botanics.

fig,axes = plt.subplots(nrows=2,ncols=1,num='Edinburgh_ts',clear=True,figsize=[6,6],sharex='all',sharey='all')
for ax,loc,title in zip(axes.flat,[ed_castle,ed_botanics],['Edinburgh Castle','Botanic Gardens']):
    ts=radar_rain.sel(**loc).sel(time=slice('2021-07-04T14:00','2021-07-04T18:00'))/12 # convert to mm/5 min
    ts.cumsum().plot.step(ax=ax) # and plot it
    ax.set_ylabel('Total Rain (mm)')
    ax.set_title(title)
    ax.axvline('2021-07-04T16:00',color='grey',linestyle='dashed')
fig.suptitle("Cumulative rain (mm) 2021-07-04")
fig.tight_layout()
fig.show()
commonLib.saveFig(fig)

