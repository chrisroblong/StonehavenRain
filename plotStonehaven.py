import matplotlib.pyplot as  plt
import stonehavenRainLib
import cartopy.crs as ccrs

# set the region of the plot
st_rgn = stonehavenRainLib.stonehaven_region
rgn = []
for k,v in st_rgn.items():
	rgn.extend([v.start,v.stop])

# get the rainfall data
resoln = '5km'
year = 2020
glob_patt = '08'
dataYr = stonehavenRainLib.nimrodRootDir / f'uk-{resoln}/{year:04d}'
files = sorted(list(dataYr.glob(f'*{year:02d}{glob_patt}[0-3][0-9]*-composite.dat.gz.tar')))
f = files[11]
rain=stonehavenRainLib.extract_nimrod_day(f, QCmax=400, check_date=True)

# plot the data
fig,ax=plt.subplots(clear=True,num="stone",subplot_kw=dict(projection=ccrs.OSGB()))
ax.set_extent(rgn,crs=ccrs.OSGB())
rain.max("time").plot(robust=True,ax=ax)
ax.plot(*stonehavenRainLib.stonehaven_crash.values(), color="red", marker="o", ms=9)
stonehavenRainLib.std_decorators(ax)
fig.show()

# plot the data
fig,ax=plt.subplots(clear=True,num="ts")
rain.sel(**stonehavenRainLib.stonehaven_crash, method="nearest").plot(ax=ax)
fig.show()


from matplotlib.animation import FuncAnimation

rainy_times = ['2020-08-12T04:15:00.000000000','2020-08-12T04:30:00.000000000','2020-08-12T04:45:00.000000000']
rainy_times.extend(['2020-08-12T05:00:00.000000000','2020-08-12T05:15:00.000000000','2020-08-12T05:30:00.000000000','2020-08-12T05:45:00.000000000'])
rainy_times.extend(['2020-08-12T06:00:00.000000000','2020-08-12T06:15:00.000000000','2020-08-12T06:30:00.000000000','2020-08-12T06:45:00.000000000'])
rainy_times.extend(['2020-08-12T07:00:00.000000000','2020-08-12T07:15:00.000000000','2020-08-12T07:30:00.000000000','2020-08-12T07:45:00.000000000'])
rainy_times.append('2020-08-12T08:00:00.000000000')

fig,ax=plt.subplots(clear=True,num="stoneanimation",subplot_kw=dict(projection=ccrs.OSGB()))
ax.set_extent(rgn,crs=ccrs.OSGB())

def animate(i):
	fig,ax=plt.subplots(clear=True,num="stoneanimation",subplot_kw=dict(projection=ccrs.OSGB()))
	ax.set_extent(rgn,crs=ccrs.OSGB())
	rain.sel(time=rainy_times[i]).plot(vmin=0,vmax=4,ax=ax)
	ax.plot(*stonehavenRainLib.stonehaven_crash.values(), color="red", marker="o", ms=9)
	stonehavenRainLib.std_decorators(ax)
	rainfigs.append(fig)
	rainaxs.append(ax)

animation = FuncAnimation(fig, animate, frames=16, interval=500)
plt.show()
