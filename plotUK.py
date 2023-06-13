import matplotlib.pyplot as  plt
import stonehavenRainLib
import cartopy.crs as ccrs
resoln = '5km'
year = 2020
glob_patt = '08'
dataYr = stonehavenRainLib.nimrodRootDir / f'uk-{resoln}/{year:04d}'
files = sorted(list(dataYr.glob(f'*{year:02d}{glob_patt}[0-3][0-9]*-composite.dat.gz.tar')))
f = files[11]
rain=stonehavenRainLib.extract_nimrod_day(f, QCmax=400, check_date=True)
fig,ax=plt.subplots(clear=True,num="stone",subplot_kw=dict(projection=ccrs.OSGB()))
rain.max("time").plot(robust=True,ax=ax)
stonehavenRainLib.std_decorators(ax)
fig.show()
