import matplotlib.pyplot as  plt
import stonehavenRainLib
import cartopy.crs as ccrs
import xarray as xr
from matplotlib.animation import FuncAnimation
import math
import numpy as np
from functools import partial
from datetime import datetime


# set the region of the plot
st_rgn = stonehavenRainLib.stonehaven_region
rgn = []
for k,v in st_rgn.items():
	rgn.extend([v.start,v.stop])
st_time = '2020-08-12T04:15:00.000000000'


def map(data):
	fig,ax=plt.subplots(clear=True,num="stonehavendaymaxrainmap",subplot_kw=dict(projection=ccrs.OSGB()))
	ax.set_extent(rgn,crs=ccrs.OSGB())
	data.plot(robust=True,ax=ax)
	ax.plot(*stonehavenRainLib.stonehaven_crash.values(), color="red", marker="o", ms=9)
	stonehavenRainLib.std_decorators(ax)
	fig.show()

def getrainfalldata(day, month, year):
	resoln = '5km'
	glob_patt = month
	dataYr = stonehavenRainLib.nimrodRootDir / f'uk-{resoln}/{year:04d}'
	files = sorted(list(dataYr.glob(f'*{year:02d}{glob_patt}[0-3][0-9]*-composite.dat.gz.tar')))
	f = files[day]
	data = stonehavenRainLib.extract_nimrod_day(f, QCmax=400, check_date=True)
	data.attrs = {}
	return data

def mapstonehavendaymaxrain():
	rain = getrainfalldata(11, '08', 2020)
	maxrain = rain.max("time")
	map(maxrain)

def graphstonehavendayrain():
	rain = getrainfalldata(11, '08', 2020)
	fig,ax=plt.subplots(clear=True,num="stonehavedayraingraph")
	rain.sel(**stonehavenRainLib.stonehaven_crash, method="nearest").plot(ax=ax)
	fig.show()

def timechunk(time, n):
	n = n-1
	timechunk = np.array([time], dtype = 'datetime64[ns]')
	for i in range(n):
		timechunk = np.append(timechunk, timechunk[0] + (i+1)*np.timedelta64(15,'m'))
	return timechunk

def animate(i, rainy_times, rain):
	fig,ax=plt.subplots(clear=True,num="stoneanimation",subplot_kw=dict(projection=ccrs.OSGB()))
	ax.set_extent(rgn,crs=ccrs.OSGB())
	rain.sel(time=rainy_times[i]).plot(vmin=0,vmax=2.5,ax=ax)
	ax.plot(*stonehavenRainLib.stonehaven_crash.values(), color="red", marker="o", ms=9)
	stonehavenRainLib.std_decorators(ax)

def animatestonehavenmorning(rainy_times):
	fig,ax=plt.subplots(clear=True,num="stoneanimation",subplot_kw=dict(projection=ccrs.OSGB()))
	ax.set_extent(rgn,crs=ccrs.OSGB())
	rain = getrainfalldata(11, '08', 2020)
	animation = FuncAnimation(fig, partial(animate, rainy_times=rainy_times, rain=rain), frames=16, interval=500)
	plt.show()

def daily_max(day, month, year):
	daysrain=getrainfalldata(day, month, year)
	print(str(day+1) + "/" + month)
	return daysrain.max("time")

def daytodate(day):
	if day < 30:
		return day, '06'
	if day < 61:
		return day-30, '07'
	return day-61, '08'

def summer_maxdate(year):
	days = range(92)
	daily_maxima = []
	for day in days:
		date, month = daytodate(day)
		daily_maxima.append(daily_max(date, month, year))
	summer_maxima = xr.concat(daily_maxima, dim='day')
	summer_maxima['day'] = days
	return summer_maxima.idxmax('day')

def add0ifbelow10(n):
	if n > 9:
		return str(n)
	return '0' + str(n)

def dayhourly(date, month, year, rain):
	print(str(date+1) + "/" + month)
	daystart = str(year) + '-' + month + '-' + add0ifbelow10(date+1) + 'T' + '00:00:00.000000000'
	starttimes = timechunk(daystart, 96)
	hourly_data = []
	badtimes = []
	for starttime in starttimes:
		times = timechunk(starttime, 4)
		hours_data = []
		try:
			for time in times:
				hours_data.append(rain.sel(time=time))
			hours_rain = xr.concat(hours_data, dim='time')
			hours_rain['time'] = times
			hourly_data.append(hours_rain.sum('time')/4)
		except KeyError:
			print("Missing data at " + np.datetime_as_string(starttime, unit = 'ns'))
			badtimes.append(starttime)
			continue
	starttimes = list(filter(lambda x: x not in badtimes, starttimes))
	hourly_rain = xr.concat(hourly_data, dim = 'time')
	hourly_rain['time'] = starttimes
	return hourly_rain

def maxhourly(year):
	days = range(0, 92)
	daily_maxhourly = []
	daily_maxhourlytimes = []
	daysrain = getrainfalldata(0, '06', year)
	nextdaysrain = getrainfalldata(1, '06', year)
	totaldata = xr.combine_by_coords(data_objects=[daysrain, nextdaysrain])
	for day in days:
		date, month = daytodate(day)
		if day == 91:
			dayshourly = dayhourly(date, month, year, totaldata)
		else:
			dayshourly = dayhourly(date, month, year, totaldata)
		if day == 0:
			dayshourly = dayshourly.to_array()
		daily_maxhourly.append(dayshourly.max("time"))
		daily_maxhourlytimes.append(dayshourly.idxmax("time"))
		if day == 91:
			break
		daysrain = nextdaysrain
		date, month = daytodate(day + 2)
		if day < 90:
			nextdaysrain = getrainfalldata(date, month, year)
			totaldata = xr.combine_by_coords(data_objects=[daysrain, nextdaysrain]).to_array()
	maxhourly = xr.concat(daily_maxhourly, dim = 'timeno')
	maxhourlytimes = xr.concat(daily_maxhourlytimes, dim = 'timeno')
	maxhourly['timeno'] = range(92)
	maxhourlytimes['timeno'] = range(92)
	maxhourlyday = maxhourly.idxmax('timeno')
	maxhourlytime = maxhourlytimes.sel(timeno=maxhourlyday).max('variable')
	return maxhourlyday

data = maxhourly(2020)
print(data)
map(data)
