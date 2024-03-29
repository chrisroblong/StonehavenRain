import matplotlib.pyplot as  plt
import numpy

import stonehavenRainLib
import cartopy.crs as ccrs
import xarray as xr
from matplotlib.animation import FuncAnimation
import math
import numpy as np
from functools import partial
from datetime import datetime
import gev_r
import commonLib
from scipy.special import gamma
from scipy.stats import genextreme

# set the region of the plot
st_rgn = stonehavenRainLib.stonehaven_region
rgn = []
for k,v in st_rgn.items():
    rgn.extend([v.start,v.stop])
st_time = '2020-08-12T04:15:00.000000000'



def map(data, titleofmap="", titleofplot=""):
    fig,ax=plt.subplots(clear=True,num=titleofmap,subplot_kw=dict(projection=ccrs.OSGB()))
    ax.set_extent(rgn,crs=ccrs.OSGB())
    data.plot(robust=True,ax=ax, cbar_kwargs=None)
    ax.plot(*stonehavenRainLib.stonehaven_crash.values(), color="red", marker="o", ms=9)
    plt.title(titleofplot)
    stonehavenRainLib.std_decorators(ax)
    commonLib.saveFig(fig)
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
        except:
            print("Other error at " + np.datetime_as_string(starttime, unit = 'ns'))
            badtimes.append(starttime)
            continue
    starttimes = list(filter(lambda x: x not in badtimes, starttimes))
    if starttimes == []:
        print("Missing day!")
        raise ValueError("Missing day")
    hourly_rain = xr.concat(hourly_data, dim = 'time')
    hourly_rain['time'] = starttimes
    return hourly_rain

def maxhourly(year):
    startday, endday = 0, 92
    days = range(startday, endday)
    startdate, startmonth = daytodate(startday)
    nextstartdate, nextstartmonth = daytodate(startday + 1)
    daily_maxhourly = []
    daily_maxhourlytimes = []
    baddays = []
    daysrain = getrainfalldata(startdate, startmonth, year)
    nextdaysrain = getrainfalldata(nextstartdate, nextstartmonth, year)
    totaldata = xr.combine_by_coords(data_objects=[daysrain, nextdaysrain])
    for day in days:
        date, month = daytodate(day)
        try:
            if day == endday - 1:
                dayshourly = dayhourly(date, month, year, totaldata)
            else:
                dayshourly = dayhourly(date, month, year, totaldata)
            if day == startday:
                dayshourly = dayshourly.to_array()
        except ValueError("Missing day"):
            baddays.append(day-startday)
            if day == endday - 1:
                break
            daysrain = nextdaysrain
            date, month = daytodate(day + 2)
            if day < endday - 2:
                nextdaysrain = getrainfalldata(date, month, year)
            totaldata = xr.combine_by_coords(data_objects=[daysrain, nextdaysrain]).to_array()
        daily_maxhourly.append(dayshourly.max("time"))
        daily_maxhourlytimes.append(dayshourly.idxmax("time"))
        if day == endday - 1:
            break
        daysrain = nextdaysrain
        date, month = daytodate(day + 2)
        if day < endday - 2:
            nextdaysrain = getrainfalldata(date, month, year)
            totaldata = xr.combine_by_coords(data_objects=[daysrain, nextdaysrain]).to_array()
    maxhourly = xr.concat(daily_maxhourly, dim='dayno')
    maxhourlytimes = xr.concat(daily_maxhourlytimes, dim='dayno')
    daynos = range(endday-startday)
    daynos = list(filter(lambda x: x not in baddays, daynos))
    maxhourly['dayno'] = daynos
    maxhourlytimes['dayno'] = daynos
    maxhourlyday = maxhourly.idxmax('dayno')
    maxhourlytime = maxhourlytimes.sel(dayno=maxhourlyday).max('variable')
    return maxhourlyday


def plot_return_period(radar_data_all, mc_rp, rp, crash_rain, qt=0.95):
    fig, ax = plt.subplots()
    radar_data = radar_data_all.sel(time_quant=qt)  # want media
    mc_rp_median = mc_rp.sel(time_quant=qt)
    ax.fill_between(mc_rp_median.Rx1h, mc_rp_median.isel(quantile=0), y2=mc_rp_median.isel(quantile=1), color='grey')
    rp.sel(time_quant=qt).plot(color='red', ax=ax, linewidth=3)
    # ax.plot(rain,emp_rp,linewidth=3,color='green')
    ax.set_xlabel("Rx1hr (mm/hr)")
    ax.set_yscale('log')
    ax.set_ylabel("Return Period (years)")
    ax.set_title("Regional Return Period")
    for v in [10, 100]:
        ax.axhline(v, linestyle='dashed')
    # add on the appropriate value for 2020
    color = stonehavenRainLib.colors['crash']
    value2020 = float(radar_data.critical2020)
    print(value2020)
    ax.axvline(value2020, color=color, linestyle='dashed')
    # plot the actual crash rainfall for 2020 and 2021.
    value = crash_rain
    ax.axvline(value, color='red', linestyle='solid')
    ax.set_xlim(0, 100.)
    ax.set_ylim(5, 1000)
    plt.show()
    commonLib.saveFig(fig)


def run_return_plot():
    q = [0.05, 0.95]  # quantiles for fit between
    rain = np.linspace(0, 100.) # rain values wanted for rp's etc
    radar_data_all = xr.load_dataset(stonehavenRainLib.dataDir / 'radar_fits' / 'reg_radar_rain.nc')
    mc_dist = xr.load_dataset(stonehavenRainLib.dataDir / 'radar_fits' / 'bootstrap_reg_radar_params.nc')
    radar_fit = xr.load_dataset(stonehavenRainLib.dataDir / 'radar_fits' / 'reg_radar_params.nc')
    ds = xr.load_dataset(stonehavenRainLib.dataDir / 'transfer_dir' / 'summary_5km_1h.nc')
    crash_rain = ds.monthlyMax.sel(time=(ds.monthlyMax.time.dt.season == 'JJA')).sel(**stonehavenRainLib.stonehaven_crash, method='nearest').sel(time='2020-08-31T00:00:00.000000000')
    rp = 1. / gev_r.xarray_sf(rain,radar_fit.Parameters, 'Rx1h')
    mc_rp = (1. / (gev_r.xarray_sf(rain, mc_dist.Parameters, 'Rx1h')+0.00001)).quantile(q, 'sample') # avoid div by 0
    plot_return_period(radar_data_all=radar_data_all, mc_rp=mc_rp, rp=rp, crash_rain=crash_rain)


"""
radar = xr.open_dataset(stonehavenRainLib.dataDir / "transfer_dir/summary_5km_1h.nc")
topog990m = stonehavenRainLib.read_90m_topog(region=stonehavenRainLib.stonehaven_region, resample=55)
top_fit_grid = topog990m.interp_like(radar.isel(time=0).squeeze())
correct = top_fit_grid
oversea = numpy.isnan(top_fit_grid)
toohigh = top_fit_grid > 400
gooddata = (top_fit_grid > 0) & (top_fit_grid < 400)
print(top_fit_grid)
print(correct.sum())
print(oversea.sum())
print(toohigh.sum())
print(gooddata.sum())
#map(top_fit_grid.where(gooddata, other = 0).where(~gooddata, other=1))
"""
"""
paramdict = dict()
paramdict['PI'] = (-0.173, 10.2, 3.9)
paramdict['1980s'] = (-0.173, 10.4, 4.02)
paramdict['2012-2021'] = (-0.173, 10.8, 4.34)
paramdict['PI+2K'] = (-0.173, 11.3, 4.79)


# Create a range of values for the x-axis
x = np.linspace(20, 50, 100)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), num='scaledgevdists')  # Assign a label to the figure

for k in paramdict.keys():
    pdf = genextreme.pdf(x, *paramdict[k])
    ax.plot(x, pdf, label=f'GEV: {k}')

ax.axvline(25.2, color='purple')
ax.set_xlim(20, 50)  # Set the x-axis limits
ax.set_ylim(0, 0.0275)
ax.set_xlabel('Rx1hr')
ax.set_ylabel('Probability Density')
ax.set_title('Event Intensity for each Time Period')
ax.legend()

# Save the plot using commonLib's savefig function
plt.show(fig)
#commonLib.saveFig(fig)
"""

"""
# Given parameters
location_values = [10.2, 10.4, 10.8, 11.3]
scale_values = [3.9, 4.02, 4.34, 4.79]
shape = 0.173

# Calculate mean and median for each distribution
for loc, scale in zip(location_values, scale_values):
    p = 0.5  # Probability value for median
    mean = loc + (scale / shape) * (gamma(1-shape) - 1)
    median = loc + (scale / shape) * ((-np.log(0.5)) ** (-shape) - 1)

    print(f"Location: {loc}, Scale: {scale}")
    print(f"Mean: {mean:.1f}")
    print(f"Median: {median:.1f}")
    print()
"""
"""
fig, ax = plt.subplots(figsize=(10, 6), num='eventsizehist')  # Assign a label to the figure

data = [ 13,  16,  20,  29,  34,  28,  21,  91,  21,  28,  24,  46,  14,
        18,  49,  74,  23,  16,  28,  19,  22,  28,  16,  72,  17,  27,
        14,  32,  11,  21,  11,  16,  24,  41,  29,  32,  58,  14,  15,
        14,  13,  22,  30,  46,  16,  72,  39,  31,  19,  12,  93,  15,
        30,  26,  41,  15,  18,  13,  35,  56,  26,  11,  59,  23,  24,
        14,  14, 127, 130,  17,  11,  14,  23,  20,  34,  28,  43]
data = [i*25 for i in data]
ax.hist(data, bins=20, edgecolor='black')
ax.set_xlabel('Event Size in km2')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Event Sizes')
commonLib.saveFig(fig)
"""
"""
#data = [6,2,4,4,5,4,7,3,5,5,3,3,6,5,4,2,3,6]
data = [140,112,119,155,108,133,132,94,133,127,142,124,145,141,120,141,158,162]
years = range(2005, 2023)
#fig, ax = plt.subplots(figsize=(10, 6), num='eventsyear')  # Assign a label to the figure
fig, ax = plt.subplots(figsize=(10, 6), num='cellsyear')
ax.bar(years, data)
ax.set_xlabel('Year')
ax.set_ylabel('Number of cells providing maxima')
ax.set_title('Bar chart of number of cells providing maxima per year')
commonLib.saveFig(fig)
"""
data = xr.open_dataarray(stonehavenRainLib.dataDir / 'number_of_yearsnew.nc').load()
map(data, titleofmap="yearsinradardata", titleofplot="Number of years in radar dataset")
