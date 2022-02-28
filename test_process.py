"""
Test processing code
"""
import pathlib
import commonLib
import xarray
import numpy as np

get_data = True  # set True to read in data; False to skip.

def maxValue(da,maxVar,dim='time'):
    """
    Return value from da where maxVar is maximum wrt dim.
    :param da: dataArray where values come from
    :param maxVar: dataArray where index max is found
    :param dim: dim over which to work. Default is 'time'
    :return: dataArray
    """

    bad = da.isnull().all('time', keep_attrs=True)  # find out where *all* data null
    indx = maxVar.argmax(dim='time', skipna=False, keep_attrs=True)  # index of maxes
    result = da.isel(time=indx)
    result = result.where(~bad)  # mask data where ALL is missing
    return result

def time_of_max(da,dim='time'):
    """

    Work out time of maxes using argmax to get index and then select the times.

    """
    bad = da.isnull().all(dim, keep_attrs=True)  # find out where *all* data null

    indx = da.argmax(dim=dim, skipna=False, keep_attrs=True)  # index of maxes
    result = da.time.isel({dim:indx})

    result = result.where(~bad)  # mask data where ALL is missing
    return result


def process(dataArray, resample='1D', total=True):
    """
    process a chunk of data from the hourly processed data
    Compute  dailyMax, dailyMaxTime and dailyMean and return them in a dataset
    :param dataArray -- input data array to be resampled and processes to daily
    :param total (default True) If True  not compute the daily total.

    """
    name_keys = {'1D': 'daily'}
    name = name_keys.get(resample, resample)

    resamp = dataArray.resample(time=resample, label='left')
    # set up the resample
    mx = resamp.max(keep_attrs=True).rename(name + 'Max')
    if total:
        tot = resamp.sum(keep_attrs=True).rename(name + 'Total')
        tot.attrs['units'] = 'mm/day'
    mxTime = resamp.map(time_of_max).rename(name + 'MaxTime')
    ds = xarray.merge([tot, mx, mxTime])

    return ds




def time_process(DS, varPrefix='daily',summaryPrefix=''):
    """
    Process a dataset of (daily) data
    :param DS -- Dataset to process
    :param outFile (default None). Name of file for summary data to output. If None  nothing will be written.
            All times in attributes or "payload" will be converted used commonLib.convert_time
    :param varPrefix (default 'daily') -- variable prefix on DataArrays in datasets)
    :param summaryPrefix (default '') -- prefix to be added to output maxes etc
    """

    mx = DS[varPrefix+'Max'].max('time', keep_attrs=True).rename(f'{summaryPrefix}Max')  # max of maxes
    mx_idx = DS[varPrefix+'Max'].fillna(0.0).argmax('time', skipna=True)  # index  of max
    mx_time = DS[varPrefix+'MaxTime'].isel(time=mx_idx).drop_vars('time').rename(f'{summaryPrefix}MaxTime')
    time_min = DS[varPrefix + 'Max'].time.min().values
    time_max = DS[varPrefix+'Max'].time.max().values
    mn = DS[varPrefix+'Total'].mean('time',keep_attrs=True)
    # actual time. -- dropping time as has nothing and will fix later

    ds = xarray.merge([mn,mx, mx_time]).expand_dims(time=[time_min])

    ds.attrs['max_time'] = time_max

    return ds

def write_data(ds,outFile,sumamryPrefix=''):
    """
    Write data to netcdf file -- need to fix times!
      So only converts maxTime
    :param ds -- dataset to be written
    :param outFile: file to write to.
    :return: converted dataset.
    """
    ds2 = ds.copy() # as modifying things...
    ds2.attrs['max_time'] = commonLib.time_convert(ds2.attrs['max_time'])
    var = sumamryPrefix+"MaxTime"
    ds2[var] = commonLib.time_convert(ds2[var])
    ds2.to_netcdf(outFile)
    return ds

data2021 = commonLib.dataDir / 'raw_data/2021'
# initialise the list...

files = sorted(list(data2021.glob('*5km-composite.dat.gz.tar')))
dailyData = []
dailyData2hr = []
last = None


def two_hr_mean(ds):
    """
    Compute two hour rolling mean.
    """
    return ds.rolling(time=2, min_periods=2).mean()


last_month = None
for f in files:
    rain = commonLib.extract_nimrod_day(f, QCmax=400).resample(time='1h').mean(
        keep_attrs=True)  # mean rain/hour units mm/hr
    if (last_month is None) or \
            (rain.time.dt.month.values != last_month):  # starting a new month so clean up, save data and start again.
        # clean up.

        # start again stuff
        last_month = rain.time.dt.month.values
        print(f"Starting month {last_month} ")
        dailyData = []
        dailyData2hr = []
    print(f, rain.time.values[0])
    dailyData.append(process(rain))
    if last is not None:
        if int((rain.time[0] - last.time[0]) / np.timedelta64(1, 'h')) == 1:
            rain2hr = two_hr_mean(last.combine_first(rain)).isel(time=slice(1, None))

    else:  # nothing to merge so just use the 24 hours we have
        rain2hr = two_hr_mean(rain)

    dailyData2hr.append(process(rain2hr))
    last = rain.isel(time=[-1])  # get last hour.

    dailyDataDS = xarray.concat(dailyData, dim='time')
    dailyData2hrDS = xarray.concat(dailyData2hr, dim='time')
