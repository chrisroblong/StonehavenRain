#!/bin/env python

"""

Process CPM data on JASMIN. Generate seasonal-totals, seasonal maxes
and time of seasonal extreme for data.


"""

import pathlib
import edinburghRainLib
import xarray
import numpy as np
import pandas as pd
import argparse
import resource # so we can see memory usage
import warnings

import pathlib
import edinburghRainLib
import xarray
import numpy as np
import pandas as pd
import argparse
import resource # so we can see memory usage every month.
import warnings
import glob
import cftime

def time_of_max(da, dim='time'):
    """

    Work out time of maxes using argmax to get index and then select the times.

    """
    da.load() # load the data -- hopefully speeds things up without overlaoding mem
    bad = da.isnull().all(dim, keep_attrs=True)  # find out where *all* data null
    if bad.all():
        print("All missing")
        #return da.isel({dim:0}) # return nan

    indx = da.argmax(dim=dim, skipna=False, keep_attrs=True)  # index of maxes
    indx = indx.load() # if indx is a dask array need to load it.
    result = da[dim].isel({dim: indx})

    result = result.where(~bad)  # mask data where ALL is missing
    return result


def process(dataArray, resample='seasonal', resampName=None, total=True):
    """
    process a chunk of data from the hourly processed data
    Compute  dailyMax, dailyMaxTime and dailyMean and return them in a dataset
    :param dataArray -- input data array to be resampled and processes to daily
    :param total (default True) If True   compute the total over the resample prd
    :param resampName -- name to give the output values.

    """
    name_keys = {'daily':'1D', 'monthly':'1M', 'seasonal':"QS-DEC"}
    resample_str = name_keys.get(resample, resample)
    if resampName is None:
        resampName=resample
        
    resamp = dataArray.resample(time=resample_str, label='left',skipna=True)
    # set up the resample
    print("Computing the mx")
    mx = resamp.max(keep_attrs=True).rename(resampName + 'Max').load()
    if total: # For CPM data is mm/hr
        print("Computing the total")
        tot = resamp.sum(keep_attrs=True).rename(resampName + 'Total').load()
        tot.attrs['units'] = 'mm'

    print("Computing the mx time")
    mxTime = resamp.map(time_of_max).rename(resampName + 'MaxTime').load()
    merge=[mx,mxTime]
    if total:
        merge.append(tot)
    ds = xarray.merge(merge).dropna('time',how='all')
    return ds

    

def outputName(rootName,rolling=None,outDir=None):
    """
    Work out output file name
    :param rootName -- rootname of file
    :param rolling -- what rolling period was used
    :param outDir -- file path for outDir
    """
    if outDir is None:
        outDir=pathlib.Path.cwd()
    file = rootName
    if rolling is not None:
        file +=f"_{rolling:d}"
    file += ".nc"
    name=outDir/file
    return name

def time_convert(DataArray,ref_yr=1970,set_attrs=True):
    """
    convert times to hours (etc) since 1st Jan of year
    :param DataAray -- dataArray values to be converted
    :param ref -- reference year as int. Default is 1970
    :return -- returns dataarray with units reset and values converted
    """
    #name_conversion=dict(m='minutes',h='hours',d='days')
    
    with xarray.set_options(keep_attrs=True):
        # Need a "null" mask
        OK = ~DataArray.isnull()
        nullT = cftime.num2date(0,units='hours since 1-1-1',calendar='360_day') # what we use for nan
        dt = DataArray.where(OK,nullT).dt
        hour = (dt.year-1970)*360*24+(dt.dayofyear-1)*24+dt.hour
        hour = hour.where(OK,np.nan) # put the nan back in.
    #u = name_conversion.get(unit,unit)
    try:
        hour.attrs['units']=f'hours since {ref_yr}-01-01'
        hour.attrs['calendar']='360_day' # should get that from data
    except AttributeError: # hour does not have attrs.
        pass
    return hour



def write_data(ds, outFile, summary_prefix='seasonal'):
    """
    Write data to netcdf file -- need to fix times!
      So only converts maxTime
    :param ds -- dataset to be written
    :param outFile: file to write to.
    :param summary_prefix. summary prefix text (usually seasonal)
    :return: converted dataset.
    """
    ds2 = ds.copy()  # as modifying things...
    try:
        ds2.attrs['max_time'] = time_convert(ds2.attrs['max_time'])
    except KeyError: # no maxTime.,
        pass
    var = summary_prefix + "MaxTime"
    ds2[var] = time_convert(ds2[var])
    # compress the ouput... (useful because most rain is zero...and quite a lot of the data is missing)
    encoding=dict()
    comp = dict(zlib=True)
    for v in ds2.data_vars:
        encoding[v]=comp
    ds2.to_netcdf(outFile,encoding=encoding)
    return ds2


# read cmd line args.
parser=argparse.ArgumentParser(description="Process UKCP18 CPM data")

parser.add_argument('files',type=str, nargs='+',help='Files to process')
parser.add_argument('--season',type=str,nargs='+',help='Seasons to process (DJF, JJA etc)')
parser.add_argument('--rolling',type=int,nargs='+',help='Additional rolling mean to apply to  data. Specifying nothing means rolling will not be done. If specified each period will be done.')
#parser.add_argument('--test','-t',action='store_true',
#                    help='If set run in test mode -- no data read or generated')
parser.add_argument('--verbose','-v',action='store_true',
                    help='If set be verbose')
parser.add_argument('--outfile','-o',type=str,help='Base name of output file',default='ens_seas_max')
parser.add_argument('--monitor','-m',action='store_true',
                    help='Monitor memory')
parser.add_argument('--region',nargs=4,type=float,help='Region to extract (x0, x1,y0,y1)')
parser.add_argument('--range',nargs=2,type=str,help='Start and end time as ISO string')
parser.add_argument('--test',action='store_true',help='Run test. Exit before processing')
args=parser.parse_args()

if args.verbose:
    print("args are \n",args)

outfile=args.outfile
    
files_glob = [glob.glob(f) for f in args.files]
# flatten array
files=[]
for f in files_glob:
    files.extend(f)
# load files
if args.verbose:
    print("reading in data")
    
chunks=dict(grid_longitude=100,grid_latitude=100)
# ds = xarray.open_mfdataset(files,
#                            chunks=chunks,#parallel=True,
#                            concat_dim='time',combine='nested',
#                            data_vars='minimal',coords='minimal',
#                            compat='override').sortby('time')

ds = xarray.open_mfdataset(files,chunks=chunks).sortby('time')

# extract data if requested
if args.region is not None:
    rgn = args.region # save some typing... 
    ds=ds.sel(grid_longitude=slice(rgn[0],rgn[1]),grid_latitude=slice(rgn[2],rgn[3]))

if args.range is not None:
    ds=ds.sel(time=slice(args.range[0],args.range[1]))

if args.verbose:
    print(f"Have {len(ds.time)} times from {ds.time.min()} to {ds.time.max()}")
    print(ds)
    print("====================================")
if args.test:
    exit()
if args.verbose:
    print("Processing")
dsmax = process(ds.pr).load()# process and load  the raw data
if args.monitor: # report on memory
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss 
    max_mem = mem
    print(f"Memory: {mem}, Max Mem {max_mem}")



if args.rolling is not None:
    seasTot = dsmax['seasonalTotal']
    all_max=[dsmax.drop_vars('seasonalTotal').expand_dims(rolling=1).assign_coords(rolling=[1])]
    for roll in args.rolling:
        print(f"Doing {roll:d} avg") 
        pr=ds.pr.rolling(time=roll).mean()
        name=f'seasonal{roll}'
        dsmx = process(pr,total=False).load()# process and load  the raw data
        dsmx = dsmx.expand_dims(rolling=1).assign_coords(rolling=[roll])
        all_max.append(dsmx)
        if args.monitor: # report on memory
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss 
            max_mem = max(max_mem,mem) 
            print(f"Memory: {mem}, Max Mem {max_mem}")
    all_max = xarray.concat(all_max,'rolling')
    all_max['seasonalTotal']=seasTot

# now write the data out       
outfile=args.outfile+"_"+str(all_max.ensemble_member_id.values[0])[2:-2] # the ensemble
syr=str(int(ds.time.dt.year.min()))
eyr=str(int(ds.time.dt.year.max()))
outfile+="_"+syr+"_"+eyr+".nc"
dsmax2=write_data(all_max, outFile=outfile)
print("Wrote data to :",outfile)


              


