#!/bin/env python
"""
Combine summary files to produce 1 file for Edinburgh Region
"""
import pathlib
import edinburghRainLib
import xarray
import numpy as np
import pandas as pd
import argparse
import edinburghRainLib
import dask

parser=argparse.ArgumentParser(description="combine processed radar data")
parser.add_argument("dir",type=str,help='Dir to process')
parser.add_argument("output",type=str,help='Name of Output file')
parser.add_argument("--nocompress","-n",action='store_true',
                    help="Do not compress output data")
parser.add_argument("--verbose","-v",action='store_true',help='Be verbose')


args=parser.parse_args()
if args.verbose:
    print(args)


dask.config.set({"array.slicing.split_large_chunks": True})
chunks=dict(time=24,projection_y_coordinate=200,projection_x_coordinate=200)
files=pathlib.Path(args.dir).glob("*monthly.nc")
# remove all the 2004 files
files_wanted = sorted([ file for file in files if '2004' not in file.name])
if args.verbose:
    print("Combining ",files_wanted)

ds=xarray.open_mfdataset(files_wanted,chunks=chunks,combine='nested').sel(**edinburghRainLib.edinburgh_region).load()
encoding=dict()

if not args.nocompress:
    # compress the ouput... useful because quite a lot of the data is missing
    comp = dict(zlib=True)
    for v in ds.data_vars:
        encoding[v]=comp

ds.to_netcdf(args.output,encoding=encoding)





