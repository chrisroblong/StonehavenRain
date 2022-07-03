# plot histograms of radar data to see if have strange outliers anywhere..

import pathlib
import handyFns
import xarray
import matplotlib.pyplot as plt
import numpy as np

nimrod_dir=pathlib.Path("/badc/ukmo-nimrod/data/composite/uk-1km")
years = range(2006,2022) # from 2004 to 2021
plt.figure(clear=True,num='QC')
plt.yscale('log')
plt.xscale('log')

bins=np.geomspace(0.1,1024,100)
file_pattern="*0701_*.dat.gz.tar" # All july data
rain=dict()
for year in years:
    directory = nimrod_dir/f'{year}'
    glob_pattern = f"*{year}{file_pattern}"
    files = directory.glob(glob_pattern)
    for file in files:
        rain[year] = handyFns.extract_nimrod_day(file)
        print(f"{year}")
        rain[year].plot.hist(bins=bins,label=f"{year}")

plt.legend(loc=2,ncol=2)
plt.show()


