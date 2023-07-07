"""
Do QC on the 1kmx1km radar data. (Plot mean seasonal rainfall & extremes), distributions & fit to GEV of quantiles from grouping.
Generated figure to be  used in SI.

"""
import scipy.stats

import commonLib
import stonehavenRainLib
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.graphics.gofplots
import  scipy.stats
import pandas as pd




file = stonehavenRainLib.dataDir/'radar_precip/summary_1km_15min.nc'
ed_rgn = {k:slice(v-50e3,v+50e3) for k,v in stonehavenRainLib.stonehaven_crash.items()}
radar = stonehavenRainLib.gen_radar_data(region=ed_rgn)
radar_precip=xarray.open_dataset(file).sel(**ed_rgn)
radar_time = 'Rx15min'
topog_grid=11

# work out summer values.
summer = radar_precip.time.dt.month.isin([6,7,8]) # summer.
SummerRain = radar_precip.monthlyMean.sel(time=summer).resample(time='QS-Jun').mean().dropna('time').load() # max rain in a season
SummerRain *= (30+31+31)/4 # factor of 4 coz 15 minute data and JASMIN code had bug...
mx_seas_rain = radar_precip.monthlyMax.sel(time=summer).load()
mx_SummerRain =mx_seas_rain.resample(time='QS-Jun').max().dropna('time') # max rain in a season

## get in the 90m topography
topogSS = stonehavenRainLib.read_90m_topog(ed_rgn,resample=topog_grid)
top_fit_grid = topogSS.interp_like(radar_precip.monthlyMax.isel(time=0).squeeze())


## plot the data for QC  purposes
cbar_kwrds=dict(orientation='horizontal',fraction=0.1,pad=0.1,label='')
projGB = ccrs.OSGB()
rn_levels=np.array([5,10,20,30,40,50,100,200])
cmap='Blues'
fig = plt.figure(num='radar_qc',clear=True,figsize=[8,7])
gs=fig.add_gridspec(nrows=2,ncols=2,height_ratios=[1.5,0.6])
ax_summer=fig.add_subplot(gs[0,0],projection=projGB)
ax_max = fig.add_subplot(gs[0,1],projection=projGB)


SummerRain.mean('time').plot(robust=True,ax=ax_summer,cbar_kwargs=cbar_kwrds,cmap=cmap)
mx_SummerRain.mean('time').plot(robust=True,ax=ax_max,cbar_kwargs=cbar_kwrds,cmap=cmap)
rgn = []
for v in stonehavenRainLib.stonehaven_crash.values():
    rgn.extend([v-50e3,v+50e3])

for ax,title in zip([ax_summer,ax_max],['Mean JJA Rain (mm)',f'Mean JJA {radar_time} (mm/hr)']):
    ax.set_extent(rgn,crs=projGB)
    stonehavenRainLib.std_decorators(ax,showregions=True)
    ax.plot(*stonehavenRainLib.stonehaven_crash.values(), transform=projGB,
            marker='o', color='purple', ms=9, alpha=0.7)
    c=top_fit_grid.plot.contour(ax=ax,levels=[200,400],colors='black',linewidths=2,linestyles=['solid','dashed'])
    ax.set_title(title)

ax_hist = fig.add_subplot(gs[1, 0])
SummerRain.plot.hist(bins=1000, density=True, ax=ax_hist, color='black')
ax_hist.set_yscale('log')
ax_hist.set_xscale('log')
ax_hist.set_xlabel("JJA total (mm)")
ax_hist_extreme = ax_hist.twiny()
mx_SummerRain.plot.hist(bins=1000, density=True, ax=ax_hist_extreme, color='red', alpha=0.5)
ax_hist_extreme.set_xscale('log')
ax_hist_extreme.set_xlabel(f"JJA {radar_time} (mm/hr)")

ax_pdf = fig.add_subplot(gs[1, 1])
dist = scipy.stats.genextreme
dd=[]
ks=[]
colors = [(0,0,0),(0,0,0.15),(0,0,0.3),(0,0,0.45),(0,0,0.6),(0,0,0.75),(0,0,0.9),(0,0,1)]
colors=['red','brown','orange','green','limegreen','aqua','blue','purple']
for q,col in zip(radar.radar.time_quant,colors):
    print(float(q))
    rr = radar.radar.sel(time_quant=float(q))
    fit = dist.fit(rr)
    d = dist(*fit)
    dd.append(d)
    ks.append(scipy.stats.kstest(rr, d.cdf))

    statsmodels.graphics.gofplots.qqplot(rr, markerfacecolor=col,markeredgecolor=col,dist=dist, fit=True,ax=ax_pdf, line='45', ms=3,label=f'{float(q*100):3.1f}')
ax_pdf.legend(ncol=3,loc='upper left',fontsize='small',handletextpad=0.2,columnspacing=0.5)
ax_pdf.set_title("QQ-Plot")
ks=pd.DataFrame(ks,index=radar.radar.time_quant)
print(ks.pvalue)

label=commonLib.plotLabel()
label.plot(np.array([ax_summer,ax_max,ax_hist,ax_pdf]))





fig.tight_layout()
fig.show()
commonLib.saveFig(fig)