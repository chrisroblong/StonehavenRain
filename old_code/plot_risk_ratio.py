"""
Plot risk ratio. All very empirical.
Approach is to fit data to current block max radar rainfall data. blocks are 20x20 km seasonal max for summer.
Then downscale data by 7.5%/K summer CET warming and also add on 2K warming and see what that does... =
"""
import matplotlib.pyplot as plt

import stonehavenRainLib
import commonLib
import xarray
import numpy as np
import scipy.stats
import numpy.random



def rgn_dist(ds,dist=50.):
    distance = 0.0
    for k, v in  stonehavenRainLib.stonehaven_crash.items():  # iterate over coords.
        distance = distance+(ds[k]-v)**2
    distance = np.sqrt(distance)
    L = (distance <= dist*1e3)
    return ds.where(L)
def comp_seas_max(file,time=None):
    """
    REad in file and compute seasonal max
    :param file: file path to dataset
    :return: the seasonal max value for the monthlyMax in Edinburgh region.
    """
    ds = xarray.load_dataset(file)
    seas_max = ds.monthlyMax.resample(time='QS-Dec').max().load()
    if time is not None:
        seas_max = seas_max.sel(time=time)
    return seas_max
def comp_crit_value(ds):
    return ds.sel(method='nearest',time='2021-06',**stonehavenRainLib.stonehaven_crash)

def fit(data,source_dist,nboot=None,guess={},rng=None):
    """
    Fit data to a distribution.
    :param data: data to be fit
    :param source_dist: distribution to use for fit
    :param guess -- a dictionary with guess values for loc and scale. Empty dict is default,
    :param nboot: number of bootstrap samples
    :return: best fit dist and nboot distributions.
    """


    if rng is None:
        rng =np.random.default_rng()
    params = source_dist.fit(data,**guess)
    dist = source_dist(*params)
    loc=dist.args[2]
    scale=dist.args[1]
    if nboot is None:
        bootstrap=None
    else: # boot strap to get fit.
        bootstrap = []
        npt = data.size
        for indx in range(0,nboot):
            data_samp=rng.choice(data,size=npt)
            params_samp = source_dist.fit(data_samp, loc=loc,scale=scale)
            bootstrap.append(source_dist(*params_samp))

    return dist,bootstrap



source_dist=scipy.stats.genextreme
cet=commonLib.read_cet() # get the CET
time_range_now=slice('2004','2019') # time range used to select data.
nboot=500

delta_CET=cet.sel(time=time_range_now).groupby('time.month').mean()-\
            cet.sel(time=slice('1850','1899')).groupby('time.month').mean()
scale_pi = 1.0-0.075*delta_CET # PI scaling for each season.
scale_p2k = 1.0+0.075*(2-delta_CET) # scaling for each season at 2K warming rel to PI.
seas_max_1km_15min = comp_seas_max("Edinburgh_extremes_1km_15min.nc").coarsen(projection_x_coordinate=25,projection_y_coordinate=25).max()
crit_values = comp_crit_value(seas_max_1km_15min) # what happened in 2021.
datax=rgn_dist(seas_max_1km_15min.sel(time=(seas_max_1km_15min.time.dt.month==6)).sel(time=time_range_now),150)
data = datax.values.flatten() #
data = data[~np.isnan(data)] # remove missing data
data_pi =data*float(scale_pi.sel(month=7))
data_p2k = data*float(scale_p2k.sel(month=7))
dist_now,dist_now_boots = fit(data,source_dist,nboot=nboot)

guess=dict(loc=dist_now.args[2],scale=dist_now.args[1])
dist_pi, dist_pi_boots = fit(data_pi,source_dist,guess=guess,nboot=nboot)
dist_p2k, dist_p2k_boots = fit(data_p2k,source_dist,guess=guess,nboot=nboot)


## plot data
x=np.linspace(50,200,40)
#x=np.linspace(5,55,50)
uncert_now=np.array([d.sf(x) for d in dist_now_boots])
uncert_pi=np.array([d.sf(x) for d in dist_pi_boots])
uncert_p2k=np.array([d.sf(x) for d in dist_p2k_boots])
fig,axes = plt.subplots(nrows=2,ncols=1,num='risk_ratio',clear=True,sharex=True)
# plot the sf values
colours=['red','green','blue']
labels = ['PI','2004-2020','PI+2K']
for dist,uncert,color,label in zip([dist_pi,dist_now,dist_p2k],
                                    [uncert_pi,uncert_now,uncert_p2k],
                                   colours,labels):
    y=1.0/dist.sf(x)
    uu =np.percentile(1.0 / uncert,[5,95],axis=0)
    axes[0].fill_between(x, uu[0,:],uu[1,:], linewidth=2, color=color,alpha=0.33)
    axes[0].plot(x,y,linewidth=2,color=color,label=label)




for dist, uncert,color, label in zip([dist_now, dist_p2k], [uncert_now,uncert_p2k],colours[1:], labels[1:]):
    rr=dist.sf(x)/dist_pi.sf(x)
    uu = np.percentile(uncert/uncert_pi, [5, 95], axis=0)
    axes[1].fill_between(x, uu[0, :], uu[1, :], linewidth=2, color=color, alpha=0.33)
    axes[1].plot(x, rr, linewidth=2, color=color)
    axes[1].axhline(1.0,color='k',linestyle='solid')
# decorate axis

axes[0].set_title("P(x>X)")
axes[0].set_yscale('log')
axes[1].set_title("Prob. Ratio (relative to PI)")
axes[0].legend()
axes[0].axhline(10) # 1 in ten year event.
axes[0].set_ylabel('Return Period (years)')
axes[1].set_ylabel('PR')
axes[1].set_xlabel("max 15-min Rainfall Rate (mm/hr)")
for ax in axes:
    ax.axvline(crit_values,linestyle='dashed')

fig.tight_layout()
fig.show()

## print out rr for crit_value.
x=crit_values
percent = [5,50,95]
crit_uncert_pi=np.array([d.sf(x) for d in dist_pi_boots])
crit_pi = dist_pi.sf(x)
for dist, dist_boots,title in zip([dist_now,dist_p2k],[dist_now_boots,dist_p2k_boots],['Now','+2K']):
    rr = float(dist.sf(x)/crit_pi)
    ratio_uncert = np.array([d.sf(x) for d in dist_boots])/crit_uncert_pi
    rr_range = np.percentile(ratio_uncert,percent)
    print(f"{title} {rr:3.1f} ({rr_range[0]:3.1f} - {rr_range[-1]:3.1f})")



