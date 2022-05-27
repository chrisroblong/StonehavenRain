"""
Code to analyse the CPM data. Produces estimate of return times and scalings.
Uncertainty comes from spatial analysis.
"""

import commonLib
import edinburghRainLib
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import gev_r  # so we can fit gev using R
import numpy as np
import matplotlib.colors

recreate_fit = False  # if true create  fit data *even* if file exists


def qsat(temperature):
    """
    Saturated humidity from temperature.
    :param temperature: temperature (in degrees c)
    :return: saturated humidity
    """
    es = 6.112 * np.exp(17.6 * temperature / (temperature + 243.5))
    return es


refresh = False  # if True then regenerate fits **even** if data exists...
outdir_gev = edinburghRainLib.dataDir / 'gev_fits'
outdir_gev.mkdir(parents=True, exist_ok=True)  # create the directory

## read in the data we need
cet = xarray.load_dataset(edinburghRainLib.dataDir / 'cet_cpm.nc').tas
cpm = xarray.load_dataset(edinburghRainLib.dataDir / 'cpm_reg_ts.nc').tas
hum = qsat(cpm)  # compute humidity
ed = xarray.load_dataset(edinburghRainLib.dataDir / 'ed_reg_ts.nc').tas
ed_extreme_precip = xarray.load_dataset(
    edinburghRainLib.dataDir / 'ed_reg_max_precip.nc').pr  ##.isel(grid_longitude=slice(10,20),grid_latitude=slice(10,20))

# get in the topographic info for the CPM. Note coords differ slightly from ed_extreme prob due to FP changes.
cpm_ht = xarray.load_dataset(edinburghRainLib.dataDir / 'orog_land-cpm_BI_2.2km.nc', decode_times=False).squeeze()
cpm_ht = cpm_ht.sel(
    longitude=slice(ed_extreme_precip.grid_longitude.min() - 0.01, ed_extreme_precip.grid_longitude.max() + 0.01),
    latitude=slice(ed_extreme_precip.grid_latitude.min() - 0.01, ed_extreme_precip.grid_latitude.max() + 0.01))
cpm_ht = cpm_ht.rename(longitude='grid_longitude', latitude='grid_latitude')
# need to fix the longitudes & latitudes...
cpm_ht = cpm_ht.assign(grid_longitude=ed_extreme_precip.grid_longitude,
                       grid_latitude=ed_extreme_precip.grid_latitude).ht
msk = (cpm_ht > 0) & (cpm_ht < 200)

# get in observed CET
obs_cet = commonLib.read_cet()
t_today = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('2005', '2021')).mean())
t_pi = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1850', '1899')).mean())

## Do the fit
stack_dims = ['time', 'ensemble_member']

xfit = dict()
covariates = dict()
for ts, title in zip([cet, cpm, ed, hum], ['CET', 'CPM_region', 'Edinburgh_region', 'Humidity']):
    file = outdir_gev / f"cov_{title}.nc"
    ts_summer = ts.resample(time='QS-DEC').mean().dropna('time')
    ts_summer = ts_summer.sel(time=(ts_summer.time.dt.month == 6))
    ts_summer['time'] = ed_extreme_precip.time  # make sure times are the same
    covariates[title] = ts_summer
    xfit[title] = gev_r.xarray_gev(ed_extreme_precip.stack(time_ensemble=stack_dims),
                                   cov=ts_summer.stack(time_ensemble=stack_dims),
                                   dim='time_ensemble', file=file, recreate_fit=refresh, verbose=True)


t_p2k = 2.0 * 0.92 + t_pi  # how much more CET is at +2K warming Values provided by Pro. Ed Hawkins (Reading)
t_p2ku = 2.0 * (0.92 + 0.32 / 2 * 1.65) + t_pi  # upper
t_p2kl = 2.0 * (0.92 - 0.32 / 2 * 1.65) + t_pi  # lower

temperatures = {'PI': t_pi, '2005-2020': t_today, 'PI+2K': t_p2k, 'PI+2Ku': t_p2ku, 'PI+2Kl': t_p2kl}

params_full = dict()
cc_scale = xarray.DataArray(np.array([.075, .075, 0]), coords=dict(parameter=['location', 'scale', 'shape']))
for name, t in temperatures.items():
    params_full[name] = gev_r.param_cov(xfit['CET'], t)
    if name != 'PI':
        if (name[-1] == ('l')) or (name[-1] == ('u')):
            CC_name = name[0:-1] + "_CC" + name[-1]
        else:
            CC_name = name + "_CC"
        delta_t = t - temperatures['PI']
        params_full[CC_name] = params_full['PI'] * (1 + cc_scale * delta_t)

## first do we have super CC effect? Work through the covariates
for key in [f for f in xfit.keys() if 'Humidity' not in f]:
    mn=covariates[key].mean()
    delta_cov = mn+1
    fit = xfit[key]
    ref = gev_r.param_cov(fit,mn)
    delta = gev_r.param_cov(fit,delta_cov)
    percent = 100*(delta/ref-1).where(msk) # percent change.
    model_horiz_coords = ['grid_latitude','grid_longitude']
    with np.printoptions(precision=1,suppress=True):

        print(f"================ {key} ===================")
        # mask data by land < 200m and show quantiles
        print(percent.where(msk).quantile([0.05,0.5,0.95],model_horiz_coords))
        # and print out the fit over the region
        print(f"Neg Log Like: {float(fit.nll.mean()):5.1f} AIC: {float(fit.nll.mean()):5.1f}")

# Second use mean values from CET covariate to examine risk and intensity changes over the region. Using CET as have PI values.
# Will show 5%, 50% & 95% values.
# Will also show CC values..

## compute probability  and intensity ratios
pratio=dict()
iratio=dict()
npt=50
rgn='CET'
rx1hr=np.linspace(10,100,npt)
sf=np.geomspace(1/5,1/250,npt)
pi_params = params_full['PI']
PI_sf = gev_r.xarray_sf(pi_params,x=rx1hr,output_dim_name='Rx1hr').where(msk)
PI_isf = gev_r.xarray_isf(pi_params,p=sf,name='Rx1hr').where(msk)
q=[0.05,0.5,0.95]
for k in ['2005-2020','PI+2K']:
    delta_t = temperatures[k] - temperatures['PI']
    p = params_full[k].where(msk)
    cc_name = k+'_CC'
    params_cc = pi_params * (1 + cc_scale * delta_t)

    iratio[k] = (gev_r.xarray_isf(p,p=sf,name='Rx1hr')/PI_isf).quantile(q,model_horiz_coords) # intensity ration at fixed p values (= rtn periods)
    iratio[cc_name] = (gev_r.xarray_isf(params_cc,p=sf,name='Rx1hr')/PI_isf).quantile(q,model_horiz_coords)

    pratio[k] = (gev_r.xarray_sf(p, x=rx1hr, output_dim_name='Rx1hr') / PI_sf).quantile(q, model_horiz_coords) # risk ratio at fixed precip values.
    pratio[cc_name] = (gev_r.xarray_sf(params_cc, x=rx1hr, output_dim_name='Rx1hr') / PI_sf).quantile(q, model_horiz_coords) # risk ratio at fixed precip values.

# and convert all intensity ratios  into % increase
for k in iratio.keys():
    iratio[k] = (iratio[k]-1.)*100

# want ratios of Dlocation/location & Dscale/scale
D=xfit[rgn].Parameters.sel(parameter=['Dlocation','Dscale'])#.where(msk)
ratio_Dparam = D/params_full['2005-2020'].sel(parameter=['location','scale']).assign_coords(parameter=['Dlocation','Dscale'])
# and want to save the scaling ratios so they can be applied to the precipitation data.
ratio_Dparam.to_netcdf(edinburghRainLib.dataDir/'gev_fits'/f'{rgn}_sens_gev_params.nc')

## and make plot
colors_ls=  {'2005-2020':('blue','solid'),'PI+2K':('red','solid'),'2005-2020_CC':('blue','dashed'),'PI+2K_CC':('red','dashed')}
fig,ax=plt.subplots(nrows=1,ncols=3,num='prob',clear=True,figsize=[11,5])
for k,v in colors_ls.items():
    col, ls =v
    plot_Args=dict(color=col,linestyle=ls)
    rp = 1.0/iratio[k].probability # rtn prd
    rn = pratio[k].Rx1hr # PI rain
    if not k.endswith('CC'): # plot uncerts.
        ax[0].fill_between(rp, y1=iratio[k].sel(quantile=0.05), y2=iratio[k].sel(quantile=0.95), alpha=0.5,**plot_Args) # intensity ratio
        ax[1].fill_between(rn, y1=pratio[k].sel(quantile=0.05), y2=pratio[k].sel(quantile=0.95), alpha=0.5, **plot_Args)  # risk ratio

    ax[0].plot(rp,iratio[k].sel(quantile=0.5),label=k,**plot_Args)
    ax[1].plot(rn, pratio[k].sel(quantile=0.5), label=k, **plot_Args)
ax[0].legend(ncol=2)

# scatter plot # small blobs for less then 200; large blobs for > 200
topog_levels=[0.01,100,200,300,400,500,1000]
norm=matplotlib.colors.BoundaryNorm(topog_levels,ncolors=255,extend='both') # as topography plots.
cm=ax[2].scatter(100*ratio_Dparam.sel(parameter='Dlocation'),100*ratio_Dparam.sel(parameter='Dscale'),c=cpm_ht,cmap='terrain',s=6,norm=norm,marker='o')
ax[2].axhline(7.5,color='black',linestyle='dashed')
ax[2].axvline(7.5,color='black',linestyle='dashed')
fig.colorbar(cm,ax=ax[2],orientation='horizontal',label='Height',fraction=0.1,pad=0.15,extend='both',ticks=topog_levels[::2])


for a, title,xtitle,ytitle in zip(ax, ['Intensity Ratio Increase (%) ', 'Probability Ratio','d loc/dT vs d scale/dT'],
                                  ['PI Return Period (Years)','Rx1hr (mm/hr)','d loc/dT (%/K)'],['% increase','PR','d scale/dT (%/K)']):
    a.set_title(title)
    a.set_xlabel(xtitle)
    a.set_ylabel(ytitle)

fig.tight_layout()
fig.show()





