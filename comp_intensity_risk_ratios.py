"""
Compute intensity and risk ratios from observed data with scalings from simulations
"""
import commonLib
import gev_r
import xarray
import matplotlib.pyplot as plt
import rioxarray
import edinburghRainLib
import scipy.stats
import numpy as np
import pandas as pd

# get in CET and compute differences for PI and PI+2K CET
# read in radar data, topogrophy and mask.
# read in simulated ratios
# read in CPM topog.
# mask to 0-200m.
# compute mean ratio of both scale & location.
# compute distribution for radar data.
# modify param fits to give PI & PI+2K
# Compute and plot intensity & probability  ratios
# to compute uncertainties...
# Bootstrap 1000 times.
# Bootstrap sample mean scalings
# bootstrap sample radar data to compute distribution (in both space and time)
# Apply random scalings to distribution to give PI & PI+2K
# Compute IR & PR.
# Work out the 5 & 95% of the final distribution.
def gen_intens_prob_ratios(radar_data,sim_scale_data,rain_values,return_period_values,temperatures,
                           PI_name='PI',model_horiz_coord='horizontal'):
    """


    :param radar_data:  radar data to be fit.
    :param sim_scale_data: simulated scale data to be used.
    :param rain: rain values wanted
    :param return_period_values: return periods wanted
    :param temperatures: temperatures. Dict -- keys are names. Values are temperature differences from PI. MUST include whatever PI_name is.
    :param PI_name: name of PI temperatures.
    :param model_horiz_coord: model horizontal co-ord or list of coordinates
    :param sample_fraction: What fraction of the data to use. Given auto-correlation recommend sampling every 10x10km.
    :return: dataframes of the intensity, probability ratios relative to PI, and gev fit to current distribution.
    """
    # setup.
    sf= 1.0/return_period_values
    # Fit the masked observed data to a GEV -- using R



    radar_fit = gev_r.gev_fit(radar_data,returnType='DataSet')
    mn_scale = sim_scale_data.mean(model_horiz_coord)
    # generate all distributions at different temperatures using the sim_scale_data to scale
    dist_radar=dict()
    for k,v in temperatures.items():
        delta = v-temperatures['2005-2020']
        dist_radar[k]= scipy.stats.genextreme(radar_fit.Parameters.sel(parameter='shape'),
                                            scale=radar_fit.Parameters.sel(parameter='scale')*(1+mn_scale.sel(parameter='Dscale')*delta),
                                            loc=radar_fit.Parameters.sel(parameter='location')*(1+mn_scale.sel(parameter='Dlocation')*delta)
                                              )

    # now to compute intensity and probability ratios relative to PI. (I think these should look like model..)

    PI=dist_radar.pop(PI_name)
    PI_sf = PI.sf(rain_values)
    PI_isf = PI.isf(sf)
    all_pr=dict()
    all_ir=dict()
    for k,v in dist_radar.items():
        d_sf = v.sf(rain_values) # probability of rain or larger
        d_isf = v.isf(sf) # rain at the prob value
        all_pr[k]=d_sf/PI_sf
        all_ir[k]=d_isf/PI_isf

    all_ir = pd.DataFrame(all_ir,index=pd.Index(1.0/sf,name='Return Period'))
    all_pr = pd.DataFrame(all_pr,index=pd.Index(rain_values,name=radar_time))
    return all_ir.to_xarray(),all_pr.to_xarray(),radar_fit
file = edinburghRainLib.dataDir/'radar_precip/summary_1km_15min.nc'
out_dir=edinburghRainLib.dataDir/'ratios'/("".join(file.stem.split('_')[1:3])) # where derived results get written
out_dir.mkdir(parents=True,exist_ok=True)
# get in observed CET
obs_cet = commonLib.read_cet()
t_today = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('2005', '2021')).mean())
t_pi = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1850', '1899')).mean())
t_p2k = 2.0 * 0.92 + t_pi  # how much more CET is at +2K warming Values provided by Prof. Ed Hawkins (Reading)
t_p2ku = 2.0 * (0.92 + 0.32 / 2 * 1.65) + t_pi  # upper
t_p2kl = 2.0 * (0.92 - 0.32 / 2 * 1.65) + t_pi  # lower
temperatures = {'PI': t_pi, '2005-2020': t_today, 'PI+2K': t_p2k, 'PI+2Ku': t_p2ku, 'PI+2Kl': t_p2kl}

# and the radar & topog

ed_rgn_OSGB = {k:slice(v - 50e3, v + 50e3) for k, v in edinburghRainLib.edinburgh_castle.items()}
topog=rioxarray.open_rasterio(edinburghRainLib.dataDir/'uk_srtm')
topog = topog.reindex(y=topog.y[::-1]).rename(x='projection_x_coordinate',y='projection_y_coordinate')
topog = topog.sel(**ed_rgn_OSGB).load().squeeze()
L=(topog > -10000) & (topog < 10000) # fix bad data. L is where data is good!
topog=topog.where(L)



radar_precip=xarray.open_dataset(file).sel(**ed_rgn_OSGB)
radar_time = 'Rx1hr'
if file.name.endswith('15min.nc'): # 15min data.
    radar_time= 'Rx15min'
if '5km' in file.name:
    topog_grid=55
else:
    topog_grid=11
# extract the summer data
summer = radar_precip.time.dt.month.isin([6,7,8]) # summer.
mean = radar_precip.monthlyMean.sel(time=summer).load()
mx_seas_rain = radar_precip.monthlyMax.sel(time=summer).load()
mx_seas_rain =mx_seas_rain.resample(time='QS-Jun').max().dropna('time') # max rain in a season

top_fit_grid = topog.coarsen(projection_x_coordinate=topog_grid, projection_y_coordinate=topog_grid,boundary='pad').mean()
top_fit_grid = top_fit_grid.interp_like(radar_precip.monthlyMax.isel(time=0).squeeze())

# get in the simulated ratios .. sens as fraction per K of CET change.
model_horiz_coords = ['grid_latitude', 'grid_longitude']
radar_horiz_coords =edinburghRainLib.horizontal_coords
rgn='CET'
sim_sens = xarray.load_dataarray(edinburghRainLib.dataDir/'gev_fits'/f'{rgn}_sens_gev_params.nc')
# get in the topographic info for the CPM. Note coords differ slightly from ed_extreme prob due to FP changes.
cpm_ht = xarray.load_dataset(edinburghRainLib.dataDir / 'orog_land-cpm_BI_2.2km.nc', decode_times=False).squeeze()
cpm_ht = cpm_ht.sel(
    longitude=slice(sim_sens.grid_longitude.min() - 0.01, sim_sens.grid_longitude.max() + 0.01),
    latitude=slice(sim_sens.grid_latitude.min() - 0.01, sim_sens.grid_latitude.max() + 0.01))
cpm_ht = cpm_ht.rename(longitude='grid_longitude', latitude='grid_latitude')
# need to fix the longitudes & latitudes...
cpm_ht = cpm_ht.assign(grid_longitude=sim_sens.grid_longitude,
                       grid_latitude=sim_sens.grid_latitude).ht

msk_cpm=(cpm_ht > 0 ) & (cpm_ht < 200)
msk_OSGB=(top_fit_grid > 0 ) & (top_fit_grid < 200) # between sea-level and 200m
## compute if neeed.
refresh=True

if refresh: # compute and save data. Done because bootstrapping is very slow.
    rng = np.random.default_rng()
    npt = 200
    n_monte_carlo = 1000  # Quick because sampling over the auto-correlation scale gives hardly any data.
    corr_scale = 20.
    sample_fraction_OSGB = (1/corr_scale)**2 # what fraction to sample
    sample_fraction_CPM= (2.2/corr_scale)**2

    rain = np.linspace(10, 150, npt)
    rtn_period = np.geomspace(5, 150, npt)

    L = msk_cpm.values.flatten()
    sim_sens_msk = sim_sens.stack(horizontal=model_horiz_coords)[:, L]
    L2 = msk_OSGB.stack(horizontal=radar_horiz_coords)
    radar_data = mx_seas_rain.stack(horizontal=radar_horiz_coords)[:, L2].values.flatten()  # data > 0 < 200
    all_ir, all_pr, dist_now = gen_intens_prob_ratios(radar_data, sim_sens_msk, rain, rtn_period, temperatures)
    all_pr.to_netcdf(out_dir/'prob_ratio.nc')
    all_ir.to_netcdf(out_dir / 'intense_ratio.nc')
    dist_now.to_netcdf(out_dir/'dist_now.nc')
    # monte carlo time...
    nradar = len(radar_data)
    nsens = sim_sens_msk.shape[1]
    mc_all_ir=[]
    mc_all_pr=[]
    mc_dist_now=[]
    print(f"Monte Carlo Sampling {n_monte_carlo} times")
    for indx in range(0, n_monte_carlo):

        if ((indx+1)%20 ==0):
            print(f"{indx}",end=' ')
        nOSGB = msk_OSGB.size
        nCPM = msk_cpm.size
        indx_OSGB = rng.permutation(nOSGB)[
                    0:int(nOSGB * sample_fraction_OSGB)]  # all points not just those between 0 and 200m.
        indx_cpm = rng.permutation(nCPM)[
                   0:int(nCPM * sample_fraction_CPM)]  # all points not just those between 0 and 200m.
        L = msk_cpm.values.flatten()[indx_cpm]
        sim_sens_msk = sim_sens.stack(horizontal=model_horiz_coords).isel(horizontal=indx_cpm)[:, L]
        L2 = msk_OSGB.stack(horizontal=radar_horiz_coords).isel(horizontal=indx_OSGB)
        radar_data = mx_seas_rain.stack(horizontal=radar_horiz_coords).isel(horizontal=indx_OSGB)[:,
                     L2].values.flatten()  # data > 0 < 200
        ir,pr,dist = gen_intens_prob_ratios(radar_data,sim_sens_msk,rain,rtn_period,temperatures)
        mc_all_ir.append(ir)
        mc_all_pr.append(pr)
        mc_dist_now.append(dist)

    mc_all_ir=xarray.concat(mc_all_ir, dim='sample')
    mc_all_pr=xarray.concat(mc_all_pr, dim='sample')
    mc_dist_now = xarray.concat(mc_dist_now,dim='sample')
    mc_all_pr.to_netcdf(out_dir / 'mc_prob_ratio.nc')
    mc_all_ir.to_netcdf(out_dir / 'mc_intense_ratio.nc')
    mc_dist_now.to_netcdf(out_dir/'mc_dist_now.nc')
else:
    all_pr = xarray.load_dataset(out_dir/'prob_ratio.nc')
    all_ir = xarray.load_dataset(out_dir / 'intense_ratio.nc')
    mc_all_pr = xarray.load_dataset(out_dir / 'mc_prob_ratio.nc')
    mc_all_ir = xarray.load_dataset(out_dir / 'mc_intense_ratio.nc')

# compute return periods for "now"
## make plot
q = [0.05, 0.95] # quantiles for fit between
rp_now = 1./gev_r.xarray_sf(dist_now.Parameters,'Rx15min',x=rain)
mc_rp_now = (1./gev_r.xarray_sf(mc_dist_now.Parameters,'Rx15min',x=rain)).quantile(q,'sample')

colors_ls=  {'2005-2020':('blue','solid'),'PI+2K':('red','solid'),'PI+2Ku':('darkred','dashed'),'PI+2Kl':('lightcoral','dashed')}
fig,ax=plt.subplots(nrows=1,ncols=3,num='prob_radar_cpm',clear=True,figsize=[11,5])
ax[2].fill_between(mc_rp_now.Rx15min,mc_rp_now.isel(quantile=0),y2=mc_rp_now.isel(quantile=1),color='grey')
rp_now.plot(color='red',ax=ax[2],linewidth=3)
ax[2].set_yscale('log')
ax[2].set_ylabel("Return Period (years)")
ax[2].set_title("Return Period from Radar Data")
for v in [10,100]:
    ax[2].axhline(v,linestyle='dashed')
for k,v in colors_ls.items():
    col, ls =v
    plot_Args=dict(color=col,linestyle=ls)
    ((all_ir[k]*100)-100).plot(ax=ax[0],**plot_Args,label=k)
    all_pr[k].plot(ax=ax[1],**plot_Args)
    # put the bootstrap empirical uncertainties on...

    boot_all_ir_range= (mc_all_ir[k].quantile(q, 'sample') * 100) - 100
    boot_all_pr_range = mc_all_pr[k].quantile(q, 'sample')
    ax[0].fill_between(boot_all_ir_range['Return Period'],
                       boot_all_ir_range.isel(quantile=0),
                       boot_all_ir_range.isel(quantile=1),
                       color=col,alpha=0.5)

    ax[1].fill_between(boot_all_pr_range['Rx15min'],
                       boot_all_pr_range.isel(quantile=0),
                       boot_all_pr_range.isel(quantile=1),
                       color=col,alpha=0.5)

    # add on CC line for intensity.
    dT= temperatures[k]-temperatures['PI']
    ax[0].axhline(dT*7.5,color=col,linewidth=3,linestyle='dotted')


ax[0].legend(ncol=2)
# add on the actual rain for 2021 at the castle and the botanics
for name,site_coords in edinburghRainLib.sites.items():
    color=edinburghRainLib.colors[name]
    value = float(mx_seas_rain.sel(**site_coords,method='nearest').sel(time='2021'))
    value2020 = float(mx_seas_rain.sel(**site_coords, method='nearest').sel(time='2020'))
    for a in ax[1:3]:
        a.axvline(value,color=color,linestyle='solid')
        a.axvline(value2020,color=color,linestyle='dashed')
for a, title,xtitle,ytitle in zip(ax, ['Intensity Ratio Increase (%) ', 'Probability Ratio'],
                                  ['Return Period (Years)',f' {radar_time} (mm/hr)'],['% increase','PR']):
    a.set_title(title)
    a.set_xlabel(xtitle)
    a.set_ylabel(ytitle)

fig.tight_layout()
fig.show()
commonLib.saveFig(fig)