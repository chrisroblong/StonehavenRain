"""
Compute intensity and risk ratios from observed data with scalings from simulations
Makes use of data on fits from plot_edinburgh_geography -- which gives bootstrap estimates for uncertainty in radar distributions.
"""
import functools

import commonLib
import gev_r
import xarray
import matplotlib.pyplot as plt
import rioxarray
import edinburghRainLib
import scipy.stats
import numpy as np
import statsmodels.api as sm


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


def scale_fn(cdf_values, params=None, params_ref=None):
    gev = scipy.stats.genextreme(*params)
    gev_ref = scipy.stats.genextreme(*params_ref)
    ratio = gev.ppf(cdf_values) / gev_ref.ppf(cdf_values)
    return ratio


def gen_intens_prob_ratios(radar_fit_all, dScale, rain_values, return_period_values, temperatures,
                           scale_ref='2005-2020',ref_name='PI'):
    """

    :param radar_data:  radar data to be fit.
    :param dScale: simulated scale data to be used.
    :param rain_values: rain values wanted
    :param return_period_values: return periods wanted
    :param temperatures: temperatures. Dict -- keys are names. Values are temperature differences from present day.
    :param scale_ref: Reference for scaling.
    :param ref_name: name of reference for which Prob and intensity ratios are being computed.
    :return: dataset of the intensity, probability ratios relative to PI, and gev fit to current distribution.
    """
    # setup.
    sf_values = 1.0 / return_period_values
    # need values in order, shape, loc, scale
    param_names = ['shape', 'location', 'scale']
    # Fit the masked observed data to a GEV -- using R
    radar_fit = radar_fit_all.sel(parameter=param_names)
    # compute the rainfall for the  specified sf from the rainfall data
    # generate all distributions at different temperatures by scaling parameters by CPM values
    scale_radar = dict()
    sf = dict()
    isf = dict()
    scales = dict()
    for k in temperatures.keys():
        delta_t = temperatures[k]-temperatures[scale_ref]
        scale = 1+(dScale * delta_t)

        scales[k] = scale  # and store it.
        s = np.append(1,scale.sel(parameter=['Dlocation','Dscale']))
        radar_fit_scale = s * radar_fit  # compute the scaled distributions.
        sf[k] = gev_r.xarray_sf(radar_fit_scale, output_dim_name='Rx15min',
                                x=rain_values)  # probability for different rain values
        isf[k] = gev_r.xarray_isf(radar_fit_scale, p=sf_values)  # rainfall for different p-values
    # now to compute intensity and probability ratios relative to 2005-2020

    ref_sf = sf.pop(ref_name)
    ref_isf = isf.pop(ref_name)
    all_pr = dict()
    all_ir = dict()
    for k in sf.keys():
        all_pr[k] = sf[k] / ref_sf
        all_ir[k] = isf[k] / ref_isf

    all_ir = xarray.Dataset(all_ir)
    all_pr = xarray.Dataset(all_pr)
    scales = xarray.Dataset(scales)
    # convert intensity ratio probabilities to return periods.
    all_ir = all_ir.assign_coords(probability=1.0 / all_ir.probability.values).rename(probability='Return Period')
    return all_ir, all_pr, radar_fit, scales


file = edinburghRainLib.dataDir / 'radar_precip/summary_1km_15min.nc'
radar_time = 'Rx15min'
out_dir = edinburghRainLib.dataDir / 'ratios' / (
    "".join(file.stem.split('_')[1:3]))  # where derived results get written
out_dir.mkdir(parents=True, exist_ok=True)

mc_radar_fit = xarray.load_dataset(edinburghRainLib.dataDir / 'radar_precip' / 'bootstrap_reg_radar_params.nc')
radar_fit = xarray.load_dataset(edinburghRainLib.dataDir / 'radar_precip' / 'reg_radar_params.nc')
radar = xarray.load_dataset(edinburghRainLib.dataDir / 'radar_precip' / 'reg_radar_rain.nc')
radar_data = radar.radar
rgn = 'CET'
bs_file= edinburghRainLib.dataDir/'gev_fits'/f'{rgn}_bootstrap_scale.nc'
mn_file= edinburghRainLib.dataDir/'gev_fits'/f'{rgn}_mean_scale.nc'
mn_ratio = xarray.load_dataarray(mn_file, decode_times=False)
bootstrap_ratio = xarray.load_dataarray(bs_file, decode_times=False)
# get in observed CET
obs_cet = commonLib.read_cet()
t_ref = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('2005','2020')).mean()) # mn temp for radar data
t_today= float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('2012','2021')).mean())
t_1980s= float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1980','1989')).mean())
t_1961_1990 = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1961', '1990')).mean())
t_pi = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1850', '1899')).mean())
temp_p2k = scipy.stats.norm(loc=2 * 0.94 + t_pi, scale=2 * 0.03)
# how much more summer-time CET is at +2K warming Values provided by Prof. Ed Hawkins (Reading)

temperatures = {'PI': t_pi, '2012-2021': t_today, '2005-2020':t_ref,'1961-1990': t_1961_1990, '1980s':t_1980s, 'PI+2K': temp_p2k.mean()}
npt=200
rain = np.linspace(5, 200, npt)
rtn_period = np.geomspace(5, 200, npt)
all_ir, all_pr, dist_now, scales = gen_intens_prob_ratios(radar_fit.Parameters, mn_ratio, rain, rtn_period,
                                                          temperatures,scale_ref='2005-2020')

ts = xarray.open_dataset(file).monthlyMax.sel(**edinburghRainLib.edinburgh_castle, method='nearest')
castle2021 = float(ts.sel(time='2021-07'))
castle2020 = float(ts.sel(time='2020-08'))
n_monte_carlo =bootstrap_ratio.shape[0]
refresh=False
## compute if neeed. Set refresh to True. If False data will be read in.

rng = np.random.default_rng()
dist_now = xarray.Dataset(dict(Parameters=dist_now))
for v, fname in zip([all_pr, all_ir, dist_now, scales],
                    ['prob_ratio.nc', 'intense_ratio.nc', 'dist_now.nc', 'scales.nc']):
    v.to_netcdf(out_dir / fname)
if refresh:

    mc_all_ir = []
    mc_all_pr = []
    mc_dist_now = []
    mc_scales = []
    temperatures_mc = temperatures.copy()
    print(f"Bootstrapping {n_monte_carlo} times")
    for indx in range(0, n_monte_carlo):

        if ((indx + 1) % 20 == 0):
            print(f"{indx}", end=' ')
        if ((indx + 1) % 200 == 0):
            print(">")

        mc_scale  = bootstrap_ratio.isel(sample=indx)
        mc_radar_params = mc_radar_fit.isel(sample=indx).Parameters
        t = temp_p2k.rvs(random_state=rng)
        temperatures_mc['PI+2K'] = t
        ir, pr, dist_now_mc, scales_mc = gen_intens_prob_ratios(mc_radar_params, mc_scale, rain, rtn_period,
                                                                temperatures_mc, scale_ref='2005-2020')
        mc_all_ir.append(ir)
        mc_all_pr.append(pr)
        mc_dist_now.append(dist_now_mc)
        mc_scales.append(scales_mc)

    mc_all_ir = xarray.concat(mc_all_ir, dim='sample')
    mc_all_pr = xarray.concat(mc_all_pr, dim='sample')
    mc_dist_now = xarray.concat(mc_dist_now, dim='sample')
    mc_scales = xarray.concat(mc_scales, dim='sample')
    mc_dist_now = xarray.Dataset(dict(Parameters=mc_dist_now))
    for v, fname in zip([mc_all_pr, mc_all_ir, mc_dist_now, mc_scales],
                        ['mc_prob_ratio.nc', 'mc_intense_ratio.nc', 'mc_dist_now.nc', 'mc_scales.nc']):
        v.to_netcdf(out_dir / fname)
else:
    all_pr = xarray.load_dataset(out_dir / 'prob_ratio.nc')
    all_ir = xarray.load_dataset(out_dir / 'intense_ratio.nc')
    dist_now = xarray.load_dataset(out_dir / 'dist_now.nc')
    scales = xarray.load_dataset(out_dir / 'scales.nc', decode_times=False)
    mc_all_pr = xarray.load_dataset(out_dir / 'mc_prob_ratio.nc')
    mc_all_ir = xarray.load_dataset(out_dir / 'mc_intense_ratio.nc')
    mc_dist_now = xarray.load_dataset(out_dir / 'mc_dist_now.nc')
    mc_scales = xarray.load_dataset(out_dir / 'mc_scales.nc', decode_times=False)

# get the expected CC rate which arises because Edinburgh Temperatures increase at about 90% of CET rates.
qsat_fit = sm.load(edinburghRainLib.dataDir/'gev_fits'/'cet_qsat_fit.sav')
f=qsat_fit.get_prediction([1,t_today])
cc_scale = float(100*(qsat_fit.params[1]/f.predicted_mean))
cc_rng = 100*qsat_fit.conf_int()[1,:]/f.predicted_mean
# compute return periods for "now"
## make plot
q = [0.05, 0.95]  # quantiles for fit between
qt = 0.95
# rp_now = 1./gev_r.xarray_sf(dist_now.Parameters,'Rx15min',x=rain)
# mc_rp_now = (1./gev_r.xarray_sf(mc_dist_now.Parameters,'Rx15min',x=rain)).quantile(q,'sample')

colors_ls = {
    #'1961-1990': ('skyblue', 'solid'),
    '1980s':('royalblue','solid'),
    '2012-2021': ('blue', 'solid'),
    'PI+2K': ('red', 'solid')}

def plot_cc(dT,ax,color,cc_scale,cc_range):
    """
    Plot the CC range
    :param dT: temperature change
    :param ax: axis on which to plot
    :param color: color to use
    :return:

    """
    ax.fill_between(rain,dT*cc_range[0],dT*cc_range[1],alpha=0.2,color=color)
    ax.axhline(dT * cc_scale, color=color, linewidth=3, linestyle='dotted')

fig, (ax, ax_sens) = plt.subplots(nrows=2, ncols=2, num='prob_radar_cpm', clear=True, figsize=[9, 6])
for k, v in colors_ls.items():
    col, ls = v
    plot_Args = dict(color=col, linestyle=ls)
    ((all_ir[k].sel(time_quant=qt) * 100) - 100).plot(ax=ax[0], **plot_Args, label=k)
    all_pr[k].sel(time_quant=qt).plot(ax=ax[1], **plot_Args)
    # put the bootstrap empirical uncertainties on...

    mc_all_ir_range = (mc_all_ir[k].sel(time_quant=qt).quantile(q, 'sample') * 100) - 100
    mc_all_pr_range = mc_all_pr[k].sel(time_quant=qt).quantile(q, 'sample')
    ax[0].fill_between(mc_all_ir_range['Return Period'],
                       mc_all_ir_range.isel(quantile=0),
                       mc_all_ir_range.isel(quantile=1),
                       color=col, alpha=0.5)

    ax[1].fill_between(mc_all_pr_range['Rx15min'],
                       mc_all_pr_range.isel(quantile=0),
                       mc_all_pr_range.isel(quantile=1),
                       color=col, alpha=0.5)

    # add on CC line for intensity.
    dT = temperatures[k] - temperatures['PI']
    plot_cc(dT,ax[0],col,cc_scale,cc_rng)


ax[0].set_xlim(5, 150)
ax_sens[0].set_xlim(2, 40)
ax[1].set_xlim(50, 150)
ax[1].set_ylim(1, 2.)
ax[1].yaxis.set_ticks_position('both')
ax_sens[1].yaxis.set_ticks_position('both')
# add some secondary axis to convert rainfall to return period and return period to rainfall for present climate.
# inverse functions
params = dist_now.Parameters.sel(parameter=['shape', 'location', 'scale'], time_quant=qt).values.tolist()
gev_now = scipy.stats.genextreme(*params)
rp_to_rain = lambda x: gev_now.isf(1.0 / x)
rain_to_rp = lambda x: 1.0 / gev_now.sf(x)

secax_rp = ax[0].secondary_xaxis('top', functions=(rp_to_rain, rain_to_rp))
secax_rain = ax[1].secondary_xaxis('top', functions=(rain_to_rp, rp_to_rain))
secax_rp.set_xticks([50, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200])  # rain
secax_rp.set_xlabel(f'Radar {radar_time} (mm/hr)', fontsize='small')
secax_rain.set_xticks([5, 10, 20, 50, 100, 150, 250])
secax_rain.set_xlabel(f'Radar Return Period (Years)', fontsize='small')
# add on the actual rain for 2021 at the castle and the botanics
color = edinburghRainLib.colors['castle']

value = float(radar.critical2021.sel(time_quant=qt))
value2020 = float(radar.critical2020.sel(time_quant=qt))

for v, ls in zip([value, value2020], ['solid', 'dashed']):
    ax[0].axvline(rain_to_rp(v), color=color, linestyle=ls)
    ax[1].axvline(v, color=color, linestyle=ls)

# plot sensitivity to changing the quantile within the event,
rp = []
for quant in dist_now.time_quant:
    gev = scipy.stats.genextreme(*dist_now.sel(time_quant=quant).Parameters.values.tolist())
    rp.append(1.0 / gev.sf(radar.critical2021.sel(time_quant=quant)))
rp = np.array(rp)
rp = xarray.DataArray(data=rp, coords=dict(time_quant=dist_now.time_quant))

tq_values = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
rp = rp.sel(time_quant=tq_values)
for k in colors_ls.keys():
    col, ls = colors_ls[k]
    plot_args = dict(linewidth=1, color=col, alpha=0.2)

    # add on CC line for intensity.
    # add on CC line for intensity.
    dT = temperatures[k] - temperatures['PI']
    plot_cc(dT,ax_sens[0],col,cc_scale,cc_rng)
    # all_pr[k].plot.line(x='Rx15min',ax=ax_sens[1],add_legend=False,**plot_args)
    ir = (all_ir[k].sel(time_quant=tq_values).sel({'Return Period': rp}, method='nearest') * 100) - 100
    ir.plot(ax=ax_sens[0], x='Return Period', color=col, ls=ls, linewidth=2, marker='o', ms=6)
    pr = all_pr[k].sel(Rx15min=radar.critical2021.sel(time_quant=tq_values), method='nearest').sel(time_quant=tq_values)
    pr.plot(ax=ax_sens[1], x='Rx15min', color=col, ls=ls, linewidth=2, marker='o', ms=6)
    # plot qt value.
    irr = ir.sel(time_quant=qt)
    ax_sens[0].plot(float(irr['Return Period']), float(irr), color=col, marker='h', ms=12)
    prr = pr.sel(time_quant=qt)
    ax_sens[1].plot(float(prr.Rx15min), float(prr), color=col, marker='h', ms=12)
    # prr.plot(ax=ax_sens[1],x='Rx15min',color=col,marker='x',ms=9)
# plot (with dots) for 2021 the PR's
# annotate the dots with qt values *100 and rounded to 1 SF
k = 'PI+2K'
pr = all_pr[k].sel(Rx15min=radar.critical2021.sel(time_quant=tq_values), method='nearest').sel(time_quant=tq_values)
ir = (all_ir[k].sel(time_quant=tq_values).sel({'Return Period': rp}, method='nearest') * 100) - 100
for time_quant in pr.time_quant:
    ppr = pr.sel(time_quant=time_quant)
    x = float(ppr.Rx15min)
    y = float(ppr)
    ax_sens[1].annotate(f'{float(time_quant * 100):3.1f}'.replace('.0', ''), (x, y),
                        textcoords='offset points', xytext=(7.5, 0), fontsize='small')

    iir = ir.sel(time_quant=float(time_quant))
    x = float(iir['Return Period'])
    y = float(iir)
    ax_sens[0].annotate(f'{float(time_quant * 100):3.1f}'.replace('.0', ''), (x, y),
                        textcoords='offset points', xytext=(3.5, -7.5), fontsize='small')

label = commonLib.plotLabel()
axis = ax.tolist()
axis.extend(ax_sens.tolist())
for a, title, xtitle, ytitle in zip(axis, ['Intensity Ratio Increase (%) ', 'Probability Ratio'] * 2,
                                    ['Return Period (Years)', f' {radar_time} (mm/hr)'] * 2, ['% increase', 'PR'] * 2):
    a.set_title(title)
    a.set_xlabel(xtitle)
    a.set_ylabel(ytitle)

    label.plot(a)
for v, ls in zip([castle2021, castle2020], ['solid', 'dashed']):
    ax_sens[1].axvline(v, color='purple', linestyle=ls)
for v in [1, 1.2, 1.4, 1.6, 1.8]:
    for a in [ax[1], ax_sens[1]]:
        a.axhline(v, color='black', linestyle='dotted')
fig.subplots_adjust(bottom=0.15)  # make room for the legend
a = fig.legend(loc='lower center', ncol=3, fontsize='small')
a.set_in_layout(False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.125)  # make room for the legend
fig.show()
commonLib.saveFig(fig)

## print out the best estimates and ranges for both IR and PR's
value = float(radar.critical2021.sel(time_quant=qt))
rp=rain_to_rp(value) # get return period.
per = lambda x: (x-1)*100.
for k in colors_ls.keys():
    delta_t = temperatures[k]-temperatures['PI']
    cc = cc_scale * delta_t
    pr = float(all_pr[k].sel(time_quant=qt).interp(Rx15min=value))
    prmin=float(mc_all_pr[k].sel(time_quant=qt).quantile(0.05,'sample').interp(Rx15min=value))
    prmax = float(mc_all_pr[k].sel(time_quant=qt).quantile(0.95,'sample').interp(Rx15min=value))
    print(f"PR {k:12s}: {pr:3.2f} ({prmin:3.2f}-{prmax:3.2f})" )
    ir = float(all_ir[k].sel(time_quant=qt).interp({'Return Period':rp}))
    irmin=float(mc_all_ir[k].sel(time_quant=qt).quantile(0.05,'sample').interp({'Return Period':rp}))
    irmax = float(mc_all_ir[k].sel(time_quant=qt).quantile(0.95,'sample').interp({'Return Period':rp}))
    print(f"IR {k:12s}: {per(ir):2.0f} ({per(irmin):2.0f}-{per(irmax):2.0f}) dT {delta_t: 2.1f} CC:{cc:2.0f} " )
