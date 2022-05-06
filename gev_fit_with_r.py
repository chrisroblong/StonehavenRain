"""
Do GEV fit using  fevd from extRemes **R** package
"""
import commonLib
import edinburghRainLib
import xarray
import scipy.stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import os
import numpy as np
import collections
os.environ['R_HOME']=r'C:\Users\stett2\AppData\Local\Microsoft\AppV\Client\Integration\FC689017-A9BB-4A9B-B971-6AC52117BA03\Root' # where R is...
import rpy2
#print(rpy2.__version__)
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.pandas2ri as rpandas2ri
import re
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.conversion import localconverter

gevFit=collections.namedtuple('gevFit',['params','std_err','AIC','negloglike'])

def qsat(temperature):
    """
    Saturated humidity from temperature.
    :param temperature: temperature (in degrees c)
    :return: saturated humidity
    """
    es=6.112*np.exp(17.6*temperature/(temperature+243.5))
    return es

def gev_fit(x,cov=None,returnTuple=False,shapeCov=False,**kwargs):
    """
    Do GEV fit using R and return named tuple of relevant values.
    :param x: Data to be fit
    :param cov: covariate value (if None then not used)
    :param returnTuple. If True returns a tuple rather than named tuple -- really for apply_ufunc
    :param shapeCov -- If True allow the shape to vary with the covariate.
    :return:
    """
    L=~np.isnan(x)
    df_data = [x[L]]# remove nan from data]
    cols=['x']
    r_code = 'fevd(x=x,data=df'
    if cov is not None:
        df_data.append(cov[L])# remove places where x was nan from cov.
        cols.append('cov')
        r_code = r_code+',location.fun=~cov,scale.fun=~cov'
        if shapeCov:
            r_code += 'shape.fun=~cov'
    r_code += ')' # add on the trailing bracket.
    r_code = 'summary('+r_code+',silent=TRUE)'

    df=pd.DataFrame(np.array(df_data).T,columns=cols)
    with rpy2.robjects.conversion.localconverter(robjects.default_converter + rpandas2ri.converter):
        robjects.globalenv['df'] = df  # push the dataframe with info into R
    fit = robjects.r(r_code)  # do the fit
    # extract the data
    params = np.array(fit.rx2('par'))
    se = np.array(fit.rx2('se.theta'))
    if cov is not None:
        if shapeCov:
            params[-2:] *= -1 # negate shape parameters as python convention differs from R
        else:
            params[-1] *= -1
            params = np.append(params,0.0) # Dshape is 0.
            se=np.append(se,0.0)


    nllh= np.array(fit.rx2('nllh'))
    aic = np.array(fit.rx2('AIC'))
    result = gevFit(params=params,std_err=se,negloglike=nllh,AIC=aic)
    if returnTuple:
        result = tuple(result)
    return result



def xarray_gev(ds,cov,shapeCov=False,**kwargs):
    """
    Fit a GEV to xarray data.
    :param ds: dataset for which GEV is to be fit
    :param cov: covariate
    :param shapeCov: allow the shape to vary with the covariate.
    :param kwargs: any kwargs passed through to the fitting function
    :return: a datset containg:
        Parameters -- the parameters of the fit; location, location wrt cov, scale, scale wrt cov, shape, shape wrt cov
        Stderr -- the standard error of the fit -- same parameters as Parameters
        nll -- negative log likelihood of the fit -- measure of the quality of the fit
        AIC -- aitkin information criteria.
    """
    kwargs['returnTuple']=False
    kwargs['shapeCov']=shapeCov
    params,std_err,nll,AIC = xarray.apply_ufunc(gev_fit, ds, cov, input_core_dims=[['time_ensemble'],['time_ensemble']],
                                                output_core_dims=[['parameter'],['parameter'],['NegLog'],['AIC']],
                                vectorize=True,kwargs=kwargs)
    # name variables and then combine into one dataset.
    pnames = ['location','Dlocation','scale','Dscale','shape','Dshape']
    params = params.rename("Parameters")
    std_err = std_err.rename("StdErr")
    nll = nll.rename('nll').squeeze()
    AIC = AIC.rename('AIC').squeeze()
    ds=xarray.Dataset(dict(Parameters=params,StdErr=std_err,nll=nll,AIC=AIC)).assign_coords(parameter=pnames)
    return ds

## use apply ufunc to generate distributions...

def fn_isf(c,loc,scale,p=None,dist=scipy.stats.genextreme):
    fdist = dist(c, loc=loc,scale=scale)
    x= fdist.isf(p) # values for 1-cdf.
    return x

def fn_sf(c,loc,scale,x=None,dist=scipy.stats.genextreme):

    fdist = dist(c, loc=loc,scale=scale)
    p= fdist.sf(x) # 1-cdf for given x
    return p

def xarray_sf(params,output_dim_name='value',**kwargs):
    """
    Compute the survival value for different values based on dataframe of fit parameters.
    :param params: xarray dataarray of shape, location and scale values
    :param output_dim_name: name of output dimension. Defualt is "value" but set it to what ever you are using. E.g "Rx1hr"
    :param kwargs: passed to fn_sf which does the computation. Must contain x which is used for the computation.
    :return:
    """
    sf = xarray.apply_ufunc(fn_sf,params.sel(parameter='shape'),params.sel(parameter='location'),params.sel(parameter='scale'),
                           output_core_dims=[[output_dim_name]],
                           vectorize=True,kwargs=kwargs)
    sf = sf.assign_coords({output_dim_name:kwargs['x']}).rename('sf')

    return sf

def xarray_isf(params,name='value',**kwargs):
    """
    Compute the inverse survival function for given values of sf.
    :param params:
    :param name:
    :param kwargs:
    :return:
    """
    output_dim_name = 'probability'
    x = xarray.apply_ufunc(fn_isf,params.sel(parameter='shape'),params.sel(parameter='location'),params.sel(parameter='scale'),
                           output_core_dims=[[output_dim_name]],
                           vectorize=True,kwargs=kwargs)
    x = x.assign_coords({output_dim_name:kwargs['p']})#.rename(name)
    return x

def param_cov(params,cov):
    p = ['location', 'scale', 'shape']
    p2 = ["D" + a.lower() for a in p]
    params_c = params.sel(parameter=p2).assign_coords(parameter=p) * cov + params.sel(parameter=p)
    return params_c

refresh=False # if True then regenerate fits **even** if data exists...
outdir_gev = edinburghRainLib.dataDir/'gev_fits'
outdir_gev.mkdir(parents=True,exist_ok=True) # create the directory

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
# R package names
packnames = (['extRemes']) # list of packages to install.
#From example rpy2 install what needs to be installed.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(robjects.vectors.StrVector(names_to_install))
for package in packnames:
    rpackages.importr(package) #so available


## Do the fit
cet = xarray.load_dataset(edinburghRainLib.dataDir/'cet_cpm.nc').tas
cpm = xarray.load_dataset(edinburghRainLib.dataDir/'cpm_reg_ts.nc').tas
hum= qsat(cpm) # compute humidity
ed= xarray.load_dataset(edinburghRainLib.dataDir/'ed_reg_ts.nc').tas
ed_extreme_precip = xarray.load_dataset(edinburghRainLib.dataDir/'ed_reg_max_precip.nc').pr##.isel(grid_longitude=slice(10,20),grid_latitude=slice(10,20))
stack_dims= ['time','ensemble_member']

xfit=dict()
for ts,title in zip([cet,cpm,ed,hum],['CET','CPM_region','Edinburgh_region','Humidity']):
    file = outdir_gev/f"cov_{title}.nc"
    if refresh or (not file.exists()): #
        print(f"Generating data for {file}")
        cet_summer=ts.resample(time='QS-DEC').mean().dropna('time')
        ts_summer = ts_summer.sel(time=(ts_summer.time.dt.month==6))
        ts_summer['time'] = ed_extreme_precip.time # make sure times are the same
        xfit[title] = xarray_gev(ed_extreme_precip.stack(time_ensemble=stack_dims), ts_summer.stack(time_ensemble=stack_dims))
        xfit[title].to_netcdf(file) # write the data.
    xfit[title]=xarray.load_dataset(file) # load the data
    print(f"Done {title}")

## get in the topographic info for the CPM. Note values differ slightly..
cpm_ht = xarray.load_dataset(edinburghRainLib.dataDir / 'orog_land-cpm_BI_2.2km.nc', decode_times=False).squeeze()
cpm_ht = cpm_ht.sel(longitude=slice(ed_extreme_precip.grid_longitude.min()-0.01, ed_extreme_precip.grid_longitude.max()+0.01),
                    latitude=slice(ed_extreme_precip.grid_latitude.min()-0.01, ed_extreme_precip.grid_latitude.max()+0.01))
cpm_ht = cpm_ht.rename(longitude='grid_longitude', latitude='grid_latitude')
# need to fix the longitudes & latitutes...
cpm_ht= cpm_ht.assign(grid_longitude=ed_extreme_precip.grid_longitude,grid_latitude=ed_extreme_precip.grid_latitude)
## generate summary
summary=dict()
summary_sd=dict()
rng=dict()
horiz_coords=['grid_latitude','grid_longitude']
hindx=np.random.random_integers(0,high=2500,size=100)
for title,fit in xfit.items():
    # noinspection PyDeprecation
    fit_d =  fit.drop(['longitude','latitude'],errors='ignore') # drop nuisance co-ords!
    summary[title]=fit_d.mean(horiz_coords)
    summary_sd[title]=fit_d.std(horiz_coords)
    #print(title, "\n Mean ============ \n",summary[title],"\n SD:",(summary_sd[title].Parameters/np.sqrt(fit_d.AIC.size)).values)  # print out summary values.
## get in the OBS CET data
obs_cet = commonLib.read_cet()
t_today = float(obs_cet.sel(time=(obs_cet.time.dt.month==7)).sel(time=slice('2004','2021')).mean())
t_pi = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1850', '1899')).mean())
## now to plot
projRot=ccrs.RotatedPole(pole_longitude=177.5,pole_latitude=37.5)
proj = ccrs.PlateCarree()
projOS=ccrs.OSGB(approx=True)
kw_cbar=dict(orientation='horizontal',fraction=0.05,pad=0.1,label='')
cmap='Blues'
fig,axes = plt.subplots(nrows=2,ncols=3,num='gevFit',clear=True,figsize=[10,7],subplot_kw=dict(projection=projOS))
rgn='CET'
fit = xfit[rgn]
fit.nll.plot(ax=axes[0][0],robust=True,transform=projRot,cbar_kwargs=kw_cbar)
axes[0][0].set_title("Negative Log Likelihood")
value_today = param_cov(fit.Parameters,t_today)
for title,ax in zip(['location','scale'],axes[0][1:]):
    value_today.sel(parameter=title).plot(ax=ax,robust=True,transform=projRot,cbar_kwargs=kw_cbar,cmap=cmap)
    cpm_ht.ht.plot.contour(ax=ax,levels=[0,200],colors=['green','brown'],linewidth=2,transform=projRot)
    ax.set_title(f"{title} Rx1hr")

for title,v,cm,ax in zip(['location','scale'],[(3,10),(5,12),(-5,5)],["Blues","Blues","RdBu"],axes[1]):
    ratio = 100*fit.Parameters.sel(parameter="D"+title)/value_today.sel(parameter=title)
    ratio.plot(ax=ax,transform=projRot,cbar_kwargs=kw_cbar,vmin=v[0],vmax=v[1],cmap=cm)
    cpm_ht.ht.plot.contour(ax=ax, levels=[0, 200], colors=['green', 'brown'], linewidth=2, transform=projRot)
    ax.set_title(r"$\Delta$"+f"{title} %Rx1hr/T")


reg = [float(fit.grid_longitude.min()),float(fit.grid_longitude.max()),
        float(fit.grid_latitude.min()),float(fit.grid_latitude.max())
       ]

ed_castle=edinburghRainLib.edinburgh_castle.values()


for ax in axes.flatten():
    ax.set_extent(reg,crs=projRot)
    ax.plot(*ed_castle, transform=projOS, color='black', marker='o', ms=12,alpha=0.5)
    edinburghRainLib.std_decorators(ax)
fig.tight_layout()
fig.show()

## compute parameters and generate distributions.

fitS=summary[rgn].Parameters

p=['location','scale','shape']
p2 = ["D"+a.lower() for a in p]
t_p2k =2.0*0.92+t_pi # how much more CET is at +2K warming.
t_p2ku = 2.0*(0.92+0.32/2*1.65)+t_pi # upper
t_p2kl = 2.0*(0.92-0.32/2*1.65)+t_pi # lower


temperatures = {'PI':t_pi,'2005-2020':t_today,'PI+2K':t_p2k,'PI+2Ku':t_p2ku,'PI+2Kl':t_p2kl}
params=dict()
params_full = dict()
cc_scale=xarray.DataArray(np.array([.075,.075,0]),coords=dict(parameter=['location','scale','shape']))
for name, t in temperatures.items():
    params[name] = param_cov(fitS,t)
    params_full[name]=param_cov(fit,t)
    if name != 'PI':
        if (name[-1] == ('l')) or (name[-1] == ('u')):
            CC_name = name[0:-1]+"_CC"+name[-1]
        else:
            CC_name = name+"_CC"
        delta_t = t - temperatures['PI']
        params[CC_name]=params['PI']*(1+cc_scale*delta_t)

# generate CC params

colors_ls=  {'2005-2020':('blue','solid'),'PI+2K':('red','solid'),'2005-2020_CC':('blue','dashed'),'PI+2K_CC':('red','dashed')}
gev=dict()
for k,v in params.items():
    gev[k] = scipy.stats.genextreme(v.sel(parameter='shape'),loc=v.sel(parameter='location'),scale=v.sel(parameter='scale'))

rtn_prd_ticks = [2,5,10,20,50,100,200] # for ticks
rtnp = np.geomspace(rtn_prd_ticks[0],rtn_prd_ticks[-1],200) # return periods
pi_gev = gev.pop('PI')
x = pi_gev.isf(1.0/rtnp) # PI precip at rtn prds.
pi_sf= 1/rtnp
pi_rtn = 1./pi_sf
#gev_rand=dict()
sd_irat=dict()
sd_rrat=dict()
## compute probability  and intensity ratios
pratio=dict()
iratio=dict()
npt=50
rx1hr=np.linspace(10,100,npt)
sf=np.geomspace(1/5,1/250,npt)
pi_params = params_full['PI'].Parameters
PI_sf = xarray_sf(pi_params,x=rx1hr,output_dim_name='Rx1hr')
PI_isf = xarray_isf(pi_params,p=sf,name='Rx1hr')
q=[0.05,0.5,0.95]
mask=(0 < cpm_ht.ht) & (cpm_ht.ht < 1000) # where we want values
for k in ['2005-2020','PI+2K']:
    delta_t = temperatures[k] - temperatures['PI']
    p = params_full[k].Parameters.where(mask)
    cc_name = k+'_CC'
    params_cc = pi_params * (1 + cc_scale * delta_t)

    iratio[k] = (xarray_isf(p,p=sf,name='Rx1hr')/PI_isf).quantile(q,horiz_coords) # intensity ration at fixed p values (= rtn periods)
    iratio[cc_name] = (xarray_isf(params_cc,p=sf,name='Rx1hr')/PI_isf).quantile(q,horiz_coords)

    pratio[k] = (xarray_sf(p, x=rx1hr, output_dim_name='Rx1hr') / PI_sf).quantile(q, horiz_coords) # risk ratio at fixed precip values.
    pratio[cc_name] = (xarray_sf(params_cc, x=rx1hr, output_dim_name='Rx1hr') / PI_sf).quantile(q, horiz_coords) # risk ratio at fixed precip values.

# and convert all intensity ratios  into % increase
for k in iratio.keys():
    iratio[k] = (iratio[k]-1.)*100
    #pratio[k] = (pratio[k] - 1.) * 100

## now let's plot -- first print out correlations when we mask to have topog >0 and < 200

maskf = mask.values.flatten()
for p in xfit[rgn].parameter:
    corr= np.corrcoef(cpm_ht.ht.values.flatten()[maskf],xfit[rgn].Parameters.sel(parameter=p).values.flatten()[maskf])[1,0]
    print(f"{p.values}: {corr:3.2f}")

fig,ax=plt.subplots(nrows=1,ncols=2,num='prob',clear=True,figsize=[11,4])
for k,v in colors_ls.items():
    col, ls =v
    dist = gev[k]
    plot_Args=dict(color=col,linestyle=ls)
    rp = 1.0/iratio[k].probability # rtn prd
    rn = pratio[k].Rx1hr # PI rain
    if not k.endswith('CC'): # plot uncerts.
        ax[0].fill_between(rp, y1=iratio[k].sel(quantile=0.05), y2=iratio[k].sel(quantile=0.95), alpha=0.5,**plot_Args) # intensity ratio
        ax[1].fill_between(rn, y1=pratio[k].sel(quantile=0.05), y2=pratio[k].sel(quantile=0.95), alpha=0.5, **plot_Args)  # risk ratio

    ax[0].plot(rp,iratio[k].sel(quantile=0.5),label=k,**plot_Args)
    ax[1].plot(rn, pratio[k].sel(quantile=0.5), label=k, **plot_Args)
ax[0].legend(ncol=2)

for a, title,xtitle,ytitle in zip(ax, ['Intensity Ratio Increase (%) ', 'Probability Ratio'],
                                  ['PI Return Period (Years)','Rx1hr (mm/hr)'],['% increase','PR']):
    a.set_title(title)
    a.set_xlabel(xtitle)
    a.set_ylabel(ytitle)

fig.tight_layout()
fig.show()
## Plot some helpful figures.
# Scatter of CPM ht vs fraction change in param relative to present_value estimates
# and PM CET vs median extreme precip.
# + obs cet as range.,

cet_summer = cet.resample(time='QS-DEC').mean().dropna('time')
cet_summer = cet_summer.sel(time=(cet_summer.time.dt.month == 6))
cet_summer['time'] = ed_extreme_precip.time  # make sure times are the same
fig,(ax_ht_params,ax_cet)=plt.subplots(nrows=1,ncols=2,clear=True,num='Scatter_ht_D',figsize=[8,5])
mask=(-1 < cpm_ht.ht) & (cpm_ht.ht < 2000) # where we want values
maskf = mask.values.flatten()
for p in ['location','scale']:
    ratio = 100*fit.Parameters.sel(parameter='D'+p)/value_today.sel(parameter=p).where(mask)
    print(f"{p} {float(ratio.mean()):3.2f} {float(ratio.std()):3.2f}")
    ax_ht_params.scatter(cpm_ht.ht.where(mask),ratio,label='%D'+p,s=4)
    corr= np.ma.corrcoef(ratio.stack(horiz=horiz_coords)[maskf],cpm_ht.ht.stack(horiz=horiz_coords)[maskf])
    print(f"Corr D{p} ht {float(corr[0][1]):4.2f}")
ax_ht_params.axhline(7.5,color='black',linestyle='dashed',linewidth=2)
for v in [1,200]:
    ax_ht_params.axvline(v,color='black',linewidth=2,linestyle='dashed')
ax_ht_params.legend()
ax_ht_params.set_ylabel('% Rx1Hr/2005-2020')
ax_ht_params.set_xlabel("Height (m)")
ax_ht_params.set_title("Time dependant parameters vs Height")

# plot the scatter of median (over the masked region) vs CET
ex_precip = ed_extreme_precip.where(mask).max(horiz_coords)
ax_cet.scatter(cet_summer,ex_precip,s=6)
L=(obs_cet.time.dt.month==7) & (obs_cet.time.dt.year > 1849)
L2=(obs_cet.time.dt.month==7) & (obs_cet.time.dt.year > 2005) & (obs_cet.time.dt.year < 2020)
min_cet = obs_cet.sel(time=L).min()
max_cet = obs_cet.sel(time=L).max()

min_cet_today = obs_cet.sel(time=L2).min()
max_cet_today = obs_cet.sel(time=L2).max()
ax_cet.plot([min_cet,t_pi,t_today,t_p2k,max_cet],[ex_precip.quantile([0.1])]*5,color='red',linewidth=4,marker='o',ms=8)
ax_cet.plot([min_cet_today,max_cet_today],[ex_precip.quantile([0.2])]*2,color='blue',linewidth=4)
ax_cet.set_title("CET vs Max Regional Rx1hr")
ax_cet.set_xlabel("CET (C)")
ax_cet.set_ylabel("Max Rx1hr")
fig.tight_layout()
fig.show()
commonLib.saveFig(fig)

