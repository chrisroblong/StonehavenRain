"""
SW to gev fit with R in python.
"""

import xarray
import scipy.stats
import pandas as pd
import os
import numpy as np
import collections

os.environ[
    'R_HOME'] = r'C:\Users\stett2\AppData\Local\Microsoft\AppV\Client\Integration\FC689017-A9BB-4A9B-B971-6AC52117BA03\Root'  # where R is...
# will need adjusting depending where R got installed
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.pandas2ri as rpandas2ri

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
# R package names
packnames = (['extRemes'])  # list of packages to install.
# From example rpy2 install what needs to be installed.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(robjects.vectors.StrVector(names_to_install))
for package in packnames:
    rpackages.importr(package)  # so available
gevFit = collections.namedtuple('gevFit', ['params', 'std_err', 'AIC', 'negloglike'])


def gev_fit(x, cov=None, returnType='named', shapeCov=False, **kwargs):
    """
    Do GEV fit using R and return named tuple of relevant values.
    :param x: Data to be fit
    :param cov: covariate value (if None then not used)
    :param returnType. Type to return, allowed is:
        named -- return a named tuple (default)
        tuple -- return a tuple -- useful for apply_ufunc
        DataSet -- return a DataSet
    :param shapeCov -- If True allow the shape to vary with the covariate.
    :return:
    """
    allowed_types={'DataSet','tuple','named'}
    if returnType not in allowed_types:
        raise ValueError(f"returnType = {returnType} not supported. Should be one of {allowed_types}")
    L = ~np.isnan(x)
    df_data = [x[L]]  # remove nan from data]
    cols = ['x']
    r_code = 'fevd(x=x,data=df'
    if cov is not None:
        df_data.append(cov[L])  # remove places where x was nan from cov.
        cols.append('cov')
        r_code = r_code + ',location.fun=~cov,scale.fun=~cov'
        if shapeCov:
            r_code += 'shape.fun=~cov'
    r_code += ')'  # add on the trailing bracket.
    r_code_summ = 'summary(' + r_code + ',silent=TRUE)'
    df = pd.DataFrame(np.array(df_data).T, columns=cols)
    with rpy2.robjects.conversion.localconverter(robjects.default_converter + rpandas2ri.converter):
        robjects.globalenv['df'] = df  # push the dataframe with info into R
    try:
        fit = robjects.r(r_code_summ)  # do the fit
        # extract the data
        if cov is not None:
            params = np.array(fit.rx2('par'))
            se = fit.rx2('se.theta')
        else:
            params = np.zeros(5)
            se=np.zeros(5)
            params[0::2]=np.array(fit.rx2('par'))
            se[0::2]=np.array(fit.rx2('se.theta'))
        if isinstance(se,rpy2.rinterface_lib.sexp.NULLType):
            # std err not present (for some reason) set everything to nan
            se[:] = np.nan
        if shapeCov: # shape covaries.
            params[-2:] *= -1  # negate two shape parameters as python convention differs from R
        else:
            params[-1] *= -1 # only got one.
            params = np.append(params, 0.0)  # Dshape is 0.
            se = np.append(se, 0.0)
        nllh = np.array(fit.rx2('nllh'))
        aic = np.array(fit.rx2('AIC'))
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        # some problem in R with the fit. Set everything to nan.
        params = np.repeat(np.nan,6)
        se=params
        nllh=np.array([np.nan])
        aic = nllh


    if returnType == 'named':
        result = gevFit(params=params, std_err=se, negloglike=nllh, AIC=aic)
    elif returnType == 'tuple':
        result = (params,se,nllh,aic)
    elif returnType == 'DataSet': # return a dataset
        pnames = ['location', 'Dlocation', 'scale', 'Dscale', 'shape', 'Dshape']
        result=xarray.Dataset(dict(Parameters=('parameter', params),
                                 StdErr=('parameter', se),
                                 nll=float(nllh),
                                 AIC=float(aic))).assign_coords(parameter=pnames)
    else:
        raise ValueError(f"Unknown  returnType={returnType}")

    return result


def xarray_gev(ds, cov=None, shapeCov=False, dim='time_ensemble', file=None, recreate_fit=False, verbose=False,**kwargs):
    """
    Fit a GEV to xarray data.
    :param ds: dataset for which GEV is to be fit
    :param cov: covariate (If None not used)
    :param shapeCov: If True allow the shape to vary with the covariate.
    :param file -- if defined save fit to this file. If file exists then read data from it and so not actually do fit.
    :param recreate_fit -- even if True even if file exists compute fit.
    :param verbose -- be verbose if True
    :param kwargs: any kwargs passed through to the fitting function
    :return: a dataset containing:
        Parameters -- the parameters of the fit; location, location wrt cov, scale, scale wrt cov, shape, shape wrt cov
        Stderr -- the standard error of the fit -- same parameters as Parameters
        nll -- negative log likelihood of the fit -- measure of the quality of the fit
        AIC -- aitkin information criteria.
    """
    if (file is not None) and file.exists() and (not recreate_fit): # got a file specified, it exists and we are not recreating fit
        ds=xarray.load_dataset(file) # just load the dataset and return it
        if verbose:
            print(f"Loaded existing data from {file}")
        return ds

    kwargs['returnType'] = 'tuple'
    kwargs['shapeCov'] = shapeCov
    if cov is None:
        params, std_err, nll, AIC = xarray.apply_ufunc(gev_fit, ds,  input_core_dims=[[dim]],
                                                   output_core_dims=[['parameter'], ['parameter'], ['NegLog'], ['AIC']],
                                                   vectorize=True, kwargs=kwargs)
    else:
        params, std_err, nll, AIC = xarray.apply_ufunc(gev_fit, ds, cov, input_core_dims=[[dim], [dim]],
                                                   output_core_dims=[['parameter'], ['parameter'], ['NegLog'], ['AIC']],
                                                   vectorize=True, kwargs=kwargs)

    pnames = ['location', 'Dlocation', 'scale', 'Dscale', 'shape', 'Dshape']
    # name variables and then combine into one dataset.

    params = params.rename("Parameters")
    std_err = std_err.rename("StdErr")
    nll = nll.rename('nll').squeeze()
    AIC = AIC.rename('AIC').squeeze()
    ds = xarray.Dataset(dict(Parameters=params, StdErr=std_err, nll=nll, AIC=AIC)).assign_coords(parameter=pnames)
    if file is not None:
        ds.to_netcdf(file) # save the dataset.
        if verbose:
            print(f"Wrote fit information to {file}")
    return ds


## use apply ufunc to generate distributions...

def fn_isf(c, loc, scale, p=None, dist=scipy.stats.genextreme):
    fdist = dist(c, loc=loc, scale=scale)
    x = fdist.isf(p)  # values for 1-cdf.
    return x


def fn_sf(c, loc, scale, x=None, dist=scipy.stats.genextreme):
    fdist = dist(c, loc=loc, scale=scale)
    p = fdist.sf(x)  # 1-cdf for given x
    return p


def xarray_sf(params, output_dim_name='value', **kwargs):
    """
    Compute the survival value for different values based on dataframe of fit parameters.
    :param params: xarray dataarray of shape, location and scale values
    :param output_dim_name: name of output dimension. Defualt is "value" but set it to what ever you are using. E.g "Rx1hr"
    :param kwargs: passed to fn_sf which does the computation. Must contain x which is used for the computation.
    :return:
    """
    sf = xarray.apply_ufunc(fn_sf, params.sel(parameter='shape'), params.sel(parameter='location'),
                            params.sel(parameter='scale'),
                            output_core_dims=[[output_dim_name]],
                            vectorize=True, kwargs=kwargs)
    sf = sf.assign_coords({output_dim_name: kwargs['x']}).rename('sf')

    return sf


def xarray_isf(params, name='value', **kwargs):
    """
    Compute the inverse survival function for given values of sf.
    :param params:
    :param name:
    :param kwargs:
    :return:
    """
    output_dim_name = 'probability'
    x = xarray.apply_ufunc(fn_isf, params.sel(parameter='shape'), params.sel(parameter='location'),
                           params.sel(parameter='scale'),
                           output_core_dims=[[output_dim_name]],
                           vectorize=True, kwargs=kwargs)
    x = x.assign_coords({output_dim_name: kwargs['p']})  # .rename(name)
    return x

def param_cov(params,cov):
    p = ['location', 'scale', 'shape']
    p2 = ["D" + a.lower() for a in p]
    params_c=params.Parameters.sel(parameter=p2).assign_coords(parameter=p) * cov + params.Parameters.sel(parameter=p)
    return params_c
