"""
Class to support empirical distributions
"""

# statsmodels.distributions.empirical_distribution
import scipy.interpolate
import numpy as np

class empDist(object):
    """
    Class for empDist. See statsmodels.distributions.empirical_distribution.ECDF for more discrete version
    """

    def __init__(self, data,copy=True):
        sort_data, count = np.unique(data, return_counts=True) # extract the unique values
        cdf_values = (np.cumsum(count)-1) / (data.size) # and work out the cdf
        #cdf_values = np.r_[cdf_values, 1.0]
        #sort_data = np.r_[sort_data, np.inf]

        self._sort_data = sort_data
        self._cdf_values = cdf_values
        self._cdf_fn = scipy.interpolate.interp1d(sort_data, cdf_values, fill_value=(0, 1.), kind='linear',bounds_error=False)

        self._inv_cdf_fn = scipy.interpolate.interp1d(cdf_values, sort_data, fill_value=(sort_data[0], sort_data[-1]),
                                                      kind='linear',bounds_error=False)


    def cdf(self, values):
        """
        Compute the cdf for specified values
        :param values: values wanted
        :return: cdf
        """

        return self._cdf_fn(values)

    def sf(self, values):
        """
        Compute the survival function (1-cdf) for specified values
        :param values: values wanted
        :return: survival function values -- computed using cdf.
        """

        return 1.0-self._cdf_fn(values)

    def ppf(self, cdf):
        """
        Compute the percent point function (inverse cdf) for specified cdf values
        :param cdf: cdf values wanted
        :return: values of empirical distribution at that value
        """
        # check all values in range(0,1)
        cdf_i=np.array(cdf) # v fast if already a numpy array
        if np.any((cdf_i < 0.0) | (cdf_i > 1)):
            raise ValueError("CDF values outside (0,1)")
        return self._inv_cdf_fn(cdf_i)

    def isf(self,isf):
        """
        Compute the inverse survival function
        :param isf: values wanted
        :return: inverse of the survivor fn using ppf
        """
        return self.ppf(1.0-isf)

    def median(self):
        return self.icdf(0.5)
    def mean(self):
        return self._sort_data[0:-1].mean()

    def sup(self):
        return (self._sort_data[0],self._sort_data[-2])

    def scale(self, scale_fn):
        """

        :param scale_fn: function which given cdf values produces a scaling
        :return: a new empDist constructed by scaling the original data with scale fn
        """
        data = np.copy(self._sort_data)#[0:-1]) # copy the data
        cdf_v = self._cdf_values#[0:-1]
        scales = scale_fn(cdf_v)
        data *= scales
        result = empDist(data,copy=False)
        return result