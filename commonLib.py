"""
Provides classes, variables and functions used across the project
"""
import pathlib
import fiona
import cartopy
import iris
import tarfile
import gzip
import tempfile
import shutil
import xarray
import pathlib
import cartopy
import cartopy.feature 
import pandas as pd
import numpy as np
dataDir = pathlib.Path(r'c:\users\stett2\data\Edinburgh_rain')
# get in the UK country borders.
#file = pathlib.Path('Countries_(December_2017)_Boundaries/Countries_(December_2017)_Boundaries.shp')
#shp  = cartopy.io.shapereader.Reader(file)
#UK_nations = cartopy.feature.ShapelyFeature(shp.geometries(),crs=cartopy.crs.OSGB(),
#                                            edgecolor='black',facecolor='none')
fig_dir = pathlib.Path("figures")
def time_convert(DataArray,ref='1970-01-01',unit='h',set_attrs=True):
    """
    convert times to hours (etc) since reference time.
    :param DataAray -- dataArray values to be converted
    :param ref -- reference time as ISO string. Default is 1970-01-01
    :param unit -- unit default is h for hours
    :return -- returns dataarray with units reset and values converted
    """
    name_conversion=dict(h='hours',d='days')
    
    with xarray.set_options(keep_attrs=True):
        hour = (DataArray - np.datetime64(ref))/np.timedelta64(1,unit)
    u = name_conversion.get(unit,unit)
    if set_attrs:
        hour.attrs['units']=f'{u} since {ref}'
    return hour


def extract_nimrod_day(file,region=None,QCmax=None):
    """
    extract rainfall data from nimrod badc archive. 
    Archive is stored as a compressed tarfil of gzipped files. 
    Algorithm opens the tarfile. Iterates through files in tarfile 
    uncompresses each file to tempfile. 
    Reads tempfile then deletes it when done.
    returns an dataset of rainfall for the whole day. Note badc archive 
    seems to be missing data so some days will not be complete. 

    :param file -- pathlib path to file for data to be extracted
    :param region (default None) -- if not None then shoul be a dict of co-ords to be extacted.
    :param QCmax (default None) -- if not None then values > QCmax are set missing as crude QC.

    :example rain=extract_nimrod_day(path_to_file,
                region = dict(projection_x_coordinate=slice(5e4,5e5),
                projection_y_coordinate=slice(5e5,1.1e6)),QCmax=400.)
    """
    rain_15min=[]
    with tarfile.open(file) as tar:
        # iterate over members uncompressing them
        for tmember in tar.getmembers():
            with tar.extractfile(tmember) as fp:
                f_out=tempfile.NamedTemporaryFile(delete=False)
                fname = f_out.name
                with gzip.GzipFile("somefilename",fileobj=fp) as f_in:
                    # uncompress the data writing to the tempfile
                    shutil.copyfileobj(f_in,f_out) # 
                    f_out.close()
                    cube = iris.load_cube(fname)
                    try: # deal with bad data by ignoring it! 
                        # doing various transforms to the cube here rather than all at once. 
                        # cubes are quite large so worth doing. 
                        da=xarray.DataArray.from_iris(cube) # read data
                        if region is not None:
                            da = da.sel(**region) # extract if requested
                        if QCmax is not None:
                            da = da.where(da <= QCmax) # set missing all values > QCmax
                        # sort out the attributes
                        da = da.assign_attrs(units=cube.units,**cube.attributes,BADCsource='BADC nimrod data')
                        rain_15min.append(da) # add to the list
                    except (ValueError,TypeError):
                        print(f"bad data in {tmember}")

                    pathlib.Path(fname).unlink() # remove the temp file.
                    
            # end loop over members          (every 15 mins)
    # end dealing with tarfile -- which will close the tar file.
    rain_15min = xarray.combine_nested(rain_15min,'time',combine_attrs='drop_conflicts') # merge list of datasets
    rain_15min=rain_15min.sortby('time')
    # make units a string so it can be saved.
    rain_15min.attrs['units']=str(rain_15min.attrs['units'])
        
    return rain_15min


def saveFig(fig, name=None, savedir=None, figtype=None, dpi=None, verbose=False):
    """


    :param fig -- figure to save
    :param name (optional) set to None if undefined
    :param savedir (optional) directory as a pathlib.Path. Path to save figure to. Default is fig_dir
    :param figtype (optional) type of figure. (If not specified then png will be used)
    :param dpi: dots per inch to save at. Default is none which uses matplotlib default.
    :param verbose:  If True (default False) printout name of file being written to
    """

    defFigType = '.png'
    if dpi is None:
        dpi = 300
    # set up defaults
    if figtype is None:
        figtype = defFigType
    # work out sub_plot_name.
    if name is None:
        fig_name = fig.get_label()
    else:
        fig_name = name

    if savedir is None:
        savedir = fig_dir

    # try and create savedir
    # possibly create the fig_dir.
    savedir.mkdir(parents=True, exist_ok=True)  # create the directory

    outFileName = savedir / (fig_name + figtype)
    if verbose:
        print(f"Saving to {outFileName}")
    fig.savefig(outFileName, dpi=dpi)

    ##

class plotLabel:
    """
    Class for plotting labels on sub-plots
    """

    def __init__(self, upper=False, roman=False,fontdict={}):
        """
        Make instance of plotLabel class
        parameters:
        :param upper -- labels in upper case if True
        :param roman -- labels use roman numbers if True
        """

        import string
        if roman:  # roman numerals
            strings = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii']
        else:
            strings = [x for x in string.ascii_lowercase]

        if upper:  # upper case if requested
            strings = [x.upper() for x in strings]

        self.strings = strings[:]
        self.num = 0
        self.fontdict=fontdict

    def label_str(self):
        """
        Return the next label
        """
        string = self.strings[self.num] + " )"
        self.num += 1
        self.num = self.num % len(self.strings)
        return string

    def plot(self, ax=None, where=None):
        """
        Plot the label on the current axis.
        :param ax -- axis to plot on. Default is current axis (using plt.gca())
        :param where -- (x,y) tuple saying where  to plot label using axis coords. Default is (-0.03,1.03)
        """

        if ax is None:
            plt_axis = plt.gca()
        else:
            plt_axis = ax
        try:
            if plt_axis.size > 1:  # got more than one element
                for a in plt_axis.flatten():
                    self.plot(ax=a, where=where)
                return
        except AttributeError:
            pass

        # now go and do the actual work!

        text = self.label_str()
        if where is None:
            x = -0.03
            y = 1.03
        else:
            (x, y) = where

        plt_axis.text(x, y, text, transform=plt_axis.transAxes,
                      horizontalalignment='right', verticalalignment='bottom',fontdict=self.fontdict)


## standard stuff for plots. 


# UK local authorities.
regions = cartopy.feature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')

# UK nations (and other national sub-units)
nations = cartopy.feature.NaturalEarthFeature(
    category='cultural',name='admin_0_map_subunits',
    scale='10m',facecolor='none',edgecolor='black')

# radar stations
metadata = pd.read_excel('radar_station_metadata.xlsx',index_col=[0],na_values=['-']).T
L=metadata.Working.str.upper() == 'Y'
metadata = metadata[L]

def std_decorators(ax):
    """
    Add a bunch of stuff to an axis
    :param ax: axis
    :return: Nada
    """


    ax.coastlines(resolution='10m')
    #ax.plot(-3.19,55.96,marker='*',color='black', alpha=0.5,ms=12,transform=cartopy.crs.PlateCarree()) # edinburgh
    ax.plot(metadata.Easting,metadata.Northing,marker='h',color='red',ms=9,linestyle='none') #  radar stations location.
    #ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)
    ax.add_feature(regions, edgecolor='red')
    ax.add_feature(nations, edgecolor='black')
    

