"""
Provides classes, variables and functions used across the project
"""
import pathlib
fig_dir = pathlib.Path("figures")
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

