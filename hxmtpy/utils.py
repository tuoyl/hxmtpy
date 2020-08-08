from __future__ import division 
import numpy as np
import numba
from astropy.io import fits

__all__ = ['FileUtils',
        'numba_glitch_filter',
        'numba_histogram',
        'lightcurve_hist',
        'lightcurve']

class FileUtils():

    def __init__(self, infile):
        self.infile = infile

    def filter(self, filter_bool, outfile, extension_num=1, **header_kwargs):
        '''
        Filter the fits file by the bool array
        '''
        hdulist = fits.open(self.infile)
        prim_hdr_old = hdulist[0].header
        prim_hdr_new = fits.PrimaryHDU(header=prim_hdr_old)
        # copy the rest of extensions
        rest_of_ext = [hdulist[i] for i in range(len(hdulist)) if ((i !=0) and (i !=extension_num))]


        # filter the specific extension table
        new_table = []
        table = hdulist[extension_num].data
        column_names = table.names
        column_type = table.formats
        for column_name in column_names:
            new_table.append( table.field(column_name)[filter_bool] )

        ## construct the new table 
        #TODO: main extension header!!
        new_column = [fits.Column(name=column_names[i], array=new_table[i], format=column_type[i]) for i in
                range(len(column_names))]
        tb = fits.BinTableHDU.from_columns(new_column)

        ## write to outfile
        ### save 
        hdul = fits.HDUList([prim_hdr_new, tb] + rest_of_ext)
        hdul.writeto(outfile, overwrite=True)

        ## copy the header of rest of extension
        hdulist_cp = fits.open(outfile)
        print(hdulist_cp[extension_num].header)
        for header_key in hdulist[extension_num].header:
            if header_key in hdulist_cp[extension_num].header:
                continue
            elif (header_key == "HISTORY") or (header_key == "COMMENT"):
                continue
            else:
                hdulist_cp[extension_num].header[header_key] = hdulist[extension_num].header[header_key]
        ## write header_kwargs
        for header_key in header_kwargs:
            hdulist_cp[extension_num].header[header_key] = header_kwargs[header_key]
        hdulist_cp.writeto(outfile, overwrite=True)


    def add_column(self, column_name, clobber=True):
        '''
        Add a column to FITS file
        '''
        hdulist = fits.open(self.infile)
        print(column_name)

@numba.njit
def numba_glitch_filter(arr_events, timedel, evtnum):
    glitch_gti_bool = np.array([True] * len(arr_events))
    for i in range(len(arr_events)-evtnum):
        time_diff = arr_events[i+1:i+evtnum] - arr_events[i:i+evtnum-1] 
        if np.all(time_diff) <= timedel:
            glitch_gti_bool[i:i+evtnum] = False
    return glitch_gti_bool
        
@numba.jit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@numba.jit(nopython=True)
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges

def lightcurve_hist(data, binsize=1, rate=True):
    """
    """
    N = (np.max(data) - np.min(data))/binsize
    lc_y, lc_x = np.histogram(data, bins=np.arange(np.min(data), np.max(data)+binsize, binsize))
    lc_x = lc_x[np.s_[:-1]]
    if rate:
        lc_y = lc_y/binsize
    return lc_x, lc_y

class lightcurve():
    """
    A Class for X-ray Light Curve
    """

    def __init__(self, time, counts, yerr=None):
        """
        initial Parameters
        ---------------------
        time : array-like
            The time series for light curve

        counts : array-like
            The counts for each time intervals
    
        yerr : array-like (optional)
            The error for light curve counts
        """

        self.time = time
        self.counts = counts
        self.yerr = yerr

    def _rebin_onebin(self, bins):

        x = self.time
        y = self.counts
        yerr = self.yerr
        new_x = new_y = np.array([])
        
        if len(yerr) != 0:
            new_yerr = np.array([])
        range_left = bins[0]
        range_right= bins[1]
        step = bins[2]
        for i in np.arange(range_left, range_right, step):
            if i+step <= range_right:
                new_x = np.append(new_x, (x[i]+x[i+step-1])/2)
                new_y = np.append(new_y, np.mean(y[i:i+step]))
                if len(yerr) !=0:
                    new_yerr = np.append(new_yerr, np.sqrt(np.sum(yerr[i:i+step]**2))/step)
            else:
                new_x = np.append(new_x, (x[i]+x[-1])/2)
                new_y = np.append(new_y, np.mean(y[i:]))
                if len(yerr) !=0:
                    new_yerr = np.append(new_yerr, np.sqrt(np.sum(yerr[i:]**2))/step)
        if len(yerr) !=0:
            return new_x, new_y, new_yerr
        else:
            return new_x, new_y

    def rebin(self, bins):
        """
        Rebining the lightcurve based on the grppha syntax.

        - rebined time is the middle point of each binning time intervals
        - rebined counts is the mean value of counts in binning time intervals
        - rebined error of counts if the error of propagation of counts error 

        Parameters
        ---------------
        bins : n*3 array-like
           The grppha syntax for rebining the light curve,
           e.g. bins = np.array([[0,10,2], [10,50,4]]) 
           rebin the 1st interval to the 10th interval every 2 bins
           and then rebining the 10th interval to 51st intervals every 4 bins.

        Returns
        ---------------
        new_x : array-like
            The new time array of rebined light curve

        new_y : array-like
            The new counts of rebined light curve 

        new_yerr : array-like (optional)
            The new error of counts for rebined light curve
        """

        x = self.time
        y = self.counts
        yerr = self.yerr
        new_x, new_y = np.array([]), np.array([])
        sizeofbins = np.size(bins)/3

        if len(yerr) !=0:
            new_yerr = np.array([])
        if sizeofbins == 1:
            if len(yerr) !=0:
                new_x, new_y, new_yerr = self._rebin_onebin(bins=bins)
            else:
                new_x, new_y = self._rebin_onebin(bins=bins)
        if sizeofbins > 1:
            for onebin in bins:
                if len(yerr) !=0:
                    new_xtmp, new_ytmp, new_yerrtmp = self._rebin_onebin(bins=onebin)
                    new_x = np.append(new_x, new_xtmp)
                    new_y = np.append(new_y, new_ytmp)
                    new_yerr = np.append(new_yerr, new_yerrtmp)
                else:
                    new_xtmp, new_ytmp = self._rebin_onebin(bins=onebin)
                    new_x = np.append(new_x, new_xtmp)
                    new_y = np.append(new_y, new_ytmp)
        if len(yerr) !=0:
            return new_x, new_y, new_yerr
        else:
            return new_x, new_y


if __name__ == "__main__":
    x = y = z = np.arange(1,100, 1)
    lc = lightcurve(x, y, z)
    print(len(np.array([0,99,1])))
    x, y, z = lc.rebin(bins=np.array([0,99,1]))
    print(x, y , z)
    x, y, z = lc.rebin(bins=np.array([[0,30,1],[30,99,5]]))
    print(x, y , z)




