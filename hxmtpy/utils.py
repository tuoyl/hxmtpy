from __future__ import division 
import numpy as np

__all__ = ['lightcurve_hist']

def lightcurve_hist(data, binsize=1, rate=True):
    """
    """
    N = (np.max(data) - np.min(data))/binsize
    print(N)
    lc_y, lc_x = np.histogram(data, bins=np.arange(np.min(data), np.max(data)+binsize, binsize))
    lc_x = lc_x[np.s_[:-1]]
    if rate:
        lc_y = lc_y/binsize
    return lc_x, lc_y
