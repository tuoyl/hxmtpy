from __future__ import absolute_import 
import numpy as np
from hxmtpy.tools.utils import lightcurve_hist

if __name__ == "__main__":
    from astropy.io import fits
    hdulist = fits.open("../../SGR1935/RAWDATA/HE/HXMT_P020402500803_HE-Evt_FFFFFF_V1_L1P.FITS")
    time = hdulist[1].data.field("TIME")
    x, y = lightcurve_hist(time, binsize=1)
    print(x, y)
