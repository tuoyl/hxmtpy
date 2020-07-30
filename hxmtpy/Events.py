from __future__ import absolute_import, division
import numpy as np
import numba
#from utils import numba_histogram, numba_bad_intervals_filter
from utils import numba_histogram, numba_glitch_filter
import matplotlib.pyplot as plt

class events():

    def __init__(self, arr_events):
        if arr_events.dtype == ">f8":
            self.events = numba.float64(arr_events) # transfer big-endian dtype to non big-endian dtype
        else:
            self.events = arr_events

    def glitch_gti_filter(self, **kwargs):
        arr_events = self.events
        glitch_gti_arr = np.array([True] * len(arr_events))
        if 'timedel' in kwargs:
            # filter glitch events by time intervals
            timedel = kwargs["timedel"]
            evtnum = kwargs["evtnum"]

            starttime = timeit.default_timer()
            glitch_gti_arr = np.logical_and(glitch_gti_arr, 
                    numba_glitch_filter(arr_events, timedel, evtnum))
            print("time of logical and: ",timeit.default_timer() - starttime)

            

        if ('lowchan' in kwargs) and ('highchan' in kwargs):
            # filter glitch events by channel
            lowchan = kwargs['lowchan']
            highchan = kwargs['highchan']
            if not hasattr(self, 'channel'):
                raise IOError("channel data deos not loaded, please use self.channel to load channel data")
            else:
                channel = self.channel
            channel_gti_arr = (channel >= lowchan) & (channel <= highchan)
            glitch_gti_arr = np.logical_and(glitch_gti_arr,
                    channel_gti_arr)

        if ('minpulsewidth' in kwargs) and ('maxpulsewidth' in kwargs):
            minpulsewidth = kwargs['minpulsewidth']
            maxpulsewidth = kwargs['maxpulsewidth']
            # filter events by pulse width
            if not hasattr(self, "pulse_width"):
                raise IOError("pulse_width data does not loaded, please use self.pulse_width to load pulse_width data")
            else:
                pulse_width = self.pulse_width
            pulsewidth_gti_arr = (pulse_width >= minpulsewidth) & (pulse_width <= maxpulsewidth)
            glitch_gti_arr = np.logical_and(glitch_gti_arr,
                    pulsewidth_gti_arr)

        print("DONE glitch filtering, return the bool array")
        return glitch_gti_arr









        
if __name__ == "__main__":
    
    import timeit
    from astropy.io import fits
    #hdulist = fits.open("../../SGR1935/RAWDATA/HE/HXMT_P020402500803_HE-Evt_FFFFFF_V1_L1P.FITS")
    starttime = timeit.default_timer()

    hdulist = fits.open("./test/HXMT_P020402500803_HE-Evt_FFFFFF_V1_L1P.FITS")
    time = hdulist[1].data.field("Time")
    he = events(time)
    he.channel = hdulist[1].data.field("Channel")
    he.pulse_width = hdulist[1].data.field("Pulse_Width")
    #he.glitch_gti_filter(timedel=0.000080, evtnum=5)
    #he.glitch_gti_filter(lowchan=0, highchan=255)
    glitch_gti_bool = he.glitch_gti_filter(lowchan=0, highchan=255, timedel=0.00008, evtnum=5, minpulsewidth=54, maxpulsewidth=70)
    print("Enetries of selected Events is : ", len(time[glitch_gti_bool]))

    print("The time difference is :", timeit.default_timer() - starttime)
