#!/usr/bin/env python
import numpy as np

class Calibration(object):

    def __init__(self, evtfile, outfile):
        self.evtfile = evtfile
        self.outfile = outfile

    def info(self):
        print("evtfile is %s"% self.evtfile)
        print("outfile is %s"% self.outfile)

class HECalibration(Calibration):
    def __init__(self, evtfile, outfile, **kwarg):
        Calibration.__init__(self, evtfile, outfile)
        if "instrument" in kwarg:
            self.instrument = kwarg["instrument"]

class MECalibration(Calibration):
    def __init__(self, *args):
        super().__init__(evtfile, outfile)

class LECalibration(Calibration):
    def __init__(self, *args):
        super().__init__(evtfile, outfile)

if __name__ == "__main__":
    cal1 = Calibration("in", "out")
    hecal1 = HECalibration("in", "out")
    print(hecal1.evtfile)
    hecal1.instrument = "HE"
    print(hecal1.instrument)
    hecal1.info()


