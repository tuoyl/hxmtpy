import numpy as np

class Calibration:

    def __init__(self, evtfile):
        self.evtfile = evtfile
        self.outfile = outfile

class HECalibration(Calibration):
    def __init__(self, *args):
        Calibration.__init__()

class MECalibration(Calibration):
    def __init__(self, *args):
        super().__init__(evtfile, outfile)

class LECalibration(Calibration):
    def __init__(self, *args):
        super().__init__(evtfile, outfile)

if __name__ == "__main__":
    hecal1 = HECalibration("in", "out")
    print(hecal1)
