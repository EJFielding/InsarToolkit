#!/usr/bin/env python3

### IMPORT MODULES ---
import os
import argparse
from glob import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


### PARSER ---
def createParser():
    parser = argparse.ArgumentParser( description='Create a map of frame centers based on existing ARIA products.')
    parser.add_argument(dest='imgfile', type=str,
            help='File search string.')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args = iargs)



### ANCILLARY FUNCTIONS ---
class ARIAname:
    def __init__(self,fname):
        '''
            Parse ARIA filename.
        '''
        # Basename
        self.name = os.path.basename(fname)

        # Parse name info
        self.__parseNameInfo__()

    def __parseNameInfo__(self):
        '''
            Parse ARIA filename.
        '''
        # Split name string
        nameinfo = self.name.split('-')

        # Track number
        self.track = nameinfo[4]

        # Dates
        dates = nameinfo[6].split('_')
        self.masterDate = datetime.strptime(dates[0],'%Y%m%d')
        self.slaveDate = datetime.strptime(dates[1],'%Y%m%d')

        # Coordinates
        coords = nameinfo[8].split('_')
        coords = [coord.strip('N') for coord in coords]
        coords = ['{}.{}'.format(coord[:2],coord[2:]) for coord in coords]
        coords = [float(coord) for coord in coords]
        self.edge1 = coords[0]
        self.edge2 = coords[1]
        self.center = (self.edge1+self.edge2)/2


def plotAcquisitions(ariaNames):
    '''
        Plot frame centers as a function of time.
    '''

    # Create lists of dates and frame centers
    frames = []
    dateList = []
    for ariaName in ariaNames:
        frame = ARIAname(ariaName)
        frames.append(frame)
        dateList.append(frame.masterDate)
        dateList.append(frame.slaveDate)

    dateList = list(set(dateList))
    dateList.sort()
    dateLabels = [date.strftime('%Y-%m-%d') for date in dateList]

    # Establish figure
    Fig = plt.figure()
    axMaster = Fig.add_subplot(211)
    axSlave = Fig.add_subplot(212)

    # Plot frame centers
    for frame in frames:
        axMaster.plot(frame.masterDate,frame.center,'ko')
        axSlave.plot(frame.slaveDate,frame.center,'ko')

    # Finish plot
    axMaster.set_ylabel('Latitude')
    axMaster.set_xticks(dateList)
    axMaster.set_xticklabels([])

    axSlave.set_ylabel('Latitude')
    axSlave.set_xlabel('Date')
    axSlave.set_xticks(dateList)
    axSlave.set_xticklabels(dateLabels,rotation=80)

    Fig.tight_layout()



### MAIN ---
if __name__ == '__main__':
    inps = cmdLineParse()

    # Discover file names
    fnames = glob(inps.imgfile)

    # Plot frame centers
    plotAcquisitions(fnames)


    plt.show()