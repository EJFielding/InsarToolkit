#!/usr/bin/env python3

### IMPORT MODULES ---
import os
import argparse
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from nameFormatting import ARIAname


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
def plotAcquisitions():
    '''
        Plot frame centers as a function of time.
    '''
    # Establish figure
    Fig = plt.figure()
    ax = Fig.add_subplot(111)



### MAIN ---
if __name__ == '__main__':
    inps = cmdLineParse()
