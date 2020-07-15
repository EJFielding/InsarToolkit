#!/usr/bin/env python3
'''
Display Relax outputs simply by pointing to the folder.
'''

### IMPORT MODULES ---
import argparse
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal


### PARSER ---
Description = '''Plot outputs of Relax model.
'''

Examples = ''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)
    parser.add_argument(dest='fldr', type=str,
        help='Folder with Relax outputs')
    return parser

def cmdParser(inpt_args=None):
    parser = createParser()
    return parser.parse_args(args=inpt_args)



### ANCILLARY FUNCTIONS ---
def detectData(fldr):
    # Construct search strings
    eastSchStr = os.path.join(fldr,'*-east.grd')
    northSchStr = os.path.join(fldr,'*-north.grd')
    upSchStr = os.path.join(fldr,'*-up.grd')

    # Find all grid files in folder
    eastFiles = glob(eastSchStr); eastFiles.sort()
    northFiles = glob

def loadData(fldr):
    pass



### MAIN FUNCTION ---
if __name__ == '__main__':
    # Gather inputs
    inps = cmdParser()

    # Load Relax data
    loadData(inps.fldr)