#!/usr/bin/env python3
'''
Extract a 1D profile from an image.
Images should ideally be in GDAL georeferenced format.
'''

### IMPORT MODULES ---
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal


### PARSER ---
def createParser():
    Description = '''Extract a 1D profile from an image.
Designed for use with georeferenced images encoded in GDAL format.
'''

    parser=argparse.ArgumentParser(description = Description,
        formatter_class = argparse.RawTextHelpFormatter)

    # Input arguments
    inputArgs = parser.add_argument_group('INPUT ARGUMENTS')
    inputArgs.add_argument(dest='imgFile', type=str, 
        help='File to plot')
    inputArgs.add_argument('-t','--image-type', dest='imgType', type=str, default='auto',
        help='Image type ([auto], ISCE/complex)')
    inputArgs.add_argument('-b','--band', dest='imgBand', type=int, default=1,
        help='Image band')
    inputArgs.add_argument('-n','--nodata', dest='noData', nargs='+', default=[],
        help='No data values')

    # Display arguments
    displayArgs = parser.add_argument_group('DISPLAY ARGUMENTS')
    displayArgs.add_argument('-c','--cmap', dest='cmap', type=str, default='viridis',
        help='Colormap')
    displayArgs.add_argument('-co','--cbar-orient', dest='cOrient', type=str,
        default='horizontal', help='Colorbar orientation ([horizontal], vertical')
    displayArgs.add_argument('-vmin','--vmin', dest='vmin', type=float, default=None,
        help='Minimum value to plot (overridden by pctmin)')
    displayArgs.add_argument('-vmax','--vmax', dest='vmax', type=float, default=None,
        help='Maximum value to plot (overridden by pctmax)')
    displayArgs.add_argument('-pctmin','--percent-min', dest='pctmin', type=float, 
        default=None, help='Minimum percent clip')
    displayArgs.add_argument('-pctmax','--percent-max', dest='pctmax', type=float,
        default=None, help='Maximum percent clip')

    # Output arugments
    outputArgs = parser.add_argument_group('OUTPUT ARGUMENTS')
    outputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true', 
        help='Verbose mode')

    return parser

def cmdParser(iargs = None):
    parser = createParser()

    return parser.parse_args(args = iargs)



### MAIN ---
if __name__ == '__main__':
    # Gather arguments
    inps = cmdParser()

    print('Don\t waste too much time')