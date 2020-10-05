#!/usr/bin/env python3

### IMPORT MODULES ---
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from osgeo import gdal
from kite import Scene
import utm


### PARSER ---
def createParser():
    parser = argparse.ArgumentParser(description='Apply Kite\'s quadtree sampling approach (Jonsson, 2002)')

    # Required inputs
    parser.add_argument(dest='kiteScene', type=str, help='Kite scene')


    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args = iargs)



### FUNCTIONS ---
def parseCoords(sc):
    '''
        Compute reference point in LaLo and UTM from Kite Scene.
        Return the leaf focal points in LaLo and UTM (km).
    '''
    # Reference coords lat/lon
    refLat = sc.frame.llLat
    refLon = sc.frame.llLon

    # Convert to UTM
    refUTM = utm.from_latlon(refLat,refLon)

    # Print reference points
    print('Reference point (lalo): {:.4f} N, {:.4f} E'.format(refLat,refLon))
    print('Reference point (utm): {:.4f} E, {:.4f} N'.format(refUTM[0],refUTM[1]))

    # XY positions of the reference points in lon/lat
    xyLaLo = np.column_stack([qt.leaf_focal_points[:,0]+refLon,
        qt.leaf_focal_points[:,1]+refLat])

    xyUTM = utm.from_latlon(xyLL[:,1],xyLL[:,0])

    xyPos = xyUTM/1000 # m to km

    return xyLaLo, xyPos



### MAIN ---
if __name__ == '__main__':
    # Gather inputs
    inps = cmdParser()


    ## Load scene
    sc = Scene.load(inps.kiteScene)
    qt = sc.quadtree

    print('Loaded Kite scene: {:s}'.format(inps.kiteScene))


    ## Parse coordinates
    xyLaLo, xyPos = parseCoords(sc)