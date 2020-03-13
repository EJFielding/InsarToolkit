#!/usr/bin/env python3
"""
	Plot vectors according to the gradient given by a three-band image. 
	 Optimally works with the xyz_pointing_vectors output by 
	 ComputeGradients.py
"""

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
# InsarToolkit modules
from viewingFunctions import mapPlot
from geoFormatting import GDALtransform
from slopeFunctions import *


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Given a rasterized DEM in Cartesian coordinates (e.g., UTM), compute the slope and slope-aspect maps. The DEM should be provided as a gdal-readable file, preferably in GeoTiff format. Alternatively, provide the slope and slope aspect maps.')

	# Input data
	parser.add_argument(dest='DEMname', type=str, help='Name of DEM in Cartesian coordinates.')


	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)