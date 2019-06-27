#!/usr/bin/env python3 

import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal 


def demSection(DEMname,xmin,xmax,ymin,ymax): 
	pass 


def DEMmatch(DEMname,bounds,xRes,yRes,outputFormat='VRT'): 
	DEM=gdal.Open(DEMname,gdal.GA_ReadOnly) 
	dem=gdal.Warp('', DEM, options=gdal.WarpOptions(format=outputFormat, outputBounds=bounds, outputType=gdal.GDT_Int16, xRes=xRes, yRes=yRes, dstNodata=-32768.0, srcNodata=-32768.0))
	gdal.Translate('matched_dem.vrt', dem, options=gdal.TranslateOptions(format='VRT')) #Make VRT
	return dem 
