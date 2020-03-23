#!/usr/bin/env python3
"""
	Remove a plane (linear ramp in x,y) from a georeferenced image
"""

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from geoFormatting import GDALtransform
from viewingFunctions import imgBackground, mapPlot


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Remove a plane (linear ramp in x,y) from a georeferenced image')
	# Data sets
	parser.add_argument(dest='mapName', type=str, help='Map data set from which to remove plane')
	parser.add_argument('-bg','--background', dest='background', default=None, help='Background value')
	parser.add_argument('-pctmin','--pctmin', dest='pctmin', type=float, default=0, help='Minimum percent clip')
	parser.add_argument('-pctmax','--pctmax', dest='pctmax', type=float, default=100, help='Maximum percent clip')

	# Outputs
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')

	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name, for difference map and analysis plots')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
## Load georeferenced map data set
def gridData(inpt,DS):
	# Geotransform and spatial parameters
	inpt.M=DS.RasterYSize; inpt.N=DS.RasterXSize
	inpt.T=GDALtransform(DS,verbose=inpt.verbose)

	# Create grid at full resolution in native units
	x=np.linspace(inpt.T.xstart,inpt.T.xend,inpt.N)
	y=np.linspace(inpt.T.ystart,inpt.T.yend,inpt.M)

	X,Y=np.meshgrid(x,y)

	return X,Y


## Save georeferenced data set
def saveMap(inpt,DS,dtrImg):
	# Construct savename
	savename='{}.tif'.format(inpt.outName)

	# GeoTiff
	driver=gdal.GetDriverByName('GTiff')
	DSout=driver.Create(savename,DS.RasterXSize,DS.RasterYSize,1,gdal.GDT_Float32)
	DSout.GetRasterBand(1).WriteArray(dtrImg)
	DSout.SetProjection(DS.GetProjection())
	DSout.SetGeoTransform(DS.GetGeoTransform())
	DSout.FlushCache()



### PLANE FITTING FUNCTIONS ---
## Fit a plane by linear inversion
def linearPlaneFit(inpt,X,Y,Z):
	"""
		Linear inversion for model:
		 ax + by = z
		
		Suit data by centering at x = 0, y = 0 and remove mean
	"""

	## Format data
	# Downsample data
	# dsFactor=int(2**downsampleFactor)

	# Format as 1d arrays
	Xsamp=X.reshape(inpt.M*inpt.N,1)
	Ysamp=Y.reshape(inpt.M*inpt.N,1)
	Zsamp=Z.reshape(inpt.M*inpt.N,1)

	# Discount masked values if applicable
	if inpt.mask is not None:
		Xsamp=Xsamp.compressed()
		Ysamp=Ysamp.compressed()
		Zsamp=Zsamp.compressed()


	## Invert for parameters
	# Length of data arrays
	Lsamp=len(Zsamp)

	# Design matrix
	G=np.hstack([Xsamp.reshape(Lsamp,1),
		Ysamp.reshape(Lsamp,1),
		np.ones((Lsamp,1))])

	# Parameter solution
	beta=np.linalg.inv(np.dot(G.T,G)).dot(G.T).dot(Zsamp)

	# Report if requested
	if inpt.verbose is True:
		print('Solved for plane.')
		print('Fit parameters: {}'.format(beta))


	## Construct plane
	# Full design matrix
	F=np.hstack([X.reshape(inpt.M*inpt.N,1),
		Y.reshape(inpt.M*inpt.N,1),
		np.ones((inpt.M*inpt.N,1))])

	# Plane
	P=F.dot(beta)
	P=P.reshape(inpt.M,inpt.N)

	return P



### MAIN ---
if __name__=='__main__':
	## Gather arguments
	inpt=cmdParser()


	## Load map data set
	# Load gdal data set
	DS=gdal.Open(inpt.mapName,gdal.GA_ReadOnly)
	img=DS.GetRasterBand(1).ReadAsArray()


	## Grid data
	X,Y=gridData(inpt,DS)


	## Mask image
	# Detect and mask background value
	if inpt.background == 'auto':
		inpt.background=imgBackground(img)

	if inpt.background is None:
		inpt.mask=None
	else:
		inpt.mask=(img==inpt.background)
		img=np.ma.array(img,mask=inpt.mask)
		X=np.ma.array(X,mask=inpt.mask)
		Y=np.ma.array(Y,mask=inpt.mask)

	# Report if requested
	if inpt.verbose is True:
		print('Loaded: {}'.format(inpt.mapName))
		if inpt.background is not None:
			print('Background value: {}'.format(inpt.background))

	# Plot original image
	Fig,ax=mapPlot(img,cmap='viridis',pctmin=inpt.pctmin,pctmax=inpt.pctmax,background=None,
		extent=inpt.T.extent,showExtent=True,cbar_orientation='horizontal',
		title='Original map')
	ax.set_xlabel('easting'); ax.set_ylabel('northing')


	## Remove plane
	# Fit plane
	P=linearPlaneFit(inpt,X,Y,img)

	# Remove plane
	dtrImg=img-P

	# Plot plane removed
	mapPlot(dtrImg,cmap='viridis',pctmin=inpt.pctmin,pctmax=inpt.pctmax,background=None,
		extent=inpt.T.extent,showExtent=True,cbar_orientation='horizontal',
		title='Plane removed')


	## Save data set
	if inpt.outName:
		saveMap(inpt,DS,dtrImg)


	plt.show(); exit()