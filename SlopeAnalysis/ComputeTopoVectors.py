#!/usr/bin/env python3
"""
	Given a rasterized DEM in Cartesian coordinates (e.g., UTM), 
	 compute the slope and slope-aspect maps.
	The DEM should be provided as a gdal-readable file, preferably
	 in GeoTiff format.
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

	# Outputs
	parser.add_argument('-o','--outName', dest='outName', type=str, required=True, help='Output name base')
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot outputs')

	# Options
	parser.add_argument('--nVectors', dest='nVectors', type=int, default=30, help='Number of vectors to plot if \'plot outputs\' option is selected.')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
## Plot gradient vectors
def plotGradientVectors(px,py,mapVals,T,n=30):
	"""
		Plot 20x20 vectors based on the pointing vector field
		INPUTS
			px are the x/east vectors
			py are the y/north vectors
			slope is the slope map
			T is the GDALtransform
	"""

	# Parameters
	M,N=px.shape

	mSkip=int(M/n)
	nSkip=int(N/n)

	east=np.arange(T.xstart,T.xend,T.xstep)
	north=np.arange(T.ystart,T.yend,T.ystep)

	E,N=np.meshgrid(east,north)

	# Resample to give nxn points
	E=E[1:-1:mSkip,1:-1:nSkip]
	N=N[1:-1:mSkip,1:-1:nSkip]

	vx=px[1:-1:mSkip,1:-1:nSkip]
	vy=py[1:-1:mSkip,1:-1:nSkip]

	# Plot
	Fig,ax=mapPlot(mapVals,cmap='viridis',pctmin=1,pctmax=99,background='auto',
			extent=T.extent,showExtent=True,cbar_orientation='horizontal',title='Vector plot')

	scale=np.abs(T.xend-T.xstart)/n+np.abs(T.yend-T.ystart)/n
	ax.quiver(E,N,scale*vx,scale*vy,
		color='r',units='xy',scale=2)


## Save georeferenced map
def saveMap(templateDS,bands,savename):
	"""
		Provide a template data set with same spatial extent and 
		 resolution as images to be saved.
	"""
	# Input parameters
	nBands=len(bands)

	# Gather parameters from template dataset
	N=templateDS.RasterXSize; M=templateDS.RasterYSize
	Proj=templateDS.GetProjection()
	Tnsf=templateDS.GetGeoTransform()

	# Save image to geotiff
	driver=gdal.GetDriverByName('GTiff')
	DSout=driver.Create(savename,N,M,nBands,gdal.GDT_Float32)
	for n,band in enumerate(bands):
		DSout.GetRasterBand(n+1).WriteArray(band)
	DSout.SetProjection(Proj)
	DSout.SetGeoTransform(Tnsf)
	DSout.FlushCache()



### MAIN ---
if __name__=='__main__':
	inpt=cmdParser()

	## Load DEM
	DEM=gdal.Open(inpt.DEMname,gdal.GA_ReadOnly)
	elev=DEM.GetRasterBand(1).ReadAsArray()

	# Determine extent
	T=GDALtransform(DEM)

	# Report if requested
	if inpt.verbose is True:
		print('Loaded DEM: {}'.format(inpt.DEMname))
		print('\tdimensions: {}'.format(elev.shape))


	## Compute gradients
	gradients=computeGradients(elev,dx=T.xstep,dy=T.ystep)

	slope=grad2slope(gradients)

	aspect=grad2aspect(gradients)


	## Construct pointing vectors
	# This vector points directly downhill at a given point
	px,py,pz=makePointingVectors(gradients)


	## Sanity checks
	# Check that slope recomputed from vectors matches original computation
	if inpt.verbose is True:
		slopeCalc=pointing2slope(px,py,pz)
		slopeDiff=slope-slopeCalc
		print('Slope difference: {} +/- {}'.format(np.nansum(slopeDiff),np.nanstd(slopeDiff)))

	# Check that aspect recomputed from vectors matches original computation
	if inpt.verbose is True:
		aspectCalc=pointing2aspect(px,py,pz)
		aspectDiff=aspect-aspectCalc
		print('Aspect difference: {} +/- {}'.format(np.nansum(aspectDiff),np.nanstd(aspectDiff)))

	# Check that all vectors are unit length
	if inpt.verbose is True:
		unitVectors=np.sqrt(px**2+py**2+pz**2)
		print('Unit vectors: {} +/- {}'.format(np.nanmean(unitVectors),np.nanstd(unitVectors)))


	## Save maps
	# Save slope map
	slopeName='{}_slope.tif'.format(inpt.outName)
	saveMap(templateDS=DEM,bands=[slope],savename=slopeName)

	# Save aspect map
	aspectName='{}_aspect.tif'.format(inpt.outName)
	saveMap(templateDS=DEM,bands=[aspect],savename=aspectName)

	# Save vector component map
	vectorName='{}_xyz_pointing_vectors.tif'.format(inpt.outName)
	saveMap(templateDS=DEM,bands=[px,py,pz],savename=vectorName)


	## Plot maps
	if inpt.plot is True:
		# Plot DEM
		mapPlot(elev,cmap='viridis',pctmin=1,pctmax=99,background='auto',
			extent=T.extent,showExtent=True,cbar_orientation='horizontal',title='DEM')

		# Plot Slope
		mapPlot(slope,cmap='viridis',pctmin=1,pctmax=99,background='auto',
			extent=T.extent,showExtent=True,cbar_orientation='horizontal',title='Slope')

		# Plot Aspect
		mapPlot(aspect,cmap='viridis',pctmin=1,pctmax=99,background='auto',
			extent=T.extent,showExtent=True,cbar_orientation='horizontal',title='Aspect')

		# Plot Vectors
		plotGradientVectors(px,py,elev,T,n=inpt.nVectors)

		plt.show()