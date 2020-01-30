#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot and regress the second map relative to the first
# 
# by Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import mode
from osgeo import gdal 
from geoFormatting import GDALtransform


### --- Parser --- ###
def createParser():
	'''
		Plot most types of Insar products, including complex images and multiband images.
	'''

	import argparse
	parser = argparse.ArgumentParser(description='Plot most types of Insar products, including complex images and multiband images')
	# Necessary 
	parser.add_argument(dest='indepName', type=str, help='Map to serve as independent variable')
	parser.add_argument(dest='depName', type=str, help='Map to serve as dependent variable')
	# Comparison options
	parser.add_argument('-b','--bounds', dest='bounds', type=str, default=None, help='Map bounds (xmin ymin xmax ymax)')
	parser.add_argument('-m','--mask', dest='mask', type=str, default=None, help='Map to serve as mask')
	parser.add_argument('-bgDep', dest='bgDep', default=None, help='Background value for dependent variable')
	parser.add_argument('-bgIndp', dest='bgIndep', default=None, help='Background value for independent variable')
	# Comparison plot options
	parser.add_argument('-f','--figtype',dest='figtype', type=str, default='points', help='Type of figure to plot comparison points. [Points]')
	parser.add_argument('-ds','--ds_factor',dest='ds_factor', type=float, default=6, help='Downsample factor (power of 2). Default = 2^6 = 64')
	parser.add_argument('-nbins','--nbins',dest='nbins', type=int, default=20, help='Number of bins for 2D histograms')
	# Map plot options
	parser.add_argument('-p','--plotMaps', dest='plotMaps', action='store_true', help='Plot maps')
	parser.add_argument('--cbar-orientation', dest='cbarOrientation', type=str, default='vertical', help='Colorbar orientation')
	parser.add_argument('--cmapDep', dest='cmapDep', type=str, default='viridis', help='Colormap for dependent map')
	parser.add_argument('--cmapIndep', dest='cmapIndep', type=str, default='viridis', help='Colormap for independent map')

	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### --- Main function --- ###
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load data
	indepDS=gdal.Open(inpt.indepName,gdal.GA_ReadOnly)
	indepTnsf=GDALtransform(indepDS)
	depDS=gdal.Open(inpt.depName,gdal.GA_ReadOnly)
	depTnsf=GDALtransform(depDS)


	## Checks
	# Check that independent map within bounds if specified
	if inpt.bounds:
		bounds=inpt.bounds.split(' ')
		indepDS=gdal.Warp('',indepDS,options=gdal.WarpOptions(format='MEM',
			outputBounds=bounds))
		indepTnsf=GDALtransform(indepDS) # recompute transform
		if inpt.verbose is True:
			print('Resampling independent image to bounds: {}'.format(inpt.bounds))

	# Check that maps have the same spatial reference
	if indepTnsf.transform!=depTnsf.transform:
		# If spatial extents are not equal, resample depedent reference frame
		#  to independent reference frame
		M=indepDS.RasterYSize; N=indepDS.RasterXSize
		depDS=gdal.Warp('',depDS,options=gdal.WarpOptions(format="MEM",
				outputBounds=bounds,width=N,height=M,
				resampleAlg='lanczos'))
		if inpt.verbose is True:
			print('Resampling dependent image to match extent and resolution')

	## Images
	# Load images
	indepImg=indepDS.GetRasterBand(1).ReadAsArray()
	indepTnsf=GDALtransform(indepDS)
	M,N=indepImg.shape

	depImg=depDS.GetRasterBand(1).ReadAsArray()
	depTnsf=GDALtransform(depDS)


	## Mask and background values
	# Mask
	Mask=np.ones((M,N))

	# indepImg[indepImg==-9999]=0
	# depImg[depImg==-9999]=0

	# # Background values from native image format
	# if np.sum(np.isnan(depImg))>1:
	# 	# If NaNs detected in dependent image, treat as masked array
	# 	depImg=np.ma.masked_invalid(depImg)
	# 	Mask=np.ma.getmask(depImg)

	# if np.sum(np.isnan(indepImg))>1:
	# 	# If NaNs detected in independent image, treat as masked array
	# 	indepImg=np.ma.masked_invalid(indepImg)
	# 	Mask=Mask*np.ma.getmask(indepImg)

	# # Assigned background values
	# if inpt.bgDep is not None:
	# 	Mask=Mask*(depImg==inpt.bgDep)

	# if inpt.bgIndep is not None:
	# 	Mask=Mask*(indepImg==inpt.bgIndep)

	# # Apply mask
	# depImg=np.ma.array(depImg,mask=Mask)
	# indepImg=np.ma.array(indepImg,mask=Mask)


	## Plot maps if requested
	if inpt.plotMaps is True:
		Fmap=plt.figure()
		# Plot independent
		axIndep=Fmap.add_subplot(121)
		caxIndep=axIndep.imshow(indepImg,cmap=inpt.cmapIndep,extent=indepTnsf.extent)
		axIndep.set_title('Independent (base)')
		Fmap.colorbar(caxIndep,orientation=inpt.cbarOrientation)
		# Plot dependent
		axDep=Fmap.add_subplot(122)
		caxDep=axDep.imshow(depImg,cmap=inpt.cmapDep,extent=depTnsf.extent)
		axDep.set_title('Dependent (compare)')
		Fmap.colorbar(caxDep,orientation=inpt.cbarOrientation)


	## Comparisons
	# Reshape into 1D arrays
	# indepImg=indepImg.reshape(M*N,1)
	# depImg=depImg.reshape(M*N,1)

	indepImg=indepImg.flatten()
	depImg=depImg.flatten()

	# # Compress to ignore mask values
	# depImg=depImg.compressed()
	# indepImg=indepImg.compressed()

	## Plot Comparisons
	# Establish figure
	F=plt.figure()
	ax=F.add_subplot(111)

	# Plot points
	if inpt.figtype.lower() in ['points','pts']:
		dsamp=int(2**inpt.ds_factor) # downsample factor
		ax.plot(indepImg[::dsamp],depImg[::dsamp],'b.')

	# Plot 2D histogram
	if inpt.figtype.lower() in ['histogram','hist']:
		H,xedges,yedges=np.histogram2d(indepImg,depImg,bins=inpt.nbins)
		H=H.T
		X,Y=np.meshgrid(xedges,yedges)
		ax.pcolormesh(X,Y,H)

	plt.show()