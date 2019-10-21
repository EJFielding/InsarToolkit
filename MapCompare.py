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


### --- Parser --- ###
def createParser():
	'''
		Plot most types of Insar products, including complex images and multiband images.
	'''

	import argparse
	parser = argparse.ArgumentParser(description='Plot most types of Insar products, including complex images and multiband images')
	# Necessary 
	parser.add_argument(dest='depName', type=str, help='Map to serve as dependent variable')
	parser.add_argument(dest='indepName', type=str, help='Map to serve as independent variable')
	# Comparison options
	parser.add_argument('-m','--mask', dest='mask', type=str, default=None, help='Map to serve as mask')
	parser.add_argument('-bgDep', dest='bgDep', default=None, help='Background value for dependent variable')
	parser.add_argument('-bgIndp', dest='bgIndep', default=None, help='Background value for independent variable')
	# Comparison plot options
	parser.add_argument('-f','--figtype',dest='figtype', type=str, default='points', help='Type of figure to plot comparison points. [Points]')
	parser.add_argument('-ds','--ds_factor',dest='ds_factor', type=float, default=6, help='Downsample factor (power of 2). Default = 2^6 = 64')
	parser.add_argument('-nbins','--nbins',dest='nbins', type=int, default=20, help='Number of bins for 2D histograms')
	# Map plot options
	parser.add_argument('-p','--plotMaps', dest='plotMaps', action='store_true', help='Plot maps')
	parser.add_argument('--cmapDep', dest='cmapDep', type=str, default='viridis', help='Colormap for dependent map')
	parser.add_argument('--cmapIndep', dest='cmapIndep', type=str, default='viridis', help='Colormap for independent map')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### --- Main function --- ###
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load data
	# Load dependent (compare) image
	depDS=gdal.Open(inpt.depName,gdal.GA_ReadOnly)
	depImg=depDS.GetRasterBand(1).ReadAsArray()
	depTnsf=depDS.GetGeoTransform()

	# Format geo transform
	M=depDS.RasterYSize; N=depDS.RasterXSize
	left=depTnsf[0]; xstep=depTnsf[1]; right=left+xstep*N
	top=depTnsf[3]; ystep=depTnsf[5]; bottom=top+ystep*M
	extent=(left, right, bottom, top)
	bounds=(left, bottom, right, top)

	# Load independent (base) image
	indepDS=gdal.Open(inpt.indepName,gdal.GA_ReadOnly)
	indepTnsf=indepDS.GetGeoTransform()

	## Checks
	# Check that maps have the same spatial reference
	if indepTnsf!=depTnsf:
		# If spatial extents are not equal, resample depedent reference frame
		#  to independent reference frame
		M=depDS.RasterYSize; N=depDS.RasterXSize
		indepDS=gdal.Warp('',indepDS,options=gdal.WarpOptions(format="MEM",
				outputBounds=bounds,xRes=xstep,yRes=ystep,
				resampleAlg='lanczos'))

	# Independent image
	indepImg=indepDS.GetRasterBand(1).ReadAsArray()

	## Mask and background values
	# Mask
	Mask=np.ones((M,N))

	# Background values from native image format
	if np.sum(np.isnan(depImg))>1:
		# If NaNs detected in dependent image, treat as masked array
		depImg=np.ma.masked_invalid(depImg)
		Mask=np.ma.getmask(depImg)

	if np.sum(np.isnan(indepImg))>1:
		# If NaNs detected in independent image, treat as masked array
		indepImg=np.ma.masked_invalid(indepImg)
		Mask=Mask*np.ma.getmask(indepImg)

	# Assigned background values
	if inpt.bgDep is not None:
		Mask=Mask*(depImg==inpt.bgDep)

	if inpt.bgIndep is not None:
		Mask=Mask*(indepImg==inpt.bgIndep)

	# Apply mask
	depImg=np.ma.array(depImg,mask=Mask)
	indepImg=np.ma.array(indepImg,mask=Mask)


	## Plot maps if requested
	if inpt.plotMaps is True:
		Fmap=plt.figure()
		# Plot dependent
		axDep=Fmap.add_subplot(121)
		caxDep=axDep.imshow(depImg,cmap=inpt.cmapDep,extent=extent)
		axDep.set_title('Dependent (compare)')
		Fmap.colorbar(caxDep,orientation='vertical')
		# Plot independent
		axIndep=Fmap.add_subplot(122)
		caxIndep=axIndep.imshow(indepImg,cmap=inpt.cmapIndep,extent=extent)
		axIndep.set_title('Independent (base)')
		Fmap.colorbar(caxIndep,orientation='vertical')


	## Comparisons
	# Reshape into 1D arrays
	depImg=depImg.reshape(M*N,1)
	indepImg=indepImg.reshape(M*N,1)

	# Compress to ignore mask values
	depImg=depImg.compressed()
	indepImg=indepImg.compressed()

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