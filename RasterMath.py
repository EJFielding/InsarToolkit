#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run mathematical operations on a raster and write it to a file.
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
		Run mathematical operations on a raster and write it to a file.
	'''

	import argparse
	parser = argparse.ArgumentParser(description='Enter an equation to modify and save a raster map')
	# Necessary 
	parser.add_argument(dest='ogMap', type=str, help='Original (unmodified) map')
	parser.add_argument('-m','--math', dest='oper', type=str, help='Math operation(s) to be applied. Use \'m\' to represent original map.')
	# Options
	parser.add_argument('-bg', '--background', dest='background', default=None, help='Convert background value to zero')
	parser.add_argument('-t_bg', '--t_background', dest='t_background', default=None, help='Background value for target raster')
	# Outputs
	parser.add_argument('-o','--outName', dest='outName', default=None, help='Name of output file')

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
	ogDS=gdal.Open(inpt.ogMap,gdal.GA_ReadOnly)
	ogImg=ogDS.GetRasterBand(1).ReadAsArray()

	# Geo transform
	ogTnsf=ogDS.GetGeoTransform()
	M=ogDS.RasterYSize; N=ogDS.RasterXSize
	left=ogTnsf[0]; xstep=ogTnsf[1]; right=left+xstep*N 
	top=ogTnsf[3]; ystep=ogTnsf[5]; bottom=top+ystep*M 
	extent=(left, right, bottom, top)


	## Handle background
	if inpt.background is not None:
		if inpt.background=='auto':
			# Auto-detect background value
			toprow=ogImg[0,:]
			leftcol=ogImg[:,0]
			rightcol=ogImg[:,-1]
			bottomrow=ogImg[-1,:]
			vals=np.concatenate([toprow,leftcol,rightcol,bottomrow])
			bg=mode(vals).mode[0]
		else:
			bg=inpt.background

		# Generate background mask
		BGmask=(ogImg==bg)

	# Target background value
	if inpt.t_background:
		t_bg=inpt.t_background # replace with original background value unless specified
	else:
		t_bg=bg # user-specified background value


	## Apply maths
	m=ogImg.copy()
	y=lambda m: eval(inpt.oper)
	modImg=y(m) # modified image

	# Target background
	modImg[BGmask==True]=t_bg


	## Save to file
	if inpt.outName:
		driver=gdal.GetDriverByName('GTiff')
		outImg=driver.Create('{}.tif'.format(inpt.outName),N,M,1,gdal.GDT_Float32)
		outImg.GetRasterBand(1).WriteArray(modImg)
		outImg.SetProjection(ogDS.GetProjection())
		outImg.SetGeoTransform(ogDS.GetGeoTransform())
		outImg.FlushCache()

	## Plots
	# Plot input
	F=plt.figure()
	ax=F.add_subplot(111)
	cax=ax.imshow(ogImg,cmap='Greys_r',extent=extent)
	F.colorbar(cax,orientation='horizontal')

	# # Plot output
	F=plt.figure()
	ax=F.add_subplot(111)
	cax=ax.imshow(modImg,cmap='Greys_r',extent=extent)
	F.colorbar(cax,orientation='horizontal')

	plt.show()