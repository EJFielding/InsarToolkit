#!/usr/bin/env python3
"""
	Compute and display the statistics of a map data set. This is
	 designed as a standalone script.
"""

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from osgeo import gdal


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compute and display the statistics of a map data set.')
	# Input data set
	parser.add_argument(dest='mapName', type=str, help='Image data set')
	# Parameters
	parser.add_argument('-m','--masks', dest='masks', default=None, nargs='+', help='Values to mask ([None], \'auto\' will mask out background)')
	parser.add_argument('-pct','--percentiles', dest='pcts', default=None, help='Percentiles to calculate (e.g., \'5,26,50,84,95\')')

	# Outputs
	parser.add_argument('-p','--plot-map', dest='plot', action='store_true', help='Plot map')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### MASKING ---
## Image background
def background(img):
	# Use mode of background values
	edgeValues=np.concatenate([img[0,:],img[-1,:],img[:,0],img[:,-1]])
	background=mode(edgeValues).mode[0] # most common value
	return background


## Generate masking array
def generateMask(inpt,img):
	mask=np.ones(img.shape)

	# Mask values
	if inpt.masks:
		for maskValue in inpt.masks:
			if maskValue=='auto':
				maskValue=background(img)

			# Apply mask value
			mask[img==maskValue]=0

	return mask



### MAIN ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load data
	# Load image data set
	DS=gdal.Open(inpt.mapName,gdal.GA_ReadOnly)
	img=DS.GetRasterBand(1).ReadAsArray()

	# Apply masks
	mask=generateMask(inpt,img)
	img=np.ma.array(img,mask=(mask==0))

	# Plot if specified
	if inpt.plot is True:
		Fig=plt.figure()
		ax=Fig.add_subplot(111)
		cax=ax.imshow(img)
		Fig.colorbar(cax,orientation='horizontal')


	## Spatial statistics
	tnsf=DS.GetGeoTransform()
	N=DS.RasterXSize; M=DS.RasterYSize
	xstart=tnsf[0]; xstep=tnsf[1]; xend=xstart+xstep*N
	ystart=tnsf[3]; ystep=tnsf[5]; yend=ystart+ystep*M


	## Image statistics
	# Remove masked values
	img=img.flatten().compressed()
	nPx=len(img)

	# Basic statistics
	Imin=np.min(img)
	Imax=np.max(img)
	Imean=np.mean(img)
	Imedian=np.median(img)
	Imode=mode(img).mode[0]

	# Percentiles
	Ilower95,Iupper95=np.percentile(img,[2.5,97.5])
	if inpt.pcts is not None:
		pcts=[float(pct) for pct in inpt.pcts.split(',')]
		Ipct=np.percentile(img,pcts)


	## Print stats
	# Print spatial statistics
	print('SPATIAL STATISTICS')
	print('X_SIZE {}'.format(N))
	print('Y_SIZE {}'.format(M))
	print('X_MIN {}'.format(xstart))
	print('X_MAX {}'.format(xend))
	print('Y_MIN {}'.format(yend))
	print('Y_MAX {}'.format(ystart))
	print('X_STEP {}'.format(xstep))
	print('Y_STEP {}'.format(ystep))
	print('PIXELS {}'.format(M*N))
	print('VALID_PIXELS {}'.format(nPx))

	# Print spectral statistics
	print('SPECTRAL STATISTICS')
	print('IMAGE_MEAN {}'.format(Imean))
	print('IMAGE_MEDIAN {}'.format(Imedian))
	print('IMAGE_MODE {}'.format(Imode))
	print('IMAGE_MIN {}'.format(Imin))
	print('IMAGE_MAX {}'.format(Imax))
	print('IMAGE_LOWER95 {}'.format(Ilower95))
	print('IMAGE_UPPER95 {}'.format(Iupper95))
	if inpt.pcts is not None:
		print('IMAGE_PCTS {}'.format(Ipct))


	plt.show()