#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find percent of zeros in a connected components map
#
# R Zinke, 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


### --- Parser --- 
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='')

	# Required
	parser.add_argument(dest='files', type=str, help='Files')

	# Image options
	parser.add_argument('-ex','--excl','--exclude', dest='exclude', type=str, default=None, help='Data slice')
	parser.add_argument('-bg','--background', dest='background', default='auto', help='Background')
	parser.add_argument('--start', dest='start', type=int, default=0, help='Start')
	parser.add_argument('--end', dest='end', type=int, default=None, help='End')
	parser.add_argument('-ds','--downsample', dest='dsample', type=float, default=0, help='Downsample factor')

	# Options
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose')
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot')

	return parser


def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Main ---
if __name__=='__main__':
	# Import modules
	import os
	from glob import glob
	import copy
	import numpy as np
	import matplotlib.pyplot as plt
	from viewingFunctions import imgBackground, mapPlot, imagettes
	from pctZeros import pctZeros

	# Gather arguments
	inpt=cmdParser()
	
	# Files to load
	fileNames=glob(inpt.files)
	imgNames=[os.path.basename(fileName) for fileName in fileNames]
	nImgs=len(imgNames)

	ext=imgNames[0].split('.')[-1]
	
	# Report if requested
	if inpt.verbose is True:
		print('Files: {}'.format(imgNames))
		print('Extension: {}'.format(ext))
		print('Nb detected: {}'.format(nImgs))

	# Handle files based on extension
	if ext=='h5':
		import h5py
		print('...imported h5py')
		DS=h5py.File(fileNames[0],'r') 
		print('Availble datasets: {}'.format(DS.keys()))
		conncomps=DS['connectComponent']
		# Copy array
		if inpt.end:
			conncomps=conncomps[inpt.start:inpt.end,:,:]
		else:
			conncomps=conncomps[inpt.start:,:,:]

	elif ext in ['tif','tiff','vrt']:
		from osgeo import gdal
		print('tiff does not work yet'); exit()


	# Convert to ones and zeros
	for i in range(conncomps.shape[0]):
		conncomps[i,:,:]=np.where(conncomps[i,:,:]>0,1,0)

	# Exclude selected ifgs
	if inpt.exclude:
		excludeList=inpt.exclude.split(' ')
		excludeList=[int(e) for e in excludeList if int(e)>=inpt.start]
		if inpt.verbose is True:
			print('Excluding: {}'.format(excludeList))
		for excl in excludeList:
			conncomps[int(excl-inpt.start),:,:]=1

	# Cumulative product
	ccMask=np.cumprod(conncomps,axis=0)

	# Indices
	nMaps=conncomps.shape[0]
	indices=range(inpt.start,inpt.start+nMaps)

	# Calculate percent zeros
	titles=['{} {:.3f}'.format(i ,pctZeros(ccMask[i-inpt.start,:,:])) for i in indices]

	# Plot inputs
	imagettes(conncomps,3,5,downsampleFactor=inpt.dsample,
		cmap='jet',background=inpt.background,
		titleList=titles,supTitle='Input')

	# Plot mask
	imagettes(ccMask,3,5,downsampleFactor=inpt.dsample,
		cmap='jet',background=inpt.background,
		titleList=titles,supTitle='Mask')


	plt.show()
	
