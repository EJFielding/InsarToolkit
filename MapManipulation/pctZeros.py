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
	parser.add_argument('-s','--slice', dest='slice', type=int, default=None, help='Data slice')
	parser.add_argument('-bg','--background', dest='background', default='auto', help='Background')

	# Options
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose')
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot')

	return parser


def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Percent zeros ---
def pctZeros(img,background=None):
	img=img.flatten()
	if background:
		img=img[img!=background]
	totalPixels=len(img)
	nbZeros=len(img[img==0])
	percentZeros=nbZeros/totalPixels
	return percentZeros


### --- Main ---
if __name__=='__main__':
	# Import modules
	import os
	from glob import glob
	import numpy as np
	import matplotlib.pyplot as plt
	from osgeo import gdal
	from viewingFunctions import imgBackground, mapPlot

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

	elif ext in ['tif','tiff','vrt']:
		print('tiff does not work yet'); exit()

	# Determine connected component percents
	if inpt.slice is not None:
		print(conncomps)
		img=conncomps[inpt.slice,:,:]

		# Background values
		if inpt.background=='auto':
			background=imgBackground(img)
		else:
			background=inpt.background
		pct=pctZeros(img,background)
		print('Percent zeros: {}'.format(pct))

		if inpt.plot is True:
			mapPlot(img,cmap='jet')#,background=background)
	else:
		pass

	plt.show()
	
