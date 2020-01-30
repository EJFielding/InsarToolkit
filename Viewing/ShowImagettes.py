#!/usr/bin/env python3 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot "imagettes" of maps within a folder to quickly assess
#  the validity of each. Leverages the "mapShow" function.
# 
# by Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from viewingFunctions import mapPlot, imagettes


### --- PARSER --- 
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Plot most types of Insar products, including complex images and multiband images')

	# Imagettes required
	parser.add_argument('-f','--fldr','--folder', dest='fldr', type=str, required=True, help='Folder where imagettes are stored')
	# Imagettes file type formatting
	parser.add_argument('-d','--h5dataset', dest='h5dataset', type=str, default='phase', help='Dataset for HDF5 file')
	parser.add_argument('-e','--extension', dest='ext', type=str, default=None, help='Extension of file names')
	parser.add_argument('-l','--filelist', dest='fileList', type=str, default=None, help='List of files in folder to plot')

	# Imagettes options
	parser.add_argument('-m','--rows', '--mRows', dest='mRows', type=int, default=3, help='Number of rows per graph')
	parser.add_argument('-n','--cols', '--nCols', dest='nCols', type=int, default=5, help='Number of columns per graph')
	parser.add_argument('--show-extent', dest='show_extent', action='store_true', help='Show map extent')

	# MapShow options
	parser.add_argument('-b','--band', dest='band', default=1, type=int, help='Band to display. Default = 1')
	parser.add_argument('-c','--color','--cmap',dest='cmap', default='viridis', type=str, help='Colormap of plot')
	parser.add_argument('-ds', '--downsample', dest='dsample', default='0', type=int, help='Downsample factor (power of 2). Default = 2^0 = 1')
	parser.add_argument('-vmin','--vmin', dest='vmin', default=None, type=float, help='Min display value')
	parser.add_argument('-vmax','--vmax', dest='vmax', default=None, type=float, help='Max display value')
	parser.add_argument('-pctmin','--pctmin', dest='pctmin', default=None, type=float, help='Min value percent')
	parser.add_argument('-pctmax','--pctmax', dest='pctmax', default=None, type=float, help='Max value percent')
	parser.add_argument('-cbar','--cbar-orientation', dest='colorbarOrientation', default=None, type=str, help='Colorbar orientation')
	parser.add_argument('-bg','--background', dest='background', default=None, help='Background value. Default is None. Use \'auto\' for outside edge of image.')
	parser.add_argument('-t','--titles', dest='titles', default=None, help='Title list')
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose')
	parser.add_argument('--plot_complex', dest='plot_complex', action='store_true', help='Plot amplitude image behind phase')
	parser.add_argument('-hist','--hist', dest='hist', action='store_true', help='Show histogram')
	parser.add_argument('--nbins', dest='nbins', default=50, type=int, help='Number of histogram bins. Default = 50')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- MAIN FUNCTION ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Discover images
	if inpt.fileList:
		# Use pre-generated list of files
		Flist=open(inpt.fileList,'r')
		fileList=Flist.readlines()
		Flist.close()
		Images=[os.path.join(inpt.fldr,f) for f in fileList] # list image files

	else:
		# Use images in folder
		Images=glob('{}'.format(inpt.fldr))

		if inpt.ext:
			l_ext=len(inpt.ext) # character length of extension
			Images=[i for i in Images if i[-l_ext:]==inpt.ext] # sort by extension

	# Number of images
	nImages=len(Images)


	## Stack into data cube
	# Check file extension
	if not inpt.ext:
		# Detect if not specified
		ext=Images[0].split('.')[-1] # file extension
	ext=ext.strip('.') # make sure no leading period
	
	# Handle files based on extension
	if ext=='h5':
		import h5py
		print('...imported h5py')
		with h5py.File(Images,'r') as DS: 
			print('Availble datasets: {}'.format(DS.keys()))
			imgs=DS[inpt.h5dataset]

	elif ext in ['tif','tiff','vrt']:
		from osgeo import gdal
		print('...imported gdal')

		# Start with list of files
		imgs=[]
		for img in Images:
			print(img)
			DS=gdal.Open(img,gdal.GA_ReadOnly)
			I=DS.GetRasterBand(1).ReadAsArray()
			imgs.append(I)
		imgs=np.array(imgs)

	# Report if specified
	if inpt.verbose is True:
		print('File extension: {}'.format(ext))
		print('Formatted into data cube. Dims = {}'.format(imgs.shape))

	# Report if specified
	if inpt.verbose is True:
		print('Images: {}'.format(Images))
		print('{} images detected'.format(nImages))


	## Plot imagettes
	if inpt.verbose is True:
		print('Plotting images...')
		print('Max {} thumbnails per figure'.format(inpt.mRows*inpt.nCols))

	if inpt.titles is not None:
		titleList=Images
	else:
		titleList=None

	# Imagette plot
	imagettes(imgs,inpt.mRows,inpt.nCols,downsampleFactor=inpt.dsample,
		cmap=inpt.cmap,vmin=inpt.vmin,vmax=inpt.vmax,pctmin=inpt.pctmin,pctmax=inpt.pctmax,
		colorbarOrientation=inpt.colorbarOrientation,background=inpt.background,
		extent=None,showExtent=False,titleList=titleList,supTitle=None)


	plt.show()
