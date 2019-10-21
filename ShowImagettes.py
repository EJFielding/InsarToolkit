#!/usr/bin/env python3 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot "imagettes" of maps within a folder to quickly assess
#  the validity of each. Leverages the "mapShow" function.
# 
# by Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from glob import glob
#import numpy as np
import matplotlib.pyplot as plt
from MapShow import mapShow


### --- PARSER --- 
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Plot most types of Insar products, including complex images and multiband images')

	# Imagettes required
	parser.add_argument('-f','--fldr','--folder', dest='fldr', type=str, required=True, help='Folder where imagettes are stored')
	# Imagettes options
	parser.add_argument('-e','--extension', dest='ext', type=str, default=None, help='Extension of file names')
	parser.add_argument('-l','--filelist', dest='fileList', type=str, default=None, help='List of files in folder to plot')

	parser.add_argument('-m','--rows', dest='nrows', type=int, default=3, help='Number of rows per graph')
	parser.add_argument('-n','--cols', dest='ncols', type=int, default=5, help='Number of columns per graph')
	parser.add_argument('--show-extent', dest='show_extent', action='store_true', help='Show map extent')

	# MapShow options
	parser.add_argument('-b','--band', dest='band', default=1, type=int, help='Band to display. Default = 1')
	parser.add_argument('-c','--color','--cmap',dest='cmap', default='viridis', type=str, help='Colormap of plot')
	parser.add_argument('-ds', '--downsample', dest='dsample', default='0', type=int, help='Downsample factor (power of 2). Default = 2^0 = 1')
	parser.add_argument('-vmin','--vmin', dest='vmin', default=None, type=float, help='Min display value')
	parser.add_argument('-vmax','--vmax', dest='vmax', default=None, type=float, help='Max display value')
	parser.add_argument('-pctmin','--pctmin', dest='pctmin', default=0, type=float, help='Min value percent')
	parser.add_argument('-pctmax','--pctmax', dest='pctmax', default=100, type=float, help='Max value percent')
	parser.add_argument('-bg','--background', dest='background', default=None, help='Background value. Default is None. Use \'auto\' for outside edge of image.')
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

	# Report if specified
	if inpt.verbose is True:
		print('Images: {}'.format(Images))
		print('{} images detected'.format(nImages))


	## Plot imagettes
	M=inpt.nrows; N=inpt.ncols
	totalThumbnails=M*N 

	if inpt.verbose is True:
		print('Plotting images...')
		print('Max {} thumbnails per figure'.format(totalThumbnails))

	# Loop through image list
	x=1 # position variable
	for i in range(nImages):
		# Generate new figure if needed
		if x%totalThumbnails==1:
			F=plt.figure() # new figure
			x=1 # reset counter

		# Load image
		inpt.imgfile=Images[i] # assign image name
		Img=mapShow(inpt)

		# Subplot
		ax=F.add_subplot(M,N,x)
		ax.imshow(Img.img,extent=Img.extent,
			cmap=inpt.cmap,vmin=Img.vmin,vmax=Img.vmax)
		ax.set_title(i+1)
		if inpt.show_extent is False:
			ax.set_xticks([]); ax.set_yticks([])

		x+=1 # update counter

	plt.show()