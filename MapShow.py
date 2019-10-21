#!/usr/bin/env python3 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot most types of Insar products, including complex 
#  images and multi-band images
# 
# by Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import mode
from osgeo import gdal 


### --- PARSER --- 
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Plot most types of Insar products, including complex images and multiband images')
	# Necessary 
	parser.add_argument(dest='imgfile', type=str, help='File to plot')
	# Options
	parser.add_argument('-b','--band', dest='band', default=1, type=int, help='Band to display. Default = 1')
	parser.add_argument('-c','--color','--cmap', dest='cmap', default='viridis', type=str, help='Colormap of plot')
	parser.add_argument('-co','--cbar-orient', dest='cbar_orient', default='horizontal', type=str, help='Colorbar orientation')
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


### --- MAP SHOW ---
class mapShow:
	# Load and analyze image
	def __init__(self,inpt):
		## Basic parameters
		# Open image using gdal
		DS=gdal.Open(inpt.imgfile,gdal.GA_ReadOnly)
		nBands=DS.RasterCount # number of image bands

		# Geo transform
		M=DS.RasterYSize; N=DS.RasterXSize
		T=DS.GetGeoTransform()
		left=T[0]; dx=T[1]; right=left+N*dx 
		top=T[3]; dy=T[5]; bottom=top+M*dy 
		extent=(left, right, bottom, top)

		# Report basic parameters
		if inpt.verbose is True:
			print('Image: {}'.format(inpt.imgfile))
			print('BASIC PARAMETERS')
			print('Number of bands: {}'.format(nBands))
			print('Spatial extent: {}'.format(extent))
			print('Pixel size (x) {}; (y) {}'.format(dx,dy))


		## Image properties
		# Load image
		img=DS.GetRasterBand(inpt.band).ReadAsArray()

		# Image type (real/complex)
		datatype=type(img[0,0])
		if isinstance(img[0,0],np.complex64):
			imgMag=np.abs(img) # amplitude
			img=np.angle(img) # phase

		# Background value
		if inpt.background:
			if inpt.background=='auto':
				# Auto-determine background value
				edgeValues=np.concatenate([img[0,:],img[-1,:],img[:,0],img[:,-1]])
				background=mode(edgeValues).mode[0] # most common value
			else:
				# Use prescribed value
				background=float(inpt.background)
			
			mask=(img==background) # mask values
			img=np.ma.array(img,mask=mask) # mask main image array

		# Report
		if inpt.verbose is True:
			print('IMAGE PROPERTIES')
			print('data type: {}'.format(datatype))
			if inpt.background:
				print('background value: {:16f}'.format(background))


		## Image statistics
		imgArray=img.reshape(-1,1) # reshape 2D image as 1D array

		# Ignore background values
		if inpt.background:
			maskArray=mask.reshape(-1,1) # reshape mask from 2D to 1D
			imgArray=imgArray[maskArray==False] # mask background values

		# Ignore "outliers"
		if inpt.vmin:
			imgArray=imgArray[imgArray>=inpt.vmin]
		if inpt.vmax:
			imgArray=imgArray[imgArray<=inpt.vmax]

		# Percentages
		vmin,vmax=np.percentile(imgArray,(inpt.pctmin,inpt.pctmax))
		imgArray=imgArray[imgArray>=vmin]
		imgArray=imgArray[imgArray<=vmax]

		# Histogram analysis
		Hvals,Hedges=np.histogram(imgArray,bins=inpt.nbins)
		Hcntrs=(Hedges[:-1]+Hedges[1:])/2 # centers of bin edges

		# Report
		if inpt.verbose is True:
			print('IMAGE STATISTICS')
			if inpt.background:
				print('Ignoring background value')
			if inpt.vmin:
				print('vmin: {}'.format(inpt.vmin))
			if inpt.vmax:
				print('vmax: {}'.format(inpt.vmax))
			print('Upper left value: {:.16f}'.format(img[0,0]))


		## Store parameters for later
		# Image
		self.img=img; del img # image
		if inpt.plot_complex is True:
			# Amplitude image
			self.imgMag=imgMag; del imgMag
			self.minMag=minMag
			self.maxMag=maxMag
		# Stats
		self.Hcntrs=Hcntrs
		self.Hvals=Hvals
		self.vmin=vmin
		self.vmax=vmax
		# Geographic
		self.extent=extent


	# Plot map
	def plotMap(self,inpt):
		# Map plot
		dsample=int(2**inpt.dsample) # downsample factor

		F=plt.figure() 
		ax=F.add_subplot(111) 
		cax=ax.imshow(self.img[::dsample,::dsample],cmap=inpt.cmap,
			vmin=self.vmin,vmax=self.vmax,extent=self.extent) # <- Main image

		if inpt.plot_complex is True:
			self.imgMag=np.ma.array(self.imgMag,mask=mask)
			self.imgMag=self.imgMag**0.5
			minMag,maxMag=np.percentile(self.imgMag,(5,95))
			ax.imshow(self.imgMag[::dsample,::dsample],
				vmin=self.minMag,vmax=self.maxMag,
				cmap='Greys_r',extent=self.extent,
				zorder=1)
			cax.set_zorder(2); cax.set_alpha(0.3)

		F.colorbar(cax,orientation=inpt.cbar_orient) 

		# Plot histogram
		if inpt.hist is True:
			Fhist=plt.figure()
			axHist=Fhist.add_subplot(111)
			markerline, stemlines, baseline = plt.stem(Hcntrs, Hvals, 
				linefmt='r',markerfmt='',use_line_collection=True)
			stemlines.set_linewidths(None); baseline.set_linewidth(0)
			axHist.plot(self.Hcntrs,self.Hvals,'k',linewidth=2)
			axHist.set_ylim([0.95*self.Hvals.min(),1.05*self.Hvals.max()])

		plt.show()


### --- MAIN FUNCTION ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	M=mapShow(inpt)
	M.plotMap(inpt)