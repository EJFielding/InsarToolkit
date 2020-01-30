#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from viewingFunctions import mapPlot, imagettes
from geoFormatting import transform2extent, GDALtransform


### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Save complex image to 2-band amplitude and phase')
	parser.add_argument('-fpath','--fpath', dest='fpath', type=str, default=None, help='Folder with image files')
	parser.add_argument('-f','--velocityFiles', dest='velocityFiles', nargs='+', type=str, help='List of velocity files')
	parser.add_argument('-w','--weightMaps', dest='weightMaps', nargs='+', type=str, default=None, help='Maps for weighting average offset value')
	parser.add_argument('-m','--maskMaps', dest='maskMaps', narg='+', type=str, default=None, help='Maps for masking values')
	parser.add_argument('-mval','--maskValue', dest='maskVal', type=float, default=0, help='Value below which pixels will be masked when computing offset')
	parser.add_argument('-e','--expec','--expectation', dest='expectation', type=str, default='mean', help='Expectation operator [\'mean\', \'median\']')
	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name')

	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('-plotInputs','--plotInputs', dest='plotInputs', action='store_true', help='Plot inputs')
	parser.add_argument('-plotOutputs','--plotOutputs', dest='plotOutputs', action='store_true', help='Plot outputs')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### --- Main function ---
if __name__=='__main__':
	## Script inputs 
	inpt=cmdParser()

	## Load images
	nImgs=len(inpt.velocityFiles)
	if inpt.fpath:
		inpt.velocityFiles=[os.path.join(inpt.fpath,file) for file in inpt.velocityFiles]
		if inpt.weightMaps:
			inpt.weightMaps=[os.path.join(inpt.fpath,file) for file in inpt.weightMaps]
		elif inpt.maskMaps:
			inpt.maskMaps=[os.path.join(inpt.fpath,file) for file in inpt.maskMaps]

	if inpt.verbose is True:
		print('Velocity files:')
		for vFile in inpt.velocityFiles:
			print('\t{}'.format(os.path.basename(vFile)))
		print('{} velocity files'.format(nImgs))
		if inpt.weightMaps:
			print('Weighting by files: {}'.format(inpt.weightMaps))
		elif inpt.maskMap:
			print('Masking by files: {}'.format(inpt.maskMaps))


	## Handle two maps at a time
	# Load base image
	Vbase=gdal.Open(inpt.velocityFiles[0],gdal.GA_ReadOnly)
	Tbase=GDALtransform(Vbase)

	if inpt.weightMaps:
		Wbase=gdal.Open(inpt.weightMaps[0],gdal.GA_ReadOnly)
	elif inpt.maskMaps:
		Mbase=gdal.Open(inpt.maskMaps[0],gdal.GA_ReadOnly)


	# For each additional image, resample to maximum spatial extent
	i=1
	for vName in inpt.velocityFiles[1:]:
		Vimg=gdal.Open(vName,gdal.GA_ReadOnly)
		Timg=GDALtransform(Vimg)

		if inpt.weightMaps:
			wName=inpt.weightMaps[i]
			Wimg=gdal.Open(wName,gdal.GA_ReadOnly)
		elif inpt.maskMaps:
			mName=inpt.maskMaps[i]
			Mimg=gdal.Open(mName,gdal.GA_ReadOnly)


		# Establish maximum bounds
		#  bounds = (xmin ymin xmax ymax)
		xMin=np.min([Tbase.xmin,Timg.xmin])
		yMin=np.min([Tbase.ymin,Timg.ymin])
		xMax=np.max([Tbase.xmax,Timg.xmax])
		yMax=np.max([Tbase.ymax,Timg.ymax])
		maxBounds=(xMin,yMin,xMax,yMax)
		if inpt.verbose is True:
			print('Max bounds: {}'.format(maxBounds))

		# Resample both images to new bounds
		Vbase=gdal.Warp('',Vbase,options=gdal.WarpOptions(format='MEM',outputBounds=maxBounds,srcNodata=0,dstNodata=0))
		Vimg=gdal.Warp('',Vimg,options=gdal.WarpOptions(format='MEM',outputBounds=maxBounds,srcNodata=0,dstNodata=0,width=Vbase.RasterXSize,height=Vbase.RasterYSize))

		if inpt.weightMaps:
			Wbase=gdal.Warp('',Wbase,options=gdal.WarpOptions(format='MEM',outputBounds=maxBounds,srcNodata=0,dstNodata=0))
			Wimg=''
		elif inpt.maskMaps:
			pass


		vBase=Vbase.ReadAsArray() # image array
		Tbase=GDALtransform(Vbase) # geo reference

		vImg=Vimg.ReadAsArray() # image array
		Timg=GDALtransform(Vimg) # geo reference
		assert Tbase.extent==Timg.extent, 'Check extents'

		vBase=np.ma.array(vBase,mask=(vBase==0)) # masked array
		vImg=np.ma.array(vImg,mask=(vImg==0)) # masked array

		#mapPlot(vImg,cmap='jet',background='auto')

		if inpt.plotInputs is True:
			imgs=[vBase,vImg]
			imagettes(imgs,mRows=1,nCols=2,
				cmap='jet',pctmin=1,pctmax=99,
				colorbarOrientation='horizontal', #background='auto',
				extent=Tbase.extent,showExtent=True,titleList=['base','new'],supTitle='Inputs')

		# Calculate difference between images in region of overlap
		D=vBase-vImg
		if inpt.plotInputs is True:
			mapPlot(D,cmap='jet',
				cbar_orientation='vertical',title='Difference (Base - Img)',
				extent=Tbase.extent,showExtent=True)

		if inpt.expectation in ['mean','average']:
			Dexpect=np.mean(D)
		elif inpt.expectation in ['median']:
			Dexpect=np.median(D)

		# Remove mean or median of overlap region from images
		vImg+=Dexpect

		# Merge into single image
		vBase.mask=np.ma.nomask
		vImg.mask=np.ma.nomask

		vBase+=vImg # merge images
		vBase[D!=np.ma.masked]/=2

		Vbase.GetRasterBand(1).WriteArray(vBase)

		if inpt.plotOutputs is True:
			mapPlot(vBase,cmap='jet',background='auto',pctmin=1,pctmax=99,
			cbar_orientation='vertical',title='Merged velocity field',
			extent=Tbase.extent,showExtent=True)

		i+=1


	## Save as single image
	if inpt.outName:
		outName='{}.tif'.format(inpt.outName)
		Driver=gdal.GetDriverByName('GTiff')
		Vout=Driver.Create(outName,vBase.shape[1],vBase.shape[0],1,gdal.GDT_Float32)
		Vout.GetRasterBand(1).WriteArray(vBase)
		Vout.SetProjection(Vbase.GetProjection())
		Vout.SetGeoTransform(Vbase.GetGeoTransform())
		Vout.FlushCache()
		print('Saved to: {}'.format(outName))



	plt.show()