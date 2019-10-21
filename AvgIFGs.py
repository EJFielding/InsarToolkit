#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from ImageTools import imgStats
from InsarFormatting import *

### --- Parser --- ###
def createParser():
	'''
		Stack interferograms to average them.
	'''

	import argparse
	parser = argparse.ArgumentParser(description='Stack interferograms to average them.')
	# Workflow 
	parser.add_argument('-t','--type',dest='prodtype',type=str, required=True, help='Type of product to analyze [\'aria\',\'extracted\']')
	# File list
	parser.add_argument('-l','--list',dest='filelist', type=str, default=None, help='List of files or dates to analyze (optional)')
	# Input for ARIA product information
	parser.add_argument('-f','--ariaFldr', dest='ariaFldr', type=str, help='Folder with ARIA products')
	# Input for Extracted product information
	parser.add_argument('-unw','--unwFldr', dest='unwFldr', type=str, default=None, help='Folder with unwrapped interferograms')
	parser.add_argument('-coh','--cohFldr', dest='cohFldr', type=str, default=None, help='Folder with coherence files')
	# Reference pixel

	# Miscellaneous
	parser.add_argument('-w','--wavelength', dest='wavelength', type=float, default=None, help='Radar wavelength')

	# Outputs
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose')
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot results')
	parser.add_argument('-o','--outName',dest='outName', type=str, default=None, help='Saves maps to output')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Loading/formatting data --- ###
## Load data for ARIA products
def loadImgDS(filename):
	DS=gdal.Open(filename,gdal.GA_ReadOnly)
	img=DS.GetRasterBand(1).ReadAsArray()
	tnsf=DS.GetGeoTransform()
	return img, tnsf


## Formatting data
# Plot map
def plotMap(img,extent,cmap='viridis'):
	F=plt.figure()
	ax=F.add_subplot(111)
	cax=ax.imshow(img,cmap=cmap,extent=extent)
	F.colorbar(cax,orientation='vertical')

# Plot maps
def plotMaps(coh,unw,extent,cmap='viridis'):
	F=plt.figure()
	# Coherence
	axCoh=F.add_subplot(121)
	cohStats=imgStats(coh,pctmin=2,pctmax=98)
	cax=axCoh.imshow(coh,cmap=cmap,extent=extent,vmin=cohStats.vmin,vmax=cohStats.vmax)
	F.colorbar(cax,orientation='horizontal')
	axUnw=F.add_subplot(122)
	unwStats=imgStats(unw,pctmin=2,pctmax=98)
	cax=axUnw.imshow(unw,cmap=cmap,extent=extent,vmin=unwStats.vmin,vmax=unwStats.vmax)
	F.colorbar(cax,orientation='horizontal')
	return F, axCoh, axUnw


### --- Main function --- ###
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Sanity checks
	# If ARIA products NETCDF files are specified, no need for separate folders with other product types
	if inpt.prodtype.lower() in ['aria']:
		assert inpt.unwFldr==None and inpt.cohFldr==None and inpt.ariaFldr!=None, 'Specify single folder for non-extracted ARIA products using -f option'
		workflow='aria'
	elif inpt.prodtype.lower() in ['extracted','extract','extr','exct']:
		assert inpt.unwFldr!=None and inpt.cohFldr!=None and inpt.ariaFldr==None, 'Specify unwrapped and coherence folders for extracted products'
		workflow='exct'

	## Detect dates and interferograms to stack
	if inpt.filelist:
		# Create list of files in filelist
		List=open(inpt.filelist)
		fileNames=[l.strip('\n') for l in List]
		List.close()
		if workflow=='aria':
			unwFileList=['NETCDF:"{}/{}":/science/grids/data/unwrappedPhase'.format(inpt.ariaFldr,n) for n in fileNames]
			cohFileList=['NETCDF:"{}/{}":/science/grids/data/coherence'.format(inpt.ariaFldr,n) for n in fileNames]
		elif workflow=='exct':
			unwFileList=['{}/{}'.format(inpt.unwFldr,n) for n in fileNames]
			cohFileList=['{}/{}'.format(inpt.cohFldr,n) for n in fileNames]
	else:
		# Create list of all files in folder
		if workflow=='aria':
			fileNames=glob.glob('{}/*.nc'.format(inpt.ariaFldr))
			unwFileList=['NETCDF:"{}/{}":/science/grids/data/unwrappedPhase'.format(inpt.ariaFldr,n) for n in fileNames]
			cohFileList=['NETCDF:"{}/{}":/science/grids/data/coherence'.format(inpt.ariaFldr,n) for n in fileNames]
		elif workflow=='exct':
			fileNames=glob.glob('{}/*.vrt'.format(inpt.unwFldr))
			fileNames=[os.path.basename(n) for n in fileNames] # remove absolute path
			unwFileList=['{}/{}'.format(inpt.unwFldr,n) for n in fileNames]
			cohFileList=['{}/{}'.format(inpt.cohFldr,n) for n in fileNames]
			assert len(unwFileList)==len(cohFileList), 'Must have same number of files in each folder'

	# Number of files
	n=len(fileNames)

	# Report file information if requested
	if inpt.verbose is True:
		print('Files listed: {}'.format(fileNames))
		print('{} interferograms detected'.format(n))


	## Dates and cumulative time
	# Create list of time intervals
	if workflow=='aria':
		print('Spatial resampling and date reformatting will need to be implemented')
		exit()
	elif workflow=='exct':
		times=[dateDiff(d.split('_')[0].strip('.vrt'),d.split('_')[1].strip('.vrt')) for d in fileNames]
	times=np.array(times) # as numpy array

	# Convert negative time to forward if necessary
	if times[0]<0:
		times=-times

	# Total time
	Ttotal=np.sum(times)

	# Report time information if requested
	if inpt.verbose is True:
		print('Time interals: {}'.format(times))
		print('Total time (years): {}'.format(Ttotal))

	## Coherence average to determine reference point location
	# First layer of coherence to set spatial reference
	avgCoh,tnsfCoh=loadImgDS(cohFileList[0])
	M,N=avgCoh.shape # pixel size of map
	T=GDALtransform(transform=tnsfCoh,shape=avgCoh.shape) # format transform
	Mask=(avgCoh==0.)

	# Loop through coherence maps
	for c in cohFileList[1:]:
			coh,tnsf=loadImgDS(c) # load map
			avgCoh+=coh # add to cumulative coherence
	avgCoh/=n # average by number of interferograms

	# Find reference point
	# Random values
	np.random.seed(0)
	xpx=np.random.randint(0,N,2000)
	ypx=np.random.randint(0,M,2000)

	# Associate coherence with values
	C=avgCoh[ypx,xpx]
	xpx=xpx[C==C.max()][0] # pixel locations
	ypx=ypx[C==C.max()][0]

	xcoord,ycoord=px2coords(tnsf,xpx,ypx)
	if inpt.verbose is True:
		print('Reference pixel location: x {}, y {}'.format(xpx,ypx))
		print('Reference coordinates: lon {} lat {}'.format(xcoord,ycoord))
		print('Coherence at reference: {}'.format(C.max()))


	## Unwrapped phase average
	# Check first layer of unwrapped phase
	avgUnw,tnsfUnw=loadImgDS(unwFileList[0])
	assert avgUnw.shape==(M,N) and tnsfUnw==tnsfCoh, 'Dimensions and extent must be same for coherence and unwrapped'

	# Update mask
	Mask*=(avgUnw==0.)

	# Loop through unwrapped interferograms
	for u in unwFileList[1:]:
		# Cumulative phase
		unw,tnsf=loadImgDS(u) # load unwrapped phase map
		unw-=unw[ypx,xpx] # subtract reference point
		avgUnw+=unw # add to cumulative phase
	avgUnw=avgUnw/Ttotal # average by cumulative time

	# Convert to cm if provided
	if inpt.wavelength:
		lengthFactor=inpt.wavelength/(4*np.pi)
		avgUnw/=lengthFactor


	## Save to file if requested
	if inpt.outName:
		# Load dataset with example parameters
		DS=gdal.Open(unwFileList[0],gdal.GA_ReadOnly)

		# Save avgCoherence map
		outNameCoh='{}_avgCoh.tif'.format(inpt.outName)
		driver=gdal.GetDriverByName('GTiff')
		Coh=driver.Create(outNameCoh,N,M,1,gdal.GDT_Float32)
		Coh.GetRasterBand(1).WriteArray(avgCoh)
		Coh.SetProjection(DS.GetProjection())
		Coh.SetGeoTransform(DS.GetGeoTransform()) 
		Coh.FlushCache() 
		print('Saved avgCoh: {}'.format(outNameCoh))

		# Save avgPhase map
		outNameUnw='{}_avgUnw.tif'.format(inpt.outName)
		driver=gdal.GetDriverByName('GTiff')
		Unw=driver.Create(outNameUnw,N,M,1,gdal.GDT_Float32)
		Unw.GetRasterBand(1).WriteArray(avgUnw)
		Unw.SetProjection(DS.GetProjection())
		Unw.SetGeoTransform(DS.GetGeoTransform())
		Unw.FlushCache()
		print('Saved avgUnw: {}'.format(outNameUnw))


	## Plot if requested
	if inpt.plot is True:
		avgCoh=np.ma.array(avgCoh,mask=Mask)
		avgUnw=np.ma.array(avgUnw,mask=Mask)
		F,axCoh,axUnw=plotMaps(avgCoh,avgUnw,T.extent) 
		# Plot reference point
		axCoh.plot(xcoord,ycoord,'k+')
		axUnw.plot(xcoord,ycoord,'k+')

		plt.show()