#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt 
from osgeo import gdal
from InsarFormatting import *
from aria2LOS import orient2pointing


### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Convert LOS to ground displacement.')
	# Required
	parser.add_argument('-ascIFGs',dest='ascIFGs',nargs='+',help='Ascending interferograms')
	parser.add_argument('-ascAzimuthAngle',nargs='+',dest='ascAzimuthAngle',type=str,help='Ascending azimuth angle')
	parser.add_argument('-ascLookAngle',nargs='+',dest='ascLookAngle',type=str,help='Ascending look angle ')

	parser.add_argument('-dscIFGs',dest='dscIFGs',nargs='+',help='Descending interferograms')
	parser.add_argument('-dscAzimuthAngle',nargs='+',dest='dscAzimuthAngle',type=str,help='Descending azimuth angle')
	parser.add_argument('-dscLookAngle',nargs='+',dest='dscLookAngle',type=str,help='Descending look angle ')
	# Optional	
	parser.add_argument('-b','--bounds',dest='bounds',default=None,help='Bounding box WSEN')
	parser.add_argument('-r','--resolution',dest='resolution',default=None,help='xy pixel resolution (degrees)')

	# Extra-optional
	parser.add_argument('-v','--verbose',dest='verbose',action='store_true',help='Verbose mode')
	parser.add_argument('--plot_inputs','--plot_inputs',dest='plot_inputs',action='store_true',help='Plot inputs')
	return parser 

def cmdParser(inpt_args=None):
	parser = createParser()
	return parser.parse_args(inpt_args)


### --- Classes ---
class Orbit:
	def __init__(self,name):
		self.name=name


### --- Processing functions ---
# Resample imagery
def resample(DS,bounds,resolution,resampled_name):
	if not os.path.exists(resampled_name):
		# Resample file if needed
		DSresampled=gdal.Warp(resampled_name,DS,
			options=gdal.WarpOptions(
			format='VRT',outputType=gdal.GDT_Float32,
			outputBounds=bounds,xRes=resolution,yRes=resolution,
			resampleAlg='lanczos',dstNodata=0,srcNodata=0))
		print('Resampled: {}'.format(resampled_name))
	else:
		# ... or load existing file
		DSresampled=gdal.Open(resampled_name,gdal.GA_ReadOnly)
		print('Exists: {}'.format(resampled_name))
	return DSresampled


# Write pointing vectors to file
def writePointers(pointing_name,bands,N,M,proj,T):
	if not os.path.exists(pointing_name):
		# Write to file if needed
		driver=gdal.GetDriverByName('GTiff')
		DSpointing=driver.Create(pointing_name,Nresampled,Mresampled,3,gdal.GDT_Float32) 
		DSpointing.GetRasterBand(1).WriteArray(bands[0])
		DSpointing.GetRasterBand(2).WriteArray(bands[1])
		DSpointing.GetRasterBand(3).WriteArray(bands[2])
		DSpointing.SetProjection(proj) 
		DSpointing.SetGeoTransform(T) 
		DSpointing.FlushCache()
		print('Wrote pointing vectors to file: {}'.format(pointing_name))
	else:
		# ... or load existing file
		DSpointing=gdal.Open(pointing_name,gdal.GA_ReadOnly)
		print('Exists: {}'.format(pointing_name))
	return DSpointing


def invertForDisp_noNS():
	pass


def invertForDisp():
	pass


### --- Plotting functions ---
# For input data set
def DSplot(phs,lkAngle,azAngle,extent=None):
	F=plt.figure()
	# Plot phase
	mask=(phs==0) # phase mask
	ax=F.add_subplot(131)
	cax=ax.imshow(np.ma.array(phs,mask=mask),cmap='jet',
		extent=extent)
	ax.set_title('phase')
	F.colorbar(cax,orientation='horizontal')
	# Plot look angle
	ax=F.add_subplot(132)
	cax=ax.imshow(np.ma.array(lkAngle,mask=mask),cmap='viridis')
	ax.set_xticks([]); ax.set_yticks([])
	ax.set_title('look ang')
	F.colorbar(cax,orientation='horizontal')
	# Plot azimuth angle
	ax=F.add_subplot(133)
	cax=ax.imshow(np.ma.array(azAngle,mask=mask),cmap='hsv')
	ax.set_xticks([]); ax.set_yticks([])
	ax.set_title('az ang')
	F.colorbar(cax,orientation='horizontal')
	return F 

# For pointing vector maps
def PointingPlot(px,py,pz,extent=None):
	# x/E component
	F=plt.figure()
	axX=F.add_subplot(131)
	caxX=axX.imshow(Px)
	axX.set_title('E component')
	F.colorbar(caxX,orientation='horizontal')
	axY=F.add_subplot(132)
	caxY=axY.imshow(Py)
	axY.set_title('N component')
	F.colorbar(caxY,orientation='horizontal')
	axZ=F.add_subplot(133)
	caxZ=axZ.imshow(Pz)
	axZ.set_title('Z component')
	F.colorbar(caxZ,orientation='horizontal')
	return F



### --- Main function ---
if __name__=="__main__":
	inpt=cmdParser()

	## Sanity check
	# Check number of interferograms
	Nasc=len(inpt.ascIFGs) # nb ascending
	Ndsc=len(inpt.dscIFGs) # nb descending
	N=Nasc+Ndsc # total nb interferograms

	assert len(inpt.ascAzimuthAngle)==Nasc, 'Need same number ascending azimuth angle files as ifgs'
	assert len(inpt.ascLookAngle)==Nasc, 'Need same number ascending look angle files as ifgs'
	assert len(inpt.dscAzimuthAngle)==Ndsc, 'Need same number descending azimuth angle files as ifgs'
	assert len(inpt.dscLookAngle)==Ndsc, 'Need same number descending look angle files as ifgs'

	if inpt.verbose is True:
		print('{} ascending interferograms:'.format(Nasc))
		if Nasc>0:
			for i in inpt.ascIFGs:
				print('\t{}'.format(i))
		print('{} descending interferograms:'.format(Ndsc))
		if Ndsc>0:
			for i in inpt.dscIFGs:
				print('\t{}'.format(i))
		print('{} total interferograms'.format(N))

	# How many ascending vs descending?
	#  Inversion will still be performed, but warning will be displayed
	#   if not ascending and descending
	if Nasc<1:
		print('WARNING: No ascending interferograms detected.')
	elif Ndsc<1:
		print('WARNING: No descending interferograms detected.')

	## Merge interferogram lists
	IFGs=inpt.ascIFGs+inpt.dscIFGs # interferogram list
	LookAngles=inpt.ascLookAngle+inpt.dscLookAngle # look angle list
	AzimuthAngles=inpt.ascAzimuthAngle+inpt.dscAzimuthAngle # azimuth angle list

	## Load maps
	'''
	Format map data using ariaExtract.py. Use class Orbit to 
	 organize data.
	'''
	orbits=[] 
	xmin=[]; xmax=[]; ymin=[]; ymax=[]
	# Loop through each ascending file
	for i in range(N):
		# Instantiate one object per satellite pass
		name='{}'.format(i) # create generic name
		orbits.append(Orbit(name)) # add object to list

		# Phase map
		DSphs=gdal.Open(IFGs[i],gdal.GA_ReadOnly) # open dataset in gdal
		orbits[i].DSphs=DSphs # assign phase to object

		# Look angle
		DSlk=gdal.Open(LookAngles[i],gdal.GA_ReadOnly)
		orbits[i].DSlk=DSlk # assign look angle to object

		# Azimuth angle
		DSaz=gdal.Open(AzimuthAngles[i],gdal.GA_ReadOnly)
		orbits[i].DSaz=DSaz # assign azimuth to object

		# Geo transform
		T=DSphs.GetGeoTransform()
		m=DSphs.RasterYSize; n=DSphs.RasterXSize
		left=T[0]; dx=T[1]; right=left+n*dx
		top=T[3]; dy=T[5]; bottom=top+m*dy
		xmin.append(left); xmax.append(right)
		ymin.append(bottom); ymax.append(top)
		extent=(left, right, bottom, top)

		# Plot if desired
		if inpt.plot_inputs is True:
			phs=DSphs.GetRasterBand(1).ReadAsArray()
			lkAngle=DSlk.GetRasterBand(1).ReadAsArray()
			azAngle=DSaz.GetRasterBand(1).ReadAsArray()

			F=DSplot(phs,lkAngle,azAngle,extent=extent)
			F.suptitle('Input{}'.format(i))

	## Resample to same spatial extent
	# Establish minimum axis limits
	xmin=min(xmin); xmax=max(xmax) # x/Lon extent 
	ymin=min(ymin); ymax=max(ymax) # y/Lat extent 
	if inpt.bounds:
		bounds=[float(i) for i in inpt.bounds.split(' ')] # optional user-specified bounds
		print('Bounds (xmin ymin xmax ymax): ',bounds)
	else:
		bounds=(xmin,ymin,xmax,ymax)   # full map bounds in gdal format 
	# Establish resolution
	if inpt.resolution:
		resolution=inpt.resolution # optional user-specified resolution
	else:
		resolution=dx # use most recent dx value

	# Check if "overlaps" folder exists
	if not os.path.exists("Overlaps"):
		os.mkdir('Overlaps')
		print('New directory: Overlaps')

	# Loop through to resample each image
	resampled_orbits=[]
	for i in range(N):
		# Instantiate one object per satellite pass
		name='Resampled{}'.format(i) # create generic name
		resampled_orbits.append(Orbit(name)) # add object to list

		# Store/read VRT files in folders
		ResampledBasename=os.path.basename(IFGs[i]).strip('.vrt')

		# Resample phase
		resampled_name='Overlaps/Sample{}_{}phs_overlap.VRT'.format(i,ResampledBasename)
		resampled_orbits[i].DSphs=resample(orbits[i].DSphs,bounds,resolution,resampled_name)

		# Resample look angle
		resampled_name='Overlaps/Sample{}_{}lk_overlap.VRT'.format(i,ResampledBasename)
		resampled_orbits[i].DSlk=resample(orbits[i].DSlk,bounds,resolution,resampled_name)

		# Resample azimuth angle
		resampled_name='Overlaps/Sample{}_{}az_overlap.VRT'.format(i,ResampledBasename)
		resampled_orbits[i].DSaz=resample(orbits[i].DSaz,bounds,resolution,resampled_name)

		# Plot if desired
		if inpt.plot_inputs is True:
			T=resampled_orbits[i].DSphs.GetGeoTransform()
			m=resampled_orbits[i].DSphs.RasterYSize
			n=resampled_orbits[i].DSphs.RasterXSize
			left=T[0]; dx=T[1]; right=left+n*dx
			top=T[3]; dy=T[5]; bottom=top+m*dy
			extent=(left, right, bottom, top)

			phs=resampled_orbits[i].DSphs.GetRasterBand(1).ReadAsArray()
			lkAngle=resampled_orbits[i].DSlk.GetRasterBand(1).ReadAsArray()
			azAngle=resampled_orbits[i].DSaz.GetRasterBand(1).ReadAsArray()

			F=DSplot(phs,lkAngle,azAngle,extent=extent)
			F.suptitle('Resampled{}'.format(i))


	## Establish pointing vector from target to satellite at each location
	# Dimensions of the problem set
	Nresampled=resampled_orbits[0].DSphs.RasterXSize # map x-dimension
	Mresampled=resampled_orbits[0].DSphs.RasterYSize # map y-dimension

	# Check if "overlaps" folder exists
	if not os.path.exists('PointingVectors'):
		os.mkdir('PointingVectors')
		print('New directory: PointingVectors')

	# Loop through each orbit
	P=[] # empty list of pointing vector maps
	for i in range(N):
		# Represent the pointing vector at each point as a MxNx3 matrix
		#  first layer of matrix is E, second layer is N, third layer is z
		Px=np.zeros((Mresampled,Nresampled))
		Py=np.zeros((Mresampled,Nresampled))
		Pz=np.zeros((Mresampled,Nresampled))
		# Load sensor orientation maps for each orbit
		lkAngle=resampled_orbits[i].DSlk.GetRasterBand(1).ReadAsArray()
		azAngle=resampled_orbits[i].DSaz.GetRasterBand(1).ReadAsArray()
		
		# Loop through pixels and fill in vector maps
		for m in range(Mresampled):
			for n in range(Nresampled):
				theta=lkAngle[m,n]
				alpha=azAngle[m,n]
				p=orient2pointing(theta,alpha,verbose=False)#inpt.verbose)
				# Record to map
				Px[m,n]=p[0] # x/E
				Py[m,n]=p[1] # y/N
				Pz[m,n]=p[2] # z/V

		# Save to file
		pointingBasename=os.path.basename(IFGs[i]).strip('.vrt')
		pointing_name='PointingVectors/Sample{}_{}pointing'.format(i,pointingBasename)
		proj=resampled_orbits[0].DSphs.GetProjection()
		T=resampled_orbits[0].DSphs.GetGeoTransform()
		bands=[Px,Py,Pz] # store pmap values as band layers
		DSpointing=writePointers(pointing_name,bands,Nresampled,Mresampled,proj,T)

		if inpt.plot_inputs is True:
			F=PointingPlot(Px,Py,Pz)
			F.suptitle('Point-Sample{}'.format(i))

		# Add maps to the list
		P.append(DSpointing)


	## Invert for ground displacement
	# Case of two interferograms
	if N==2:
		# Assume NS component is zero


	# Plot if requested
	if inpt.plot_inputs is True:
		plt.show()