#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 
from osgeo import gdal
from InsarFormatting import *
from aria2LOS import *


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
def invertForDisp():
	pass

### --- Plotting functions ---
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
	bounds=(xmin,ymin,xmax,ymax)   # full map bounds in gdal format 

	# Loop through to resample each image
	resampled_orbits=[]
	for i in range(N):
		# Instantiate one object per satellite pass
		name='Resampled{}'.format(i) # create generic name
		resampled_orbits.append(Orbit(name)) # add object to list

		# Resample phase
		resampled_orbits[i].DSphs=gdal.Warp('',orbits[i].DSphs,
			options=gdal.WarpOptions(
			format='MEM',outputType=gdal.GDT_Float32,
			outputBounds=bounds,xRes=dx,yRes=dy,
			resampleAlg='lanczos',dstNodata=0,srcNodata=0))

		# Resample look angle
		resampled_orbits[i].DSlk=gdal.Warp('',orbits[i].DSlk,
			options=gdal.WarpOptions(
			format='MEM',outputType=gdal.GDT_Float32,
			outputBounds=bounds,xRes=dx,yRes=dy,
			resampleAlg='lanczos',dstNodata=0,srcNodata=0))

		# Resample azimuth angle
		resampled_orbits[i].DSaz=gdal.Warp('',orbits[i].DSaz,
			options=gdal.WarpOptions(
			format='MEM',outputType=gdal.GDT_Float32,
			outputBounds=bounds,xRes=dx,yRes=dy,
			resampleAlg='lanczos',dstNodata=0,srcNodata=0))

		# Plot if desired
		if inpt.plot_inputs is True:
			phs=resampled_orbits[i].DSphs.GetRasterBand(1).ReadAsArray()
			lkAngle=resampled_orbits[i].DSlk.GetRasterBand(1).ReadAsArray()
			azAngle=resampled_orbits[i].DSaz.GetRasterBand(1).ReadAsArray()

			F=DSplot(phs,lkAngle,azAngle,extent=extent)
			F.suptitle('Resampled{}'.format(i))



	if inpt.plot_inputs is True:
		plt.show()