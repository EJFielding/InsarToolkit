#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
from osgeo import gdal
from InsarFormatting import *
from ImageTools import imgStats

### --- Parser ---
def createParser():
	'''
		Provide a list of triplets to investigate phase closure.
		Requires a list of triplets---Use "tripletList.py" script to
		 generate this list. 
	'''

	import argparse
	parser = argparse.ArgumentParser(description='Determine phase closure.')
	# Required inputs
	parser.add_argument('-l','--triplet-list', dest='tlist', type=str, required=True, help='Nx3 list of interferograms representing triplets to be considered')
	parser.add_argument('-f','--fldr', dest='fldr', type=str, required=True, help='Folder with unwrapped interferograms')
	# Reference pixel
	parser.add_argument('-refLoLa','--refLoLa', dest='refLoLa', nargs=2, type=float, default=None, help='Lon lat for reference pixel')
	parser.add_argument('-refXY','--refXY', dest='refXY', nargs=2, type=int, default=None, help='X and Y locations for reference pixel')
	# Toss out bad maps
	parser.add_argument('--toss', dest='toss', type=str, default=None, help='Toss out maps by ID number')
	# Masking options
	parser.add_argument('--watermask', dest='watermask', type=str, default=None, help='Watermask')
	# Query points
	parser.add_argument('--lims', dest='lims', nargs=2, type=float, default=None, help='Misclosure plots y-axis limits (rads)')
	parser.add_argument('--trend','--fit-trend', dest='trend', type=str, default=None, help='Fit a linear or periodic trend to the misclosure points [\'linear\', \'periodic\']')

	# Outputs
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose')
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot results')
	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Saves maps to output')
	parser.add_argument('-ot','--outType',dest='outType', type=str, default='GTiff', help='Format of output file')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Loading/formatting data ---
## Load data for ARIA products
def loadImgDS(filename):
	DS=gdal.Open(filename,gdal.GA_ReadOnly)
	img=DS.GetRasterBand(1).ReadAsArray()
	return img


## Formatting data
def plotMap(img,TransformObj=None,cmap='viridis',title=None,ref=None,query=None):
	F=plt.figure()
	img=np.ma.array(img,mask=(img==img[0,0]))
	stats=imgStats(img,pctmin=2,pctmax=98)
	ax=F.add_subplot(111)
	cax=ax.imshow(img,
		vmin=-np.pi, vmax=np.pi, cmap=cmap,
		extent=TransformObj.extent)
	F.colorbar(cax,orientation='horizontal')
	if title:
		ax.set_title(title)
	if ref:
		ax.plot(ref[0],ref[1],'ks')
	if query and qX is not None:
		ax.plot(query[0],query[1],'bd')
	return F

def plotMeanMisclosure(img,TransformObj=None,cmap='viridis',title=None,ref=None,query=None):
	F=plt.figure()
	img=np.ma.array(img,mask=(img==img[0,0]))
	stats=imgStats(img,pctmin=2,pctmax=98)
	ax=F.add_subplot(111)
	cax=ax.imshow(img,
		vmin=stats.vmin, vmax=stats.vmax, cmap=cmap,
		extent=TransformObj.extent)
	F.colorbar(cax,orientation='horizontal')
	if title:
		ax.set_title(title)
	if ref:
		ax.plot(ref[0],ref[1],'ks')
	if query and qX is not None:
		ax.plot(query[0],query[1],'bd')

	return F


## Plot misclosure points
# Misclosure values
def misclosureTimePts(event):
	# Location
	lon=event.xdata; lat=event.ydata
	px,py=coords2px(tnsf,lon,lat)

	# Mean misclosure
	meanValue=MeanMisclosure[py,px]
	print('lon {:4f} lat {:4f}; px {} py {} = (mean) {}'.format(lon,lat,px,py,meanValue))

	## Misclosure as a function of time
	# Misclosure data
	misclosure_pts=Misclosure[py,px,:]

	# Trend of misclosure data
	if inpt.trend.lower() in ['linear']:
		# Linear trend
		trend=fit_linear_trend(dates,misclosure_pts) # use linear trend fit
		t_finescale=np.linspace(dates.min(),dates.max(),1000)
		trend.reconstruct(t_finescale)
		if inpt.verbose is True:
			print('\tTrend--Linear: {:.3f}'.format(trend.A))
	elif inpt.trend.lower() in ['periodic']:
		# Periodic + linear trend
		trend=fit_periodic_trend(dates,misclosure_pts) # use periodic trend fit
		t_finescale=np.linspace(dates.min(),dates.max(),1000)
		trend.reconstruct(t_finescale)
		if inpt.verbose is True:
			print('\tTrend--Periodic: {:.3f}; Linear: {:.3f}'.format(trend.P,trend.C))

	# Misclosure plot
	axMisclosure.cla()
	axMisclosure.plot(dates,misclosure_pts,'-ko',zorder=3)
	axMisclosure.axhline(0,color=(0.65,0.65,0.65),zorder=1)
	axMisclosure.set_ylabel('misclosure (rad)')
	if inpt.trend is not None:
		axMisclosure.plot(t_finescale,trend.yhat,color=(0.3,0.4,0.7),zorder=2)

	## Cumulative misclosure as a function of time
	# Cum. misclosure data
	cum_misclosure_pts=np.cumsum(misclosure_pts)

	# Cum. misclosure plot
	axCumMisclosure.cla()
	axCumMisclosure.plot(dates,cum_misclosure_pts,'-ko',zorder=3)
	axCumMisclosure.axhline(0,color=(0.65,0.65,0.65),zorder=1)
	axCumMisclosure.set_ylabel('cum. miscl. (rad)')
	axCumMisclosure.set_xlabel('years since {}'.format(refDate))

	F_misclosure.canvas.draw()


### --- Main function ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load input files
	# Load list of triplets
	listFile=open(inpt.tlist,'r')
	Lines=listFile.readlines()
	listFile.close()

	triplets=[l.strip('\n') for l in Lines] # Triplet list
	
	# Remove invalid dates
	if inpt.toss:
		toss=[int(i) for i in inpt.toss.split()] # convert to integers
		triplets=[t for i,t in enumerate(triplets) if i not in toss]

	nTriplets=len(triplets)
	if inpt.verbose is True:
		print('{} triplets specified'.format(nTriplets))

	# Which maps to toss, if any
	if inpt.toss:
		toss=[int(t) for t in inpt.toss.split()]
	else:
		toss=[] # empty list
		print('toss:',toss)

	## Template dataset and prep
	# 3D array for storing data
	templateName=os.path.join(inpt.fldr,triplets[0].split()[0])
	DStemplate=gdal.Open(templateName,gdal.GA_ReadOnly) # use first ifg as template
	tnsf=DStemplate.GetGeoTransform(); Tnsf=GDALtransform(DStemplate); proj=DStemplate.GetProjection()
	N=DStemplate.RasterXSize; M=DStemplate.RasterYSize	
	Misclosure=np.zeros((M,N,nTriplets)) # 3D array (y,x,n)

	# Reference date
	refDate=triplets[0].split()[0].split('_')[1]
	if inpt.verbose is True:
		print('Reference date: {}'.format(refDate))

	# Reference points
	if inpt.refLoLa:
		xcoord,ycoord=inpt.refLoLa # Lon/lat
		xpx,ypx=coords2px(tnsf,xcoord,ycoord) # pixels
		assert inpt.refXY is None, 'Only specify ref point in coordinates or pixels; not both'
	elif inpt.refXY:
		xpx,ypx=inpt.refXY # X Y
		xcoord,ycoord=px2coords(tnsf,xpx,ypy) # coordinates


	## Apply masking
	Mask=np.ones((M,N))

	# Watermask
	if inpt.watermask:
		# Resample
		WM=gdal.Open(inpt.watermask,gdal.GA_ReadOnly)
		WM=gdal.Warp('',WM,options=gdal.WarpOptions(format='MEM',outputBounds=Tnsf.bounds,xRes=Tnsf.xstep,yRes=Tnsf.ystep))
		Mask*=WM.ReadAsArray()

	## Calculate phase misclosure
	dates=[]
	# Loop through each triplet
	for t in range(nTriplets):
		# Evaluate each triplet
		triplet=triplets[t].split()
		IJ=triplet[0]; IJname=os.path.join(inpt.fldr,IJ)
		JK=triplet[1]; JKname=os.path.join(inpt.fldr,JK)
		IK=triplet[2]; IKname=os.path.join(inpt.fldr,IK)
		IJmap=loadImgDS(IJname); IJmap*=Mask
		JKmap=loadImgDS(JKname); JKmap*=Mask
		IKmap=loadImgDS(IKname); IKmap*=Mask

		# Zero at reference point
		# IJmap-=IJmap[ypx,xpx]
		# JKmap-=JKmap[ypx,xpx]
		# IKmap-=IKmap[ypx,xpx]

		# Record time from reference date
		dates.append(dateDiff(refDate,IJ.split('_')[1]))

		# Sum phases
		SumIJK=IJmap+JKmap-IKmap

		Misclosure[:,:,t]=SumIJK
		if inpt.verbose is True:
			print('Triplet {}: {}'.format(t,triplets[t]))


	## Compute misclosure
	# Mean misclosure
	MeanMisclosure=np.mean(Misclosure,axis=2)
	F_mean_misclosure=plotMeanMisclosure(MeanMisclosure,Tnsf,cmap='jet',title='Mean misclosure',ref=(xcoord,ycoord))

	# Misclosure time points
	dates=np.array(dates)
	F_misclosure=plt.figure()
	axMisclosure=F_misclosure.add_subplot(211); axCumMisclosure=F_misclosure.add_subplot(212)
	F_misclosure.suptitle('Phase misclosure over time')

	F_mean_misclosure.canvas.mpl_connect('button_press_event',misclosureTimePts)


	## Save to file if requested
	if inpt.outName:
		outName='Mean_Misclosure_{}'.format(inpt.outName)
		if inpt.outType.lower() in ['gtiff','tiff','tif']:
			# Save mean misclosure map
			outName=outName+'.tif'
			driver=gdal.GetDriverByName('GTiff')
			OutMap=driver.Create(outName,N,M,1,gdal.GDT_Float32)
			OutMap.GetRasterBand(1).WriteArray(MeanMisclosure)
			OutMap.SetProjection(proj)
			OutMap.SetGeoTransform(tnsf)
			OutMap.FlushCache()
		elif inpt.outType.lower() in ['hdf5','h5']:
			outName=outName+'.h5'
			OutMap=h5py.File(outName,'w')
			OutData=OutMap.create_dataset("MeanMisclosure",(M,N))
			OutData[:,:]=MeanMisclosure
		print('Saved: {}'.format(outName))


	plt.show()