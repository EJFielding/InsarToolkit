#!/usr/bin/env python3
import os
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import h5py
# from InsarFormatting import *
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
	parser.add_argument('-f','--fname','--filename', dest='fname', type=str, required=True, help='Name of HDF5 file')
	parser.add_argument('-d','--dataset', dest='subDS', type=str, help='Name of sub-data set')
	# Date/time criteria
	parser.add_argument('--min-time', dest='minTime', type=float, default=None, help='Minimum amount of time (years) between acquisitions in a pair')
	parser.add_argument('--max-time', dest='maxTime', type=float, default=None, help='Maximum amount of time (years) between acquisitions in a pair')
	# Toss out bad maps
	parser.add_argument('--toss', dest='toss', type=str, default=None, help='Toss out maps by ID number')
	# Reference pixel
	parser.add_argument('-refXY','--refXY', dest='refXY', nargs=2, type=int, default=None, help='X and Y locations for reference pixel')
	# Masking options
	parser.add_argument('--watermask', dest='watermask', type=str, default=None, help='Watermask')
	# Query points
	parser.add_argument('--lims', dest='lims', nargs=2, type=float, default=None, help='Misclosure plots y-axis limits (rads)')
	parser.add_argument('--trend','--fit-trend', dest='trend', type=str, default=None, help='Fit a linear or periodic trend to the misclosure points [\'linear\', \'periodic\']')

	# Outputs
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose')
	parser.add_argument('--plot-inputs','--plotInputs', dest='plotInputs', action='store_true', help='Plot inputs')
	parser.add_argument('--plot-misclosure','--plotMisclosure', dest='plotMisclosure', action='store_true', help='Plot imagettes of misclosure')
	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Saves maps to output')
	parser.add_argument('-ot','--outType', dest='outType', type=str, default='GTiff', help='Format of output file')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Loading/formatting data ---
## Formatting data

# Dates
def formatDates(dateDS):
	# List of master-slave date pairs
	datePairs=dateDS[:,:].astype('int') # Reformat as appropriate data type
	# List of unique dates
	allDates=[]; [allDates.extend(pair) for pair in datePairs] # add dates from pairs
	dates=[]; [dates.append(d) for d in allDates if d not in dates] # limit to unique dates
	if inpt.verbose is True:
		print('Date pairs:\n{}'.format(datePairs))
		print('Unique dates:\n{}'.format(dates))
	return dates, datePairs

# Date difference
def days_between(d1,d2):
	d1 = datetime.strptime(str(d1),"%Y%m%d")
	d2 = datetime.strptime(str(d2),"%Y%m%d")
	return abs((d2-d1).days)

# Triplets
def formatTriplets(datePairs,inpt):
	# Loop through date pairs to find all valid triplet combinations
	nDatePairs=datePairs.shape[0] # number of date pairs
	triplets=[]
	for pairIJ in datePairs:
		# Pair IJ - first date; second date
		dateI=pairIJ[0]; dateJ=pairIJ[1]
		# Pairs JK - pairs with second date as master
		pairsJK=datePairs[datePairs[:,0]==dateJ]
		if len(pairsJK)>0:
			# Pairs IK - pairs with I as master and K as slave
			for dateK in pairsJK[:,1]:
				pairsIK=datePairs[(datePairs[:,0]==dateI) & (datePairs[:,1]==dateK)]
				if len(pairsIK)>0:
					# Record valid date pairs to list
					pairList=[[dateI,dateJ],[dateJ,dateK],[dateI,dateK]]
					triplets.append(pairList)
	nTriplets=len(triplets)

	# Remove pairs that do not meet temporal requirements
	failed_conditions=[] # make a list of indexes where conditions not met
	for t in range(nTriplets):
		# List pairs
		pairIJ=triplets[t][0]
		pairJK=triplets[t][1]
		pairIK=triplets[t][2]

		# Compute time intervals
		IJdt=days_between(pairIJ[0],pairIJ[1])/365
		JKdt=days_between(pairJK[0],pairJK[1])/365
		IKdt=days_between(pairIK[0],pairIK[1])/365

		# Check against conditions
		if inpt.maxTime: # maxTime
			conditions_met=[False,False,False]
			conditions_met[0]=True if IJdt<inpt.maxTime else False
			conditions_met[1]=True if JKdt<inpt.maxTime else False
			conditions_met[2]=True if IKdt<inpt.maxTime else False
			# Record index if not all maxTime conditions met
			if sum(conditions_met)<3:
				failed_conditions.append(t)

		if inpt.minTime: # minTime
			conditions_met=[False,False,False]
			conditions_met[0]=True if IJdt>inpt.minTime else False
			conditions_met[1]=True if JKdt>inpt.minTime else False
			conditions_met[2]=True if IKdt>inpt.minTime else False
			# Record index if not all minTime conditions met
			if sum(conditions_met)<3:
				failed_conditions.append(t)

	# Unique values of failed conditions
	failed_conditions=list(set(failed_conditions))
	failed_conditions=failed_conditions[::-1] # work from back-forward
	[triplets.pop(c) for c in failed_conditions]

	# Report if requested
	if inpt.verbose is True:
		print('Triplets:')
		[print(t) for t in triplets]
		print(nTriplets)
	return triplets


# Format index list
def formatIndexList(triplets,datePairs):
	'''
		# Create a n x 3 list of indices representing the slices in a data cube
		#  Col [0] = IJ index, Col [1] = JK index, Col [2] = IK index
	'''
	# Define basic parameters locally
	nTriplets=len(triplets)
	nDatePairs=len(datePairs)
	indices=np.arange(nDatePairs)
	ndxList=np.zeros((nTriplets,3))

	# Loop through each triplet
	for t in range(nTriplets):
		triplet=triplets[t] # examine each triplet
		IJ=triplet[0] # first pair in triplet
		JK=triplet[1] # second pair in triplet
		IK=triplet[2] # third pair in triplet
		IJndx=indices[(datePairs[:,0]==IJ[0]) & (datePairs[:,1]==IJ[1])]
		JKndx=indices[(datePairs[:,0]==JK[0]) & (datePairs[:,1]==JK[1])]
		IKndx=indices[(datePairs[:,0]==IK[0]) & (datePairs[:,1]==IK[1])]
		ndxList[t,:]=np.array([IJndx,JKndx,IKndx]).astype(int).squeeze(1)
	return ndxList

# Calculate misclosure
def calcMisclosure(ndxList,ifgCube,inpt):
	nTriplets=ndxList.shape[0] # number of triplets
	if inpt.verbose is True:
		print('Data cube shape: {}'.format(ifgCube.shape))

	Misclosure=np.zeros((nTriplets,ifgCube.shape[1],ifgCube.shape[2]))

	# Loop through triplets 
	for t in range(nTriplets):
		# IJ + JK - IK
		IJndx=ndxList[t][0]
		JKndx=ndxList[t][1]
		IKndx=ndxList[t][2]

		# Data maps
		IJ=ifgCube[IJndx,:,:] # IJ phase
		JK=ifgCube[JKndx,:,:] # JK phase
		IK=ifgCube[IKndx,:,:] # IK phase

		# # Reference points
		# px,py=inpt.refXY
		# IJ-=IJ[py,px]
		# JK-=JK[py,px]
		# IK-=IK[py,px]

		Misclosure[t,:,:]=IJ+JK-IK

	return Misclosure


# Plot map
def plotMeanMisclosure(img,cmap='viridis',ref=None,limits=None):
	F=plt.figure()
	img=np.ma.array(img,mask=(img==img[0,0]))
	if limits is None:
		stats=imgStats(img,pctmin=2,pctmax=98)
		limits=(stats.vmin,stats.vmax)
	ax=F.add_subplot(111)
	cax=ax.imshow(img,
		vmin=limits[0], vmax=limits[1], cmap=cmap)
	F.colorbar(cax,orientation='horizontal')
	ax.set_title('Mean misclosure')
	if ref:
		ax.plot(ref[0],ref[1],'ks')
	return F

# Plot imagettes
def plotImagettes(dataCube,normalization='pct'):
	M=3
	N=4
	totalFigs=M*N

	nTriplets=dataCube.shape[0]

	f=1
	for t in range(nTriplets):
		if f%totalFigs==1:
			F=plt.figure() # spawn new figure
			f=1 # reset counter
		if normalization in ['pct']:
			stats=imgStats(dataCube[t,:,:],pctmin=2,pctmax=98)
			limits=(stats.vmin,stats.vmax)
		elif normalization in ['pi']:
			limits=(-np.pi,np.pi)
		ax=F.add_subplot(M,N,f)
		cax=ax.imshow(dataCube[t,:,:],
			vmin=limits[0],vmax=limits[1], cmap='jet')
		F.colorbar(cax,orientation='horizontal')
		ax.set_xticks([]); ax.set_yticks([])
		ax.set_title('{}'.format(t))
		f+=1 # update counter

## Plot misclosure points
# Misclosure values
def misclosureTimePts(event):
	# Location
	px=event.xdata; py=event.ydata
	px=int(px); py=int(py)

	# Mean misclosure
	meanValue=MeanMisclosure[py,px]
	print('px {} py {} = (mean) {}'.format(px,py,meanValue))

	## Misclosure as a function of time
	# Misclosure data
	misclosure_pts=Misclosure[:,py,px]

	# Trend of misclosure data
	if inpt.trend in ['linear']:
		# Linear trend
		trend=fit_linear_trend(dates,misclosure_pts) # use linear trend fit
		t_finescale=np.linspace(dates.min(),dates.max(),1000)
		trend.reconstruct(t_finescale)
		if inpt.verbose is True:
			print('\tTrend--Linear: {:.3f}'.format(trend.A))
	elif inpt.trend in ['periodic']:
		# Periodic + linear trend
		trend=fit_periodic_trend(dates,misclosure_pts) # use periodic trend fit
		t_finescale=np.linspace(dates.min(),dates.max(),1000)
		trend.reconstruct(t_finescale)
		if inpt.verbose is True:
			print('\tTrend--Periodic: {:.3f}; Linear: {:.3f}'.format(trend.P,trend.C))

	# Misclosure plot
	axMisclosure.cla()
	axMisclosure.plot(misclosure_pts,'-ko',zorder=3)
	axMisclosure.axhline(0,color=(0.65,0.65,0.65),zorder=1)
	axMisclosure.set_ylabel('misclosure (rad)')
	if inpt.trend is not None:
		axMisclosure.plot(t_finescale,trend.yhat,color=(0.3,0.4,0.7),zorder=2)

	## Cumulative misclosure as a function of time
	# Cum. misclosure data
	cum_misclosure_pts=np.cumsum(misclosure_pts)

	# Cum. misclosure plot
	axCumMisclosure.cla()
	axCumMisclosure.plot(cum_misclosure_pts,'-ko',zorder=3)
	axCumMisclosure.axhline(0,color=(0.65,0.65,0.65),zorder=1)
	axCumMisclosure.set_ylabel('cum. miscl. (rad)')
	axCumMisclosure.set_xlabel('years since {}'.format(refDate))

	F_misclosure.canvas.draw()


# --- Linear trend ---
# Combination of sinusoids, linear trend, and offset
#  A*t + B
class fit_linear_trend:
	def __init__(self,t,y):
		# Basic parameters
		n=len(t) # number of data points
		self.t=t 

		# Design matrix
		#  Gb = y
		G=np.ones((n,2)) # all ones
		G[:,0]=t 
		self.G=G # save design matrix

		# Invert for parameters
		#  b = Ginv y ~= (GTG)-1 GT y
		self.beta=np.linalg.inv(np.dot(G.T,G)).dot(G.T).dot(y)
		self.A=self.beta[0] # linear slope
		self.B=self.beta[1] # DC offset

	def reconstruct(self,t=None):
		if t is not None:
			# Use new timepoints
			n=len(t) # new length
			G=np.ones((n,2)) # reconstruct design matrix
			G[:,0]=t 
		else:
			# Use old timepoints
			G=self.G

		# Estimate timeseries
		self.yhat=np.dot(G,self.beta)

# --- Periodic trend ---
# Combination of sinusoids, linear trend, and offset
#  A*cos(t) + B*sin(t) + C*t + D
class fit_periodic_trend:
	def __init__(self,t,y):
		# Basic parameters
		n=len(t) # number of data points
		self.t=t 

		# Design matrix
		#  Gb = y
		G=np.ones((n,4)) # all ones
		G[:,0]=np.cos(2*np.pi*t) 
		G[:,1]=np.sin(2*np.pi*t)
		G[:,2]=t 
		self.G=G # save design matrix

		# Invert for parameters
		#  b = Ginv y ~= (GTG)-1 GT y
		self.beta=np.linalg.inv(np.dot(G.T,G)).dot(G.T).dot(y)
		self.A=self.beta[0] # cosine coefficient
		self.B=self.beta[1] # sine coefficient
		self.C=self.beta[2] # linear slope
		self.D=self.beta[3] # DC offset
		self.P=np.sqrt(self.A**2+self.B**2) # magnitude of periodic signal

	def reconstruct(self,t=None):
		if t is not None:
			# Use new timepoints
			n=len(t) # new length
			G=np.ones((n,4)) # reconstruct design matrix
			G[:,0]=np.cos(2*np.pi*t) 
			G[:,1]=np.sin(2*np.pi*t)
			G[:,2]=t 
		else:
			# Use old timepoints
			G=self.G

		# Estimate timeseries
		self.yhat=np.dot(G,self.beta)


### --- Main function ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load HDF5 dataset
	DS=h5py.File(inpt.fname,'r')
	if inpt.verbose is True:
		print(DS.keys())

	if inpt.plotInputs is True:
		plotImagettes(DS[inpt.subDS])

	#plotImagettes(DS['connectComponent'])

	dates,datePairs=formatDates(DS['date'])
	triplets=formatTriplets(datePairs,inpt)

	# Remove invalid dates
	if inpt.toss:
		toss=[int(i) for i in inpt.toss.split()] # convert to integers
		triplets=[t for i,t in enumerate(triplets) if i not in toss]	

	# Reference date
	refDate=dates[0]
	if inpt.verbose is True:
		print('Reference date: {}'.format(refDate))


	## Calculate phase misclosure
	# Format index List
	ndxList=formatIndexList(triplets,datePairs)

	# Report basic info if requested
	if inpt.verbose is True:
		nDates=len(dates)
		nDatePairs=datePairs.shape[0]
		nTriplets=len(triplets)
		print('Nb unqiue dates: {}'.format(nDates))
		print('Nb interferograms: {}'.format(nDatePairs))
		print('Nb triplets: {}'.format(nTriplets))

	# Calculate misclosure
	Misclosure=calcMisclosure(ndxList,DS[inpt.subDS],inpt)

	MeanMisclosure=np.mean(Misclosure,axis=0)

	if inpt.plotMisclosure is True:
		plotImagettes(Misclosure,normalization='pi')
	F_mean_misclosure=plotMeanMisclosure(MeanMisclosure,cmap='jet',ref=inpt.refXY)

	# Misclosure time points
	dates=np.array(dates)
	F_misclosure=plt.figure()
	axMisclosure=F_misclosure.add_subplot(211); axCumMisclosure=F_misclosure.add_subplot(212)
	F_misclosure.suptitle('Phase misclosure over time')

	if inpt.trend is not None:
		inpt.trend=inpt.trend.lower()
	F_mean_misclosure.canvas.mpl_connect('button_press_event',misclosureTimePts)


	# ## Save to file if requested
	# if inpt.outName:
	# 	outName='Mean_Misclosure_{}'.format(inpt.outName)
	# 	if inpt.outType.lower() in ['gtiff','tiff','tif']:
	# 		# Save mean misclosure map
	# 		outName=outName+'.tif'
	# 		driver=gdal.GetDriverByName('GTiff')
	# 		OutMap=driver.Create(outName,N,M,1,gdal.GDT_Float32)
	# 		OutMap.GetRasterBand(1).WriteArray(MeanMisclosure)
	# 		OutMap.SetProjection(proj)
	# 		OutMap.SetGeoTransform(tnsf)
	# 		OutMap.FlushCache()
	# 	elif inpt.outType.lower() in ['hdf5','h5']:
	# 		outName=outName+'.h5'
	# 		OutMap=h5py.File(outName,'w')
	# 		OutData=OutMap.create_dataset("MeanMisclosure",(M,N))
	# 		OutData[:,:]=MeanMisclosure
	# 	print('Saved: {}'.format(outName))


	plt.show()