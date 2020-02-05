#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For a given set of interferograms, compute the cumulative phase
#  misclosure.
# 
# Rob Zinke 2020
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


### IMPORT MODULES ---
import numpy as np 
import matplotlib.pyplot as plt 
import h5py
from datetime import datetime
# InsarToolkit modules
from dateFormatting import formatHDFdates, createTriplets
from viewingFunctions import mapStats, imagettes


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compute the cumulative misclosure of phase triplets based on a \
		set of interferograms saved in the MintPy HDF5 data stucture.')

	# Input data
	parser.add_argument(dest='dataset', type=str, help='Name of MintPy .h5 data set.')
	parser.add_argument('-s','--subDS', dest='subDS', type=str, default='unwrapPhase', help='Sub-data set [e.g., unwrapPhase_phaseClosure, default=unwrapPhase]')
	parser.add_argument('-x','--excl','--exclude', dest='excl', type=str, default=None, help='List of date pairs to exclude [e.g., ')

	# Reference point
	parser.add_argument('-refX', dest='refX', default='auto', help='Reference X pixel')
	parser.add_argument('-refY', dest='refY', default='auto', help='Reference Y pixel')
	
	# Basic outputs
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('--plotInputs', dest='pltInputs', action='store_true', help='Plot input interferograms')
	parser.add_argument('--plotTriplets', dest='pltTriplets', action='store_true', help='Plot valid phase triplets')

	# Misclosure map formatting
	parser.add_argument('--plotMisclosure', dest='pltMisclosure', action='store_true', help='Plot cumulative misclosure maps')
	parser.add_argument('--misclosure-limits', dest='miscLims', type=float, nargs=2, default=None, help='Cumulative misclosure plot color limits')
	parser.add_argument('--abs-misclosure-limit', dest='absMiscLim', type=float, default=None, help='Absolute cumulative misclosure plot color limits')

	# Misclosure analysis
	parser.add_argument('-a','--analysis', dest='analysis', action='store_true', help='Misclosure timeseries analysis')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
def plotTriplets(IJ,JK,IK,misclosure,refX,refY):
	# Create figure and axes
	tripletFig=plt.figure()

	# Map stats
	IJstats=mapStats(IJ,pctmin=2,pctmax=98)
	JKstats=mapStats(JK,pctmin=2,pctmax=98)
	IKstats=mapStats(IK,pctmin=2,pctmax=98)
	misclosureStats=mapStats(misclosure,pctmin=2,pctmax=98)

	# Plot maps
	cbar_orient='horizontal'
	axIJ=tripletFig.add_subplot(221)
	caxIJ=axIJ.imshow(np.ma.array(IJ,mask=(IJ==IJstats.background)),
		cmap='jet',vmin=IJstats.vmin,vmax=IJstats.vmax)
	tripletFig.colorbar(caxIJ,orientation=cbar_orient)
	axIJ.plot(refX,refY,'ks')

	axJK=tripletFig.add_subplot(222)
	caxJK=axJK.imshow(np.ma.array(JK,mask=(JK==JKstats.background)),
		cmap='jet',vmin=IJstats.vmin,vmax=IJstats.vmax)
	tripletFig.colorbar(caxJK,orientation=cbar_orient)
	axJK.plot(refX,refY,'ks')

	axIK=tripletFig.add_subplot(223)
	caxIK=axIK.imshow(np.ma.array(IK,mask=(IK==IKstats.background)),
		cmap='jet',vmin=IJstats.vmin,vmax=IJstats.vmax)
	tripletFig.colorbar(caxIK,orientation=cbar_orient)
	axIK.plot(refX,refY,'ks')

	axMisclosure=tripletFig.add_subplot(224)
	caxMisclosure=axMisclosure.imshow(np.ma.array(misclosure,mask=(misclosure==misclosureStats.background)),
		cmap='jet',vmin=misclosureStats.vmin,vmax=misclosureStats.vmax)
	tripletFig.colorbar(caxMisclosure,orientation=cbar_orient)
	axMisclosure.plot(refX,refY,'ks')

	# Formatting
	axIJ.set_xticks([]); axIJ.set_yticks([]); axIJ.set_title('I_J')
	axJK.set_xticks([]); axJK.set_yticks([]); axJK.set_title('J_K')
	axIK.set_xticks([]); axIK.set_yticks([]); axIK.set_title('I_K')
	axMisclosure.set_xticks([]); axMisclosure.set_yticks([])
	axMisclosure.set_title('Misclosure')

	return tripletFig


## Plot cumulative misclosure
class misclosureMap:
	def __init__(self,Fig,ax,img,refX,refY,title=None,background=None,vmin=None,vmax=None):
		self.Fig=Fig; self.ax=ax
		self.img=img
		self.refX=refX; self.refY=refY
		self.title=title
		self.background=background
		self.vmin=vmin
		self.vmax=vmax
		self.cbar=None # format colorbar only once

		# Mask background if specified
		if background:
			self.img=np.ma.array(img,mask=(img==background))

	def plotax(self):
		# Plot
		cax=self.ax.imshow(self.img,vmin=self.vmin,vmax=self.vmax,cmap='jet')
		self.ax.plot(self.refX,self.refY,'ks')
		self.ax.set_xticks([]); self.ax.set_yticks([])
		if self.title:
			self.ax.set_title(self.title)
		# Format colorbar once only
		if not self.cbar:
			self.cbar=self.Fig.colorbar(cax,orientation='horizontal')
			self.cbar.set_label('radians')


## Plot series
def plotSeries(name,ax,series,timeAxis=False):
	ax.plot(cumTime,series,'-k.')

	ax.set_ylabel(name+'\nradians')
	if timeAxis is False:
		ax.set_xticks([])
	else:
		ax.set_xlabel('time (yrs)')


## Analyze misclosure stack
def analyzeStack(event):
	print('Stack analysis')

	# Location
	px=event.xdata; py=event.ydata
	px=int(round(px)); py=int(round(py))

	# Report position and cumulative values
	print('px {} py {}'.format(px,py)) # report position
	print('Cumulative misclosure: {}'.format(cumMisclosure[py,px]))
	print('Abs cumulative misclosure: {}'.format(cumAbsMisclosure[py,px]))

	# Plot query points on maps
	cumMiscMap.ax.cla(); cumMiscMap.plotax() # clear and replot map
	cumMiscMap.ax.plot(px,py,color='k',marker='o',markerfacecolor='w',zorder=3)

	cumAbsMiscMap.ax.cla(); cumAbsMiscMap.plotax()
	cumAbsMiscMap.ax.plot(px,py,color='k',marker='o',markerfacecolor='w',zorder=3)

	# Timeseries
	# Plot misclosure over time
	print('Misclosure: {}'.format(misclosureStack[:,py,px,]))
	miscSeriesAx.cla() # misclosure
	plotSeries('misclosure',miscSeriesAx,misclosureStack[:,py,px])
	cumMiscSeriesAx.cla() # cumulative misclosure
	plotSeries('cum. miscl.',cumMiscSeriesAx,np.cumsum(misclosureStack[:,py,px]))
	absMiscSeriesAx.cla() # absolute misclosure
	plotSeries('(abs. miscl.)',absMiscSeriesAx,absMiscStack[:,py,px])
	cumAbsMiscSeriesAx.cla() # cumulative absolute misclosure
	plotSeries('(cum. abs. miscl.)',cumAbsMiscSeriesAx,np.cumsum(absMiscStack[:,py,px]),
		timeAxis=True)
	miscSeriesFig.tight_layout()

	# Draw outcomes
	cumMiscMap.Fig.canvas.draw()
	cumAbsMiscMap.Fig.canvas.draw()
	miscSeriesFig.canvas.draw()



### CALCULATE MISCLOSURE ---
def calcMisclosure(inpt):
	## Loading
	# Load HDF5 data set
	with h5py.File(inpt.dataset,'r') as dataset:
		if inpt.verbose is True:
			print(dataset.keys())

		# Format dates
		dates,datePairs=formatHDFdates(dataset['date'],verbose=inpt.verbose)

		# Interferograms
		phsCube=dataset[inpt.subDS][:] # default is unwrapped phase (no corrections)
		P,M,N=phsCube.shape

		# Close dataset
		dataset.close()


	## Triplet formulation
	# Formulate triplets
	allDates=np.array(dates) # convert to numpy array for sorting
	allDates=np.sort(dates) # sort smallest-> largest
	triplets=createTriplets(allDates,verbose=inpt.verbose)
	pairList=np.ndarray.tolist(datePairs) # convert numpy array to list for matching

	# Exclude date pairs from pair list -- CAUTION! The pair list will diverge
	#  from the datePairs list here.
	excludePairs=[pair.split('_') for pair in inpt.excl.split(' ')]
	excludePairs=[[int(pair[0]), int(pair[1])] for pair in excludePairs] # convert values to integers

	includeTriplets=[]
	for triplet in triplets:
		c=0
		# Count number of exclude pairs in each triplet
		for excludePair in excludePairs: c+=triplet.count(excludePair)
		if c==0: includeTriplets.append(triplet)
	triplets=includeTriplets; del includeTriplets # reassign name

	if inpt.verbose is True:
		print('Excluded triplets with pairs: {}'.format(excludePairs))

	# Find triplets for which all interferograms exist
	validTriplets=[] # emtpy list to be populated by triplets for which ifgs exist
	tripletIndices=[]
	for triplet in triplets:
		# Separate dates
		IJdates=triplet[0] # first pair of dates
		JKdates=triplet[1] # second pair of dates
		IKdates=triplet[2] # third pair of dates

		# Find index of date pair
		try:
			IJndx=pairList.index(IJdates)
			JKndx=pairList.index(JKdates)
			IKndx=pairList.index(IKdates)
			tripletIndices.append([IJndx,JKndx,IKndx]) # add indices to list if all indices found
			validTriplets.append(triplet) # add triplet to list if all indices found
		except:
			pass
	nValidTriplets=len(validTriplets); inpt.nValidTriplets=nValidTriplets

	# Report if requested
	if inpt.verbose is True:
		print('Valid triplets:')
		[print(triplet) for triplet in validTriplets]
		print('{} valid triplets'.format(nValidTriplets))


	# Plot inputs if requested
	if inpt.pltInputs is True:
		imagettes(phsCube,4,4,cmap='jet',downsampleFactor=1,pctmin=2,pctmax=98,background='auto',
		titleList=pairList,supTitle=None)


	## Compute misclosure
	misclosureStack=[] # misclosure
	absMiscStack=[] # absolute value of misclosure
	for t in range(nValidTriplets):
		indices=tripletIndices[t]
		IJ=phsCube[indices[0],:,:]
		JK=phsCube[indices[1],:,:]
		IK=phsCube[indices[2],:,:]

		# Zero at reference point
		if inpt.refY is 'auto' or inpt.refX is 'auto':
			inpt.refY=np.random.randint(0,M,1)
			inpt.refX=np.random.randint(0,N,1)
		else:
			inpt.refX=int(inpt.refX); inpt.refY=int(inpt.refY)
			IJ-=IJ[inpt.refY,inpt.refX]
			JK-=JK[inpt.refY,inpt.refX]
			IK-=IK[inpt.refY,inpt.refX]

		if inpt.verbose is True:
			print('Calculating misclosure. Ref X pixel: {}; Ref Y pixel: {}'.format(inpt.refX,inpt.refY))

		# Calculate misclosure
		misclosure=IJ+JK-IK
		absMisclosure=np.abs(misclosure)

		if inpt.pltTriplets:
			tripletFig=plotTriplets(IJ,JK,IK,misclosure,inpt.refX,inpt.refY)
			tripletFig.suptitle(validTriplets[t])

		misclosureStack.append(misclosure)
		absMiscStack.append(absMisclosure) # append to stack


	# Convert lists to 3D arrays
	misclosureStack=np.array(misclosureStack)
	absMiscStack=np.array(absMiscStack)

	# Cumulative misclosure
	cumMisclosure=np.sum(misclosureStack,axis=0)
	cumAbsMisclosure=np.sum(absMiscStack,axis=0)


	## Determine cumulative time since beginning of series
	startDate=datetime.strptime(str(validTriplets[0][0][0]),"%Y%m%d")
	lag0dates=[datetime.strptime(str(triplet[0][0]),"%Y%m%d") for triplet in validTriplets] # grab first date from each triplet
	cumTime=[lag0date-startDate for lag0date in lag0dates]
	cumTime=[time.days/365.25 for time in cumTime]

	# Report if requested
	if inpt.verbose is True:
		print('Series start date: {}'.format(startDate))
		print('Cumulative time: {}'.format(cumTime))


	return misclosureStack, absMiscStack, cumMisclosure, cumAbsMisclosure, cumTime



### PLOT MISCLOSURE ---
# Plot cumulative misclosure and cum abs misclosure
def plotMisclosure(cumMisclosure, cumAbsMisclosure, inpt):
	## Map stats - clip to inner 96% unless specified
	cumMiscStats=mapStats(cumMisclosure,pctmin=2,pctmax=98)
	if inpt.miscLims: # Specified limits
		cumMiscStats.vmin=inpt.miscLims[0]; cumMiscStats.vmax=inpt.miscLims[1]
	cumAbsMiscStats=mapStats(cumAbsMisclosure,pctmin=2,pctmax=98)
	if inpt.absMiscLim: # Specified upper limit
		cumAbsMiscStats.vmin=0; cumAbsMiscStats.vmax=inpt.absMiscLim


	## Plot cumulative misclosure
	title='Cumulative misclosure ({} triplets)'.format(inpt.nValidTriplets)
	figCumMiscMap=plt.figure(); axCumMiscMap=figCumMiscMap.add_subplot(111)
	cumMiscMap=misclosureMap(figCumMiscMap,axCumMiscMap,cumMisclosure,inpt.refX,inpt.refY,title=title,
		background=cumMiscStats.background, vmin=cumMiscStats.vmin,vmax=cumMiscStats.vmax)
	cumMiscMap.plotax()


	## Plot cumulative absolute misclosure
	title='Cumulative absolute misclosure ({} triplets)'.format(inpt.nValidTriplets)
	figCumAbsMiscMap=plt.figure(); axCumAbsMiscMap=figCumAbsMiscMap.add_subplot(111)
	cumAbsMiscMap=misclosureMap(figCumAbsMiscMap,axCumAbsMiscMap,cumAbsMisclosure,inpt.refX,inpt.refY,title=title,
		background=cumAbsMiscStats.background, vmin=cumAbsMiscStats.vmin,vmax=cumAbsMiscStats.vmax)
	cumAbsMiscMap.plotax()

	return cumMiscMap, cumAbsMiscMap



### MAIN CALL ---
if __name__=='__main__':
	## Gather arguments
	inpt=cmdParser()

	## Calculate misclosure
	misclosureStack, absMiscStack, cumMisclosure, cumAbsMisclosure, cumTime=calcMisclosure(inpt)

	# Plot and analyze misclosure
	if inpt.pltMisclosure is True:
		## Plot misclosure
		cumMiscMap, cumAbsMiscMap=plotMisclosure(cumMisclosure, cumAbsMisclosure, inpt)


		## Misclosure analysis
		# Spawn misclosure figure
		miscSeriesFig=plt.figure('Misclosure')
		miscSeriesAx=miscSeriesFig.add_subplot(411)
		cumMiscSeriesAx=miscSeriesFig.add_subplot(412)
		absMiscSeriesAx=miscSeriesFig.add_subplot(413)
		cumAbsMiscSeriesAx=miscSeriesFig.add_subplot(414)

		cumMiscMap.Fig.canvas.mpl_connect('button_press_event',analyzeStack)
		cumAbsMiscMap.Fig.canvas.mpl_connect('button_press_event',analyzeStack)


	plt.show()