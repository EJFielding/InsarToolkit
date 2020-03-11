#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For a given set of interferograms, compute the cumulative phase
#  misclosure.
# 
# Rob Zinke 2020
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


### IMPORT MODULES ---
import os
from glob import glob
import numpy as np 
import matplotlib.pyplot as plt 
import h5py
from datetime import datetime
# InsarToolkit modules
from dateFormatting import udatesFromPairs, createTriplets
from viewingFunctions import mapStats, imagettes


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compute the cumulative misclosure of phase triplets based on a \
		set of interferograms saved in the MintPy HDF5 data stucture.\nThis program assumes that all files are \
		coregistered and cover the same area.')

	# Input data
	parser.add_argument(dest='dataset', type=str, help='Name of folders, files, or HDF5 dataset')
	parser.add_argument('-t','--dataType', dest='dataType', type=str, default=None, help='(Recommended) Manually specify data type (ARIA, ISCE, MintPy, [None])')
	parser.add_argument('-s','--subDS', dest='subDS', type=str, default='unwrapPhase', help='Sub-data set [e.g., unwrapPhase_phaseClosure, default=unwrapPhase]')

	# Triplet formulation
	parser.add_argument('-l','--lags', dest='lags', type=int, default=1, help='Number of lags, e.g., 2 lags = [n1-n0, n2-n1, n2-n0]')
	parser.add_argument('--mintime','--min-time', dest='minTime', type=str, default=None, help='Minimum time span of pairs in triplets (days)')
	parser.add_argument('--maxtime','--max-time', dest='maxTime', type=str, default=None, help='Maximum time span of pairs in triplets (days)')

	# Reference point
	parser.add_argument('-refX', dest='refX', default='auto', help='Reference X pixel')
	parser.add_argument('-refY', dest='refY', default='auto', help='Reference Y pixel')

	# Vocalization
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('--print-files', dest='printFiles', action='store_true', help='Print list of detected files')
	parser.add_argument('--print-dates', dest='printDates', action='store_true', help='Print list of unique dates')

	# Plots
	parser.add_argument('--plot-pairs', dest='plotPairs', action='store_true', help='Plot interferogram pairs in schematic form')
	parser.add_argument('--plot-inputs', dest='plotInputs', action='store_true', help='Plot input interferograms')
	parser.add_argument('--plot-triplets', dest='plotTriplets', action='store_true', help='Plot triplets and misclosure.')

	# Misclosure map formatting
	parser.add_argument('--misclosure-limits', dest='miscLims', type=float, nargs=2, default=None, help='Cumulative misclosure plot color limits')
	parser.add_argument('--abs-misclosure-limit', dest='absMiscLim', type=float, default=None, help='Absolute cumulative misclosure plot color limits')

	return parser


def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### DATA CLASS ---
## Class containing input data and dates, etc.
class dataSet:
	def __init__(self):
		self.ifgs=[]
		self.pairs=[]
		self.dates=[]



### LOAD DATA ---
## Detect data type
def detectDataType(inpt):
	# Search for specified data files
	inpt.files=glob(inpt.dataset)
	if inpt.printFiles is True: print('Files detected:\n{}'.format(inpt.files))

	# Summarize file detection
	inpt.nFiles=len(inpt.files)
	if inpt.verbose is True: print('{} files detected'.format(inpt.nFiles))

	# Determine data type (ARIA, ISCE, MintPy)
	if inpt.dataType:
		assert inpt.dataType.lower() not in ['aria','isce','mintpy'], \
			'Data type not found. Must be ARIA, ISCE, MintPy'
	else:
		# Auto-detect data type
		if inpt.dataset[-3:]=='.h5': 
			inpt.dataType='HDF5'
		elif inpt.dataset[-4:]=='.vrt':
			inpt.dataType='ARIA'
		else:
			# Check if files are directories
			try:
				os.listdir(inpt.files[0])
				inpt.dataType='ISCE'
			except:
				print('Data type unrecognized. Please specify explicitly')

	# Report if requested
	if inpt.verbose is True: print('Data type: {}'.format(inpt.dataType))


## Load data as 3D array
def loadData(inpt):
	if inpt.verbose is True: print('Loading data...')

	# Detect data type (ARIA, ISCE, MintPy)
	detectDataType(inpt)

	# Load data based on filetype
	if inpt.dataType=='ARIA':
		data=loadARIA(inpt)
	elif inpt.dataType=='ISCE':
		data=loadISCE(inpt)
	elif inpt.dataType=='MintPy':
		data=loadMintPy(inpt)
	else:
		print('No data loaded'); exit()

	return data


## Load ARIA data set
def loadARIA(inpt):
	from osgeo import gdal
	data=dataSet()

	# Loop through to load each file and append to list
	for file in inpt.files:
		# Load using gdal
		DS=gdal.Open(file,gdal.GA_ReadOnly)

		# Append IFG to list
		data.ifgs.append(DS.GetRasterBand(1).ReadAsArray())

		# Append pair name to list
		fname=os.path.basename(file)
		fname=fname.split('.')[0] # remove extension
		pairName=fname.split('_')[::-1] # reverse for older-younger
		data.pairs.append(pairName)

	# Convert IFG list to array
	data.ifgs=np.array(data.ifgs)
	data.M,data.N=data.ifgs[0,:,:].shape # map dimensions
	if inpt.verbose is True: print('Data array shape: {}'.format(data.ifgs.shape))

	# Unique dates from pairs
	data.dates=udatesFromPairs(data.pairs,verbose=inpt.verbose); data.dates.sort()
	if inpt.printDates is True: print('Dates: {}'.format(data.dates))

	return data


## Load ISCE data set
def loadISCE(inpt):
	from osgeo import gdal
	data=dataSet()

	# Loop through to load each file and append to list
	for file in inpt.files:
		# Load using gdal
		ifgName='{}/filt_fine.unw.vrt'.format(file) # format name
		DS=gdal.Open(ifgName,gdal.GA_ReadOnly) # open

		# Append IFG to list
		data.ifgs.append(DS.GetRasterBand(1).ReadAsArray())

		# Append pair name to list
		fname=os.path.basename(file)
		pairName=fname.split('_')
		data.pairs.append(pairName)

	# Convert IFG list to array
	data.ifgs=np.array(data.ifgs)
	data.M,data.N=data.ifgs[0,:,:].shape # map dimensions
	if inpt.verbose is True: print('Data array shape: {}'.format(data.ifgs.shape))

	# Unique dates from pairs
	data.dates=udatesFromPairs(data.pairs,verbose=inpt.verbose); data.dates.sort()
	if inpt.printDates is True: print('Dates: {}'.format(data.dates))

	return data


## Load MintPy data set
def loadMintPy(inpt):
	import h5py

	return data



### CALCULATE MISCLOSURE ---
def calcMisclosure(inpt,data):
	if inpt.verbose is True: print('Calculating misclosure...')

	# Empty placeholders
	data.miscStack=[]
	data.absMiscStack=[]

	for triplet in inpt.triplets:
		print(triplet)
		# Triplet date pairs
		IJdates=triplet[0]
		JKdates=triplet[1]
		IKdates=triplet[2]

		# Triplet ifg indices
		IJndx=data.pairs.index(IJdates)
		JKndx=data.pairs.index(JKdates)
		IKndx=data.pairs.index(IKdates)

		# Interferograms
		IJ=data.ifgs[IJndx,:,:]
		JK=data.ifgs[JKndx,:,:]
		IK=data.ifgs[IKndx,:,:]

		# Normalize to reference point
		IJ-=IJ[inpt.refY,inpt.refX]
		JK-=JK[inpt.refY,inpt.refX]
		IK-=IK[inpt.refY,inpt.refX]

		# Compute misclosure
		misclosure=IJ+JK-IK
		absMisclosure=np.abs(misclosure)

		# Plot if requested
		if inpt.plotTriplets:
			tripletFig=plotTriplets(IJ,JK,IK,misclosure,inpt.refX,inpt.refY)
			tripletFig.suptitle(triplet)

		# Append to stack
		data.miscStack.append(misclosure)
		data.absMiscStack.append(absMisclosure)

	# Convert lists to 3D arrays
	data.miscStack=np.array(data.miscStack)
	data.absMiscStack=np.array(data.absMiscStack)

	# Cumulative misclosure
	data.cumMisclosure=np.sum(data.miscStack,axis=0)
	data.cumAbsMisclosure=np.sum(data.absMiscStack,axis=0)



### PLOTTING FUNCTIONS ---
## Plot triplets
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
		if background is not None:
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


## Plot misclosure
def plotMisclosure(inpt,data):
	## Map stats - clip to inner 96% unless specified
	cumMiscStats=mapStats(data.cumMisclosure,pctmin=2,pctmax=98)
	if inpt.miscLims: # Specified limits
		cumMiscStats.vmin=inpt.miscLims[0]; cumMiscStats.vmax=inpt.miscLims[1]
	cumAbsMiscStats=mapStats(data.cumAbsMisclosure,pctmin=2,pctmax=98)
	if inpt.absMiscLim: # Specified upper limit
		cumAbsMiscStats.vmin=0; cumAbsMiscStats.vmax=inpt.absMiscLim


	## Plot cumulative misclosure
	title='Cumulative misclosure ({} triplets)'.format(inpt.nTriplets)
	figCumMiscMap=plt.figure(); axCumMiscMap=figCumMiscMap.add_subplot(111)
	cumMiscMap=misclosureMap(figCumMiscMap,axCumMiscMap,data.cumMisclosure,inpt.refX,inpt.refY,title=title,
		background=cumMiscStats.background, vmin=cumMiscStats.vmin,vmax=cumMiscStats.vmax)
	cumMiscMap.plotax()


	## Plot cumulative absolute misclosure
	title='Cumulative absolute misclosure ({} triplets)'.format(inpt.nTriplets)
	figCumAbsMiscMap=plt.figure(); axCumAbsMiscMap=figCumAbsMiscMap.add_subplot(111)
	cumAbsMiscMap=misclosureMap(figCumAbsMiscMap,axCumAbsMiscMap,data.cumAbsMisclosure,inpt.refX,inpt.refY,title=title,
		background=cumAbsMiscStats.background, vmin=cumAbsMiscStats.vmin,vmax=cumAbsMiscStats.vmax)
	cumAbsMiscMap.plotax()

	return cumMiscMap, cumAbsMiscMap



### MAIN CALL ---
if __name__=='__main__':
	## Gather arguments
	inpt=cmdParser()

	## Load data based on data type
	data=loadData(inpt)
	if inpt.plotPairs is True:
		from viewingFunctions import plotDatePairs
		plotDatePairs(data.pairs)

	# Plot data if requested
	if inpt.plotInputs is True:
		imagettes(data.ifgs,4,5,cmap='jet',downsampleFactor=3,pctmin=2,pctmax=98,background='auto',	titleList=data.pairs,supTitle=None)


	## Formulate valid triplets
	# List of triplets based on all dates
	inpt.triplets=createTriplets(data.dates,
		lags=inpt.lags,minTime=inpt.minTime,maxTime=inpt.maxTime,
		verbose=inpt.verbose)

	# Validate triplets against list of pairs
	validTriplets=[]
	for triplet in inpt.triplets:
		count=0
		for pair in triplet:
			if pair in data.pairs: count+=1
		if count==3: validTriplets.append(triplet)
	inpt.triplets=validTriplets
	inpt.nTriplets=len(inpt.triplets)

	if inpt.verbose is True:
		print('{} triplets validated'.format(inpt.nTriplets))


	## Detect reference point
	if inpt.refY is 'auto' or inpt.refX is 'auto':
		inpt.refY=np.random.randint(0,data.M,1)[0]
		inpt.refX=np.random.randint(0,data.N,1)[0]
	else:
		inpt.refX=int(inpt.refX); inpt.refY=int(inpt.refY)

	if inpt.verbose is True: print('Reference points: Y {}; X {}'.format(inpt.refY,inpt.refX))
	

	## Calculate misclosure
	calcMisclosure(inpt,data)

	# Plot misclosure
	cumMiscMap,cumAbsMiscMap=plotMisclosure(inpt,data)


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