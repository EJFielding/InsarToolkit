#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute the cumulative phase change for 
#  different pairs of interferograms
# 
# Rob Zinke 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import h5py
from dateFormatting import formatHDFdates, cumulativeTime
from listPairs import listPairs, plotPairs
import numpy as np
from viewingFunctions import mapPlot, imagettes
import matplotlib.pyplot as plt
from SaveHDF import saveHDF


### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compare differences in phase velocity from HDF5 files.')
	# Folder with dates
	parser.add_argument('-f', dest='filename', type=str, required=True, help='Folder with products')
	parser.add_argument('-d', dest='subDataset', type=str, help='Name of sub-data set')
	# Date/time criteria
	parser.add_argument('--start-date',dest='startDate', type=int, default=None, help='Start date of acquisitions')
	#parser.add_argument('--min-time',dest='minTime', type=float, help='Minimum amount of time (years) between acquisitions in a pair')
	#parser.add_argument('--max-time',dest='maxTime', type=float, help='Maximum amount of time (years) between acquisitions in a pair')
	# Ancillary plotting
	parser.add_argument('--plotInputs', dest='plotInputs', action='store_true', help='Plot inputs')
	parser.add_argument('--plotPairs', dest='plotPairs', action='store_true', help='Plot pairs')
	# Outputs
	parser.add_argument('-v','--verbose',dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('-o','--outName',dest='outName', type=str, default=None, help='Output name')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Main function ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load and format files 
	DS=h5py.File(inpt.filename,'r')
	PhaseCube=DS[inpt.subDataset]

	# List of dates
	dates=DS['date']
	dates,datePairs=formatHDFdates(dates,verbose=inpt.verbose)
	nDatePairs=len(datePairs) # number of interferograms

	# Sort by start date
	if inpt.startDate:
		datePairsCropped=[list(pair) for pair in datePairs if min(pair)>=inpt.startDate]
		nDatePairsCropped=len(datePairsCropped)
		validIndices=[ndx for ndx in range(nDatePairs) if list(datePairs[ndx]) in datePairsCropped]
		dates=[date for date in dates if date>=inpt.startDate]
		datePairs=datePairs[validIndices]
		PhaseCube=PhaseCube[validIndices]
		nDatePairs=len(datePairs)
		if inpt.verbose is True:
			print('Valid indices',validIndices)
			print('Unique dates:',dates)
			print('Date Pairs:',datePairs)
			print('Nb valid date pairs: {}'.format(nDatePairs))

	# Create lists of pairs
	n_n1_pairs=listPairs(dates,1,method='ordered',pairOrder='OldNew',validDates=datePairs,verbose=inpt.verbose)
	n_n2_pairs=listPairs(dates,2,method='ordered',pairOrder='OldNew',validDates=datePairs,verbose=inpt.verbose)

	n_y1_pairs=listPairs(dates,1,method='interval',interval=365,pairOrder='OldNew',validDates=datePairs,verbose=inpt.verbose)
	n_y2_pairs=listPairs(dates,2,method='interval',interval=365,pairOrder='OldNew',validDates=datePairs,verbose=inpt.verbose)

	# Construct list of indices pointing to slices for each list of pairs
	n_n1_indices=[ndx for ndx in range(nDatePairs) if list(datePairs[ndx]) in n_n1_pairs]; Nn1=len(n_n1_indices)
	n_n2_indices=[ndx for ndx in range(nDatePairs) if list(datePairs[ndx]) in n_n2_pairs]; Nn2=len(n_n2_indices)

	n_y1_indices=[ndx for ndx in range(nDatePairs) if list(datePairs[ndx]) in n_y1_pairs]; Ny1=len(n_y1_indices)
	n_y2_indices=[ndx for ndx in range(nDatePairs) if list(datePairs[ndx]) in n_y2_pairs]; Ny2=len(n_y2_indices)

	# Report if requested
	if inpt.verbose is True:
		print('N-N1 indices: {}'.format(Nn1))
		print('N-N2 indices: {}'.format(Nn2))
		print('N-Y1 indices: {}'.format(Ny1))
		print('N-Y2 indices: {}'.format(Ny2))

	# Plot pairs if requested
	if inpt.plotPairs is True:
		plotPairs(n_n1_pairs,title='n-n+1 pairs')
		plotPairs(n_n2_pairs,title='n-n+2 pairs',refDate=np.min(n_n1_pairs[0]))
		plotPairs(n_y1_pairs,title='n-yr+1 pairs',refDate=np.min(n_n1_pairs[0]))
		plotPairs(n_y2_pairs,title='n-yr+2 pairs',refDate=np.min(n_n1_pairs[0]))

	## Compute average phase velocity for each set of pairs
	# Create data cubes with the layers corresponding to n-n+X pairs
	n1_cube=PhaseCube[n_n1_indices,:,:]
	n2_cube=PhaseCube[n_n2_indices,:,:]
	y1_cube=PhaseCube[n_y1_indices,:,:]
	y2_cube=PhaseCube[n_y2_indices,:,:]

	if inpt.plotInputs is True:
		imagettes(n1_cube,mRows=3,nCols=4,cmap='jet',background='auto',pctmin=2,pctmax=98,supTitle='N-N1 pairs')


	# Compute cumulative time
	n1_cumTime=cumulativeTime(n_n1_pairs)
	n2_cumTime=cumulativeTime(n_n2_pairs)
	y1_cumTime=cumulativeTime(n_y1_pairs)
	y2_cumTime=cumulativeTime(n_y2_pairs)


	# Cumulatively sum layers in datacube
	n1_cumCube=np.cumsum(n1_cube,axis=0)
	n2_cumCube=np.cumsum(n2_cube,axis=0)
	y1_cumCube=np.cumsum(y1_cube,axis=0)
	y2_cumCube=np.cumsum(y2_cube,axis=0)


	## Save outputs if requested
	if inpt.outName:
		# Save n-n1 pairs
		outData={}
		outName='{}_nn1_pairs.h5'.format(inpt.outName)
		outData['datePairs']=n_n1_pairs
		outData['cumulativeTime']=n1_cumTime
		outData['phase']=n1_cube
		outData['cumulativePhase']=n1_cumCube
		saveHDF(outName,outData,verbose=inpt.verbose)

		# Save n-n2 pairs
		outData={}
		outName='{}_nn2_pairs.h5'.format(inpt.outName)
		outData['datePairs']=n_n2_pairs
		outData['cumulativeTime']=n2_cumTime
		outData['phase']=n2_cube
		outData['cumulativePhase']=n2_cumCube
		saveHDF(outName,outData,verbose=inpt.verbose)

		# Save n-yr1 pairs
		outData={}
		outName='{}_nyr1_pairs.h5'.format(inpt.outName)
		outData['datePairs']=n_y1_pairs
		outData['cumulativeTime']=y1_cumTime
		outData['phase']=y1_cube
		outData['cumulativePhase']=y1_cumCube
		saveHDF(outName,outData,verbose=inpt.verbose)

		# Save n-yr2 pairs
		outData={}
		outName='{}_nyr2_pairs.h5'.format(inpt.outName)
		outData['datePairs']=n_y2_pairs
		outData['cumulativeTime']=y2_cumTime
		outData['phase']=y2_cube
		outData['cumulativePhase']=y2_cumCube
		saveHDF(outName,outData,verbose=inpt.verbose)



	plt.show()