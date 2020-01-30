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
# InsarToolkit modules
from dateFormatting import formatHDFdates
from viewingFunctions import imagettes


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compute the cumulative misclosure of phase triplets based on a \
		set of interferograms saved in the MintPy HDF5 data stucture.')

	parser.add_argument(dest='dataset', type=str, help='Name of MintPy .h5 data set.')
	parser.add_argument('-s','--subDS', dest='subDS', type=str, default='unwrapPhase', help='Sub-data set [e.g., unwrapPhase_phaseClosure, default=unwrapPhase]')
	
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('--plotInputs', dest='pltInputs', action='store_true', help='Plot input interferograms')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### MAIN FUNCTION ---
if __name__=='__main__':
	## Loading
	# Gather arguments
	inpt=cmdParser()

	# Load HDF5 data set
	with h5py.File(inpt.dataset,'r') as dataset:
		if inpt.verbose is True:
			print(dataset.keys())

		# Format dates
		dates,datePairs=formatHDFdates(dataset['date'],verbose=inpt.verbose)

		# Interferograms
		phsCube=dataset[inpt.subDS] # default is unwrapped phase (no corrections)

		if inpt.pltInputs is True:
			imagettes(phsCube,4,4,cmap='jet',downsampleFactor=3,pctmin=2,pctmax=98,background='auto',
			titleList=None,supTitle=None)


		## Analysis
		# Formulate triplets
		allDates=np.array(dates) # convert to numpy array for sorting
		allDates=np.sort(dates) # sort smallest-> largest

		print('All dates: {}'.format(allDates))
		

		
		dataset.close()

		plt.show()