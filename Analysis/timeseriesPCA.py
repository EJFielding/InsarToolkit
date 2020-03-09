#!/usr/bin/env python3

"""
	This script applies principal component analysis to
	 a MintPy timeseries.h5 data cube.
"""

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
import h5py
from viewingFunctions import imagettes, mapPlot
from datetime import datetime
from imagePCA import stackPCA


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compute the cumulative misclosure of phase triplets based on a \
		set of interferograms saved in the MintPy HDF5 data stucture.')

	# Input data
	parser.add_argument(dest='filename', type=str, help='Name of MintPy .h5 file.')

	# Outputs
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('--plot-inputs', dest='plotInputs', action='store_true', help='Plot inputs')
	parser.add_argument('--plot-retention', dest='plotRetention', action='store_true', help='Plot variance retention')
	parser.add_argument('--show-pcs', dest='showPCs', action='store_true', help='Show all PCS')
	parser.add_argument('--plot-pcs', dest='plotPCs', type=int, default=0, help='Specify which number of PCs to plot (e.g., 3)')


	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### FUNCTIONS ---
## Load data from MintPy HDF5 file
def loadData(inpt):
	if inpt.verbose is True: print('Loading MintPy file: {}'.format(inpt.filename))

	# Load displacement data cube from MintPy HDF5 data set
	with h5py.File(inpt.filename,'r') as tsFile:
		# Report dictionary keys
		if inpt.verbose is True: print(tsFile.keys())
		
		# Store timeseries data cube
		dates=tsFile['date'][:].astype(str)
		displacements=tsFile['timeseries'][:]

		# Remove first entry in list which is all zeros
		displacements=displacements[1:,:,:]

		# Report cube size
		if inpt.verbose is True:
			print('Dates: {}'.format(len(dates)))
			print('Displacements cube size: {}'.format(displacements.shape))

		tsFile.close()

	# Return data cube
	return dates,displacements



### MAIN ---
if __name__=='__main__':
	## Gather inputs
	inpt=cmdParser()


	## Load data
	dates,displacements=loadData(inpt)


	# Plot data if requested
	if inpt.plotInputs is True:
		imagettes(displacements.tolist(),4,5,cmap='viridis',downsampleFactor=3,pctmin=None,pctmax=None,background=0,supTitle=None)


	## Compute PCA
	PCs=stackPCA(displacements)


	## Outputs
	if inpt.plotRetention is True:
		PCs.plotRetention()

	if inpt.showPCs is True:
		PCs.plotPCs()

	if inpt.plotPCs > 0:
		for i in range(inpt.plotPCs):
			title='PC {}'.format(i+1)
			mapPlot(PCs.PCstack[i,:,:],cmap='viridis',pctmin=2,pctmax=98,background='auto',cbar_orientation='horizontal',title=title)



	plt.show()