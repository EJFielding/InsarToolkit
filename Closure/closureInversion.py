#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use SBAS inversion to measure cumulative displacement
#  over time for closure analysis
# 
# Rob Zinke 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np 
import h5py
from dateFormatting import formatHDFdates, udatesFromPairs
from SBASrz import SBASrz

### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compare differences in phase velocity from HDF5 files.')
	# Folder with dates
	parser.add_argument('-f', dest='filename', required=True, type=str, help='HDF5 file')
	parser.add_argument('-d', dest='dataset', required=True, type=str, help='Unwrapped phase dataset')
	# Outputs
	parser.add_argument('-v','--verbose',dest='verbose', action='store_true', help='Verbose mode')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Main function ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	# Load data
	DS=h5py.File(inpt.filename,'r')
	if inpt.verbose is True:
		print('Keys: {}'.format(DS.keys()))
	dataStack=DS[inpt.dataset] # load stack of unwrapped interferograms
	try:
		datePairs=DS['datePairs']
	except:
		dates,datePairs=formatHDFdates(DS['date'])
		datePairs=[[datePairs[i][0],datePairs[i][1]] for i in range(datePairs.shape[0])]
		print('Formatting date pairs from MintPy convention')

	if inpt.verbose is True:
		print('Data stack dimensions: {}'.format(dataStack.shape))

	## Inversion
	SBASrz(dataStack[:,1000:1001,1500:1501],datePairs,verbose=inpt.verbose)