#!/usr/bin/env python3
import os
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import h5py
from ImageTools import imgStats
from PhaseAnalysisHDF import *


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

# N-x pairs
def makePairs(dates,x,validPairs=None):
	nDates=len(dates)
	pairs=[]
	for i in range(0,nDates-x):
		NxPair=[dates[i],dates[i+x]]
		print('pair',NxPair)
		pairs.append(NxPair)
	
	# Check against existing pairs if list given
	if validPairs:
		


### --- Main ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load HDF5 dataset
	DS=h5py.File(inpt.fname,'r')
	if inpt.verbose is True:
		print(DS.keys())

	# Dates
	dates, datePairs=formatDates(DS['date'])

	# List pairs
	n1_pairs=makePairs(dates,1,validPairs=datePairs)