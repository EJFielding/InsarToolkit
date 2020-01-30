#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a list of date pairs
# 
# Rob Zinke 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from datetime import *
from dateFormatting import daysBetween
from generalFormatting import listUnique

### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Create a list of n-X pairs given a list of dates.')
	# Folder with pair list
	parser.add_argument('-t','--type',dest='prodtype',type=str, default='exct', help='Type of product to analyze [\'extracted\' (only)]')
	# Folder with dates
	parser.add_argument('-f',dest='filelist', type=str, required=True, help='Folder with products')
	# Pairing 
	parser.add_argument('-X', dest='X', type=int, required=True, help='Interval for pair (n+X)')
	parser.add_argument('-m','--pairing-method', dest='pairingMethod', type=str, default='ordered', help='Pairing method')
	parser.add_argument('--interval', dest='interval', type=float, default=0, help='Length of pairing interval in years')
	# Date/time criteria
	parser.add_argument('--min-time',dest='minTime', type=float, help='Minimum amount of time (years) between acquisitions in a pair')
	parser.add_argument('--max-time',dest='maxTime', type=float, help='Maximum amount of time (years) between acquisitions in a pair')
	# Outputs
	parser.add_argument('-v','--verbose',dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('-o','--outName',dest='outName', type=str, default=None, help='Output name')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Processing functions ---
## List pairs
def listPairs(dates,X,method='Ordered',interval=0,pairOrder='OldNew',minTime=None,maxTime=None,validDates=None,verbose=False):
	'''
		Create a list of pairs based on an ordered list of dates, or
		 based on the length of interval between dates. If method is
		 "interval" specify the desired interval in days.

		Specify the order dates in the output list ['OldNew' or 
		 'NewOld'].

		Optionally check against the minimum or maximum allowable
		 time range, or a list of "valid" existing dates.
	'''

	# Pre-formatting
	#  Ensure that dates are unique and ordered oldest-newest
	dates=listUnique(dates); dates.sort()

	# Setup parameters
	nDates=len(dates)
	if verbose is True:
		print('Generating n+{} pairs'.format(X))
		print('nb unique dates: {}'.format(nDates))

	## Create list of dates
	datePairs=[] # empty list of pairs

	# ... using "ordered" method
	if method.lower() in ['ordered']:
		for n in range(0,nDates-X):
			pair=[dates[n],dates[n+X]]
			if pairOrder=='NewOld':
				pair=pair[::-1]
			# Add pair to list
			datePairs.append(pair)
	# ... using "interval" method
	elif method.lower() in ['interval']:
		for n in range(nDates):
			# Calculate interval between current date and all other dates
			currentDate=dates[n] # use current date as reference
			otherDates=[date for date in dates if date!=currentDate]
			intervals=[daysBetween(currentDate,otherDate) for otherDate in otherDates]
			intervals=np.array([np.abs(i-interval) for i in intervals]) # relative to interval desired
			orderingIdx=np.argsort(intervals)
			otherDates_ordered=np.array(otherDates)[orderingIdx] # order other dates
			pair=[currentDate,otherDates_ordered[X-1]] # construct pair
			if pairOrder=='NewOld':
				pair=pair[::-1]
			# Add pair to list
			datePairs.append(pair)
	else:
		print('Invalid method for list generation'); exit()

	## Check against list of valid date pairs
	if validDates is not None:
		validDates=[list(dates) for dates in list(validDates)]
		invalidDates=[pair for pair in datePairs if pair not in validDates]
		# Remove invalid dates from datePairs
		[datePairs.remove(pair) for pair in invalidDates]

	## Outputs
	# Report if requested
	if verbose is True:
		print('Pairs:\n{}'.format(datePairs))

	return datePairs

## Plot pairs
def plotPairs(datePairs,refDate=None,title=None):
	nPairs=len(datePairs)
	y=1

	# Reference date
	if not refDate:
		refDate=np.min(datePairs[0])

	# Create figure
	F=plt.figure()
	ax=F.add_subplot(111)

	# Plot date spans
	for pair in datePairs:
		# Convert dates to time since reference date
		date1=daysBetween(pair[0],refDate)/365.25
		date2=daysBetween(pair[1],refDate)/365.25

		ax.plot([date1,date2],[y,y],'-k.',alpha=0.7)

		y+=1

	# Plot formatting
	ax.set_xlabel('time since {} (years)'.format(refDate))
	ax.set_yticks([])
	if title:
		ax.set_title(title)



### --- Main function ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## If filetype is HDF5 from MintPy
	if inpt.prodtype.lower() in ['h5','hdf','hdf5','mintpy']:
		import h5py
		from dateFormatting import formatHDFdates

		DS=h5py.File(inpt.filelist,'r')

		dates,datePairs=formatHDFdates(DS['date'],verbose=inpt.verbose)