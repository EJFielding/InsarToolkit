#!/usr/bin/env python3

"""
	Provide a list of dates in format YYYYMMDD to create a list
	 of pairs based on specified criteria.

	Rob Zinke, 2020
"""

### Import essential modules ---
import os
from glob import glob
from datetime import datetime


### Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Provide a list of dates in format YYYYMMDD to create a list of pairs based on the specified criteria.')
	# File arguments
	parser.add_argument(dest='dateFile', type=str, help='Text file with dates listed, one per row, in format YYYYMMDD')

	# Date arguments
	parser.add_argument('--start-date', dest='startDate', type=int, default=None, help='Earliest valid date')
	parser.add_argument('--end-date', dest='endDate', type=int, default=None, help='Latest valid date')
	parser.add_argument('--months', dest='months', type=int, default=None, nargs='+', help='Specify allowable reference date months')

	# Pair arguments
	parser.add_argument('--min-interval', dest='minIntvl', type=int, default=6, help='Min interval in days (default = 6)')
	parser.add_argument('--max-interval', dest='maxIntvl', type=int, default=1000, help='Max interval in days (default = 1000)')
	parser.add_argument('-l','--lags', dest='lags', type=int, default=1, help='Number of lags')

	# Step arguments
	parser.add_argument('--step', dest='step', type=int, default=None, help='Minimum time span between adjacent pairs in days (default = None)')

	# Outputs
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot pairs')
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### Ancillary functions ---
## Load data
def loadDates(inpt):
	with open(inpt.dateFile,'r') as dateFile:
		# Read in data
		dates=dateFile.readlines()

		# Close file
		dateFile.close()

	# Sanity check
	dates=[date.strip('\n') for date in dates]
	bad_dates=[]
	for date in dates:
		if len(date)!=8: bad_dates.append(date)
		try:
			int(date)
		except:
			bad_dates.append(date)
	nBadDates=len(bad_dates)

	if nBadDates>0:
		print('{} bad dates detected: {}'.format(nBadDates,bad_dates))
		exit()
	else:
		if inpt.verbose is True:
			print('Input format validated. {} unique dates detected.'.format(len(dates)))

	# Order dates
	dates.sort()

	return dates


## Select pairs by attributes
def selectDates(inpt,dates):
	# Start date
	if inpt.startDate:
		selectDates=[dates[ndx] for ndx,date in enumerate(dates) if (int(date)>inpt.startDate)]
		dates=selectDates; del selectDates

	# End date
	if inpt.endDate:
		selectDates=[dates[ndx] for ndx,date in enumerate(dates) if (int(date)<inpt.endDate)]
		dates=selectDates; del selectDates

	# Months of the year
	if inpt.months:
		selectDates=[dates[ndx] for ndx,date in enumerate(dates) if (int(date[4:6]) in inpt.months)]
		dates=selectDates; del selectDates

	return dates


## Construct date pairs - earlier_later
def createPairs(inpt,dates):
	# Convert dates to datetime objects
	epochs=[datetime.strptime(date,'%Y%m%d') for date in dates]

	# Construct list of master dates
	if inpt.step:
		masters=[]; masters.append(epochs[0])
		# Master dates step by specified amount
		for epoch in epochs[1:]:
			curr_master=masters[-1]
			if (epoch-curr_master).days>inpt.step:
				masters.append(epoch)
	else:
		# No selection criteria applied to master image
		masters=epochs[:]

	# Loop through each date - earlier date will be the master
	pairs=[]
	for master in masters:
		# Find all dates that meet interval criteria
		intvls=[(epoch-master) for epoch in epochs]
		slaves=[epoch for ndx,epoch in enumerate(epochs) if (intvls[ndx].days>=inpt.minIntvl) and (intvls[ndx].days<=inpt.maxIntvl)]

		# Create-master slave pairs for the specified number of lags
		master=str(master.date()).replace('-','') # convert to string format
		for n in range(inpt.lags):
			try:
				# Test if sufficient number of slaves exist
				slave=slaves[n]
				slave=str(slave.date()).replace('-','')
				pair='{}_{}'.format(master,slave)
				pairs.append(pair)
			except:
				pass

	return pairs


## Plot pairs
def plotPairs(pairs):
	# Format pairs into nested list
	#  [[ref, sec],
	#   [ref, sec]]
	pairs=[pair.split('_') for pair in pairs] # convert to nested list
	pairs=[[int(pair[0]),int(pair[1])] for pair in pairs] # convert to integers

	import matplotlib.pyplot as plt
	from viewingFunctions import plotDatePairs
	plotDatePairs(pairs)

	plt.show()



### Main ---
if __name__=='__main__':
	## Script inputs 
	inpt=cmdParser()

	## Load list of dates
	dates=loadDates(inpt)

	## Trim date list based on specified criteria
	dates=selectDates(inpt,dates)

	# Print date list if requested
	if inpt.verbose is True:
		[print(date) for date in dates]
		print('{} valid dates'.format(len(dates)))


	## Construct pairs
	pairs=createPairs(inpt,dates)

	# Print pair list if requested
	if inpt.verbose is True:
		[print(pair) for pair in pairs]
		print('{} pairs created'.format(len(pairs)))


	## Plot if requested
	if inpt.plot is True:
		plotPairs(pairs)


	# Save to file if requested
	if inpt.outName:
		if inpt.outName[-4:]!='.txt': inpt.outName+='.txt'
		with open(inpt.outName,'w') as outFile:
			for pair in pairs:
				outFile.write('{}\n'.format(pair))
			outFile.close()
		if inpt.verbose is True:
			print('List saved to: {}'.format(inpt.outName))

