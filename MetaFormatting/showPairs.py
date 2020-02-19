#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# by Rob Zinke 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from glob import glob
from datetime import datetime

def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='List the interferogram date pairs given a folder of ARIA products, interferograms, or a MintPy HDF5 file.')
	# File arguments
	parser.add_argument(dest='searchStr', type=str, help='Search string, e.g., \"*.nc\"')
	parser.add_argument('-e','--ext','--extension', dest='extension', type=str, default=None, help='File extension (optional)')

	# Date arguments
	parser.add_argument('--start-date', dest='startDate', type=int, default=None, help='Earliest valid date')
	parser.add_argument('--end-date', dest='endDate', type=int, default=None, help='Latest valid date')
	parser.add_argument('--min-interval', dest='minIntvl', type=int, default=None, help='Minimum time interval')
	parser.add_argument('--max-interval', dest='maxIntvl', type=int, default=None, help='Maximum time interval')
	parser.add_argument('--months', dest='months', type=int, default=None, nargs='+', help='Specify allowable reference date months')

	# Outputs
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot pairs')
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)

### Files class ---
class Files:
	def __init__(self):
		pass


### Ancillary functions ---
# Find most common item in list
def mostFrequent(List): 
	counter=0
	common_item=List[0]
	for item in List:
		curr_frequency=List.count(item)
		if(curr_frequency>counter):
			counter=curr_frequency
			common_item=item
	return item



### Analysis functions ---
## Locate files
def locateFiles(inpt):
	# Files in folder
	files=Files() # instantiate object
	files.files=glob(inpt.searchStr)
	files.nFiles=len(files.files)

	# Report if requested
	if inpt.verbose is True:
		print('{} files detected'.format(files.nFiles))

	# Find common file extension
	if inpt.extension:
		if inpt.verbose is True: print('Extension: {}'.format(inpt.extension))
	else:
		# Isolate file extensions
		exts=[fname.split('.')[-1] for fname in files.files]
		# Find most common file name
		inpt.extension=mostFrequent(exts)
		if inpt.verbose is True: print('Automatically detecting extension: {}'.format(inpt.extension))
	return files


## Find pairs of dates
def findDatePairs(inpt,files):
	if inpt.extension=='nc':
		# Extract dates from filename
		from nameFormatting import ARIAname
		files.pairs=[]
		files.intvls=[]

		for fname in files.files:
			# Extract date information
			name=ARIAname(fname)
			refDate=datetime.strptime(name.RefDate, '%Y%m%d')
			secDate=datetime.strptime(name.SecDate, '%Y%m%d')

			date1=str(refDate.date()).replace('-','')
			date2=str(secDate.date()).replace('-','')

			pair='{}_{}'.format(date1,date2)

			# Calculate time between dates
			time_intvl=refDate-secDate

			# Add to list only if unique
			if pair not in files.pairs:
				files.pairs.append(pair)
				files.intvls.append(str(time_intvl.days))

			# Report if requested
			if inpt.verbose is True:
				print('{}: {} days'.format(pair,time_intvl.days))


## Select pairs by attributes
def selectPairs(inpt,files):
	# Start date
	if inpt.startDate:
		secDates=[int(pair.split('_')[1]) for pair in files.pairs]
		selectPairs=[files.pairs[ndx] for ndx,date in enumerate(secDates) if (date>inpt.startDate)]
		selectIntvls=[files.intvls[ndx] for ndx,date in enumerate(secDates) if (date>inpt.startDate)]

		files.pairs=selectPairs; del selectPairs
		files.intvls=selectIntvls; del selectIntvls

	# End date
	if inpt.endDate:
		refDates=[int(pair.split('_')[0]) for pair in files.pairs]
		selectPairs=[files.pairs[ndx] for ndx,date in enumerate(refDates) if (date<inpt.endDate)]
		selectIntvls=[files.intvls[ndx] for ndx,date in enumerate(refDates) if (date<inpt.endDate)]

		files.pairs=selectPairs; del selectPairs
		files.intvls=selectIntvls; del selectIntvls

	# Allowable months
	if inpt.months:
		refDates=[pair.split('_')[0] for pair in files.pairs]
		selectPairs=[files.pairs[ndx] for ndx,date in enumerate(refDates) if (int(date[4:6]) in inpt.months)]
		selectIntvls=[files.intvls[ndx] for ndx,date in enumerate(refDates) if (int(date[4:6]) in inpt.months)]

		files.pairs=selectPairs; del selectPairs
		files.intvls=selectIntvls; del selectIntvls

	# Minimum interval
	if inpt.minIntvl:
		intvls=[int(intvl) for intvl in files.intvls]
		selectPairs=[files.pairs[ndx] for ndx,intvl in enumerate(intvls) if (intvl>=inpt.minIntvl)]
		selectIntvls=[files.intvls[ndx] for ndx,intvl in enumerate(intvls) if (intvl>=inpt.minIntvl)]
		
		files.pairs=selectPairs; del selectPairs
		files.intvls=selectIntvls; del selectIntvls

	# Maximum interval
	if inpt.maxIntvl:
		intvls=[int(intvl) for intvl in files.intvls]
		selectPairs=[files.pairs[ndx] for ndx,intvl in enumerate(intvls) if (intvl<=inpt.maxIntvl)]
		selectIntvls=[files.intvls[ndx] for ndx,intvl in enumerate(intvls) if (intvl<=inpt.maxIntvl)]

		files.pairs=selectPairs; del selectPairs
		files.intvls=selectIntvls; del selectIntvls


## Plot pairs
def plotPairs(files):
	# Format pairs into nested list
	#  [[ref, sec],
	#   [ref, sec]]
	pairs=[pair.split('_') for pair in files.pairs] # convert to nested list
	pairs=[[int(pair[0]),int(pair[1])] for pair in pairs] # convert to integers

	import matplotlib.pyplot as plt
	from viewingFunctions import plotDatePairs
	plotDatePairs(pairs)

	plt.show()



### Main ---
if __name__=='__main__':
	## Script inputs 
	inpt=cmdParser()

	## Locate files
	if inpt.searchStr[-2:] in ['h5']:
		# Treat HDF5 files differently from other types
		print('Does not yet work with HDF5 files'); exit
	else:
		# Detect files based on search string
		files=locateFiles(inpt)


	## Create list of date pairs
	findDatePairs(inpt,files)


	## Remove dates by time interval
	selectPairs(inpt,files)

	## Report final pairs and time intervals
	print('Final selection:')
	[print('{} {} days'.format(pair,files.intvls[ndx])) for ndx,pair in enumerate(files.pairs)]
	print('{} pairs'.format(len(files.pairs)))


	## Plot if requested
	if inpt.plot is True:
		plotPairs(files)


	# Save to file if requested
	if inpt.outName:
		outName='{}.txt'.format(inpt.outName)
		with open(outName,'w') as outFile:
			for pair in files.pairs:
				outFile.write('{}\n'.format(pair))
			outFile.close()
		if inpt.verbose is True:
			print('List saved to: {}'.format(outName))

