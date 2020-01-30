#!/usr/bin/env python3 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get a list of dates and the average time of
#  acquisition from list of ARIA GUNW IFGs
# 
# by Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from glob import glob
from nameFormatting import ARIAname
from generalFormatting import listUnique
from dateFormatting import avgTime


### --- PARSER --- 
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Plot most types of Insar products, including complex images and multiband images')

	# Required
	parser.add_argument(dest='files', type=str, help='ARIA GUNW IFGS from which to retrieve dates/times')

	# Options
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name')

	return parser


def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### --- MAIN FUNCTION ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	# Gather file names
	fnames=glob(inpt.files) # detect relevant files
	nFiles=len(fnames) # number of files

	# Parse filenames
	products=[ARIAname(fname) for fname in fnames]

	# Parse dates
	refDates=[product.RefDate for product in products] # refernce date list
	secDates=[product.SecDate for product in products] # secondary date list
	allDates=refDates+secDates # list of all dates
	uDates=listUnique(allDates) # list unique dates
	nDates=len(uDates) # number of unique dates

	# Parse times
	allTimes=[product.time for product in products] # central times
	maxTime=max(allTimes); minTime=min(allTimes)
	meanTime=avgTime(allTimes) # find average time
	timeHrs=meanTime.h+meanTime.m/60+meanTime.s/3600 # fraction hours

	# Print if requested
	if inpt.verbose is True:
		print('Unique dates: {}'.format(uDates))
		print('Files: {}'.format(nFiles))
		print('Dates: {}'.format(nDates))
		print('Time - max: {}; min: {}'.format(maxTime,minTime))
		print('Mean time: {}'.format(meanTime.time))
		print('Mean time (hrs): {}'.format(timeHrs))

	# Save to file if requested
	if inpt.outName:
		outName='{}.txt'.format(inpt.outName)
		with open(outName,'w') as Fout:
			for date in uDates:
				Fout.write('{}\n'.format(date))
			Fout.close()
		print('Saved to: {}'.format(outName))
