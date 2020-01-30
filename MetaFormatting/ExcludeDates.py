#!/usr/bin/env python3 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate a list of dates that are invalid
# 
# by Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from generalFormatting import listUnique


'''
	Generate a list of ifg dates that are not included 
	 in a list of "allowable" dates
'''

### --- PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Plot most types of Insar products, including complex images and multiband images')

	# Arguments
	parser.add_argument('-f','--fldr','--folder', dest='fldr', type=str, required=True, help='Folder where interferogram names are stored')
	parser.add_argument('-l','--dateList', dest='dateList', type=str, required=True, help='List of allowalbe dates')
	parser.add_argument('-o','--outName', dest='outName', type=str, default =None, help='Output base name')

	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- MAIN FUNCTION ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()


	## List of interferogram dates
	ifgs=glob(inpt.fldr)
	ifgs=[os.path.basename(ifg.split('.')[0]) for ifg in ifgs]

	allDates=[]
	[allDates.extend(ifg.split('_')) for ifg in ifgs]

	uniqueDates=listUnique(allDates)

	if inpt.verbose is True:
		print('Interferograms: {}'.format(ifgs))
		print('Unique dates: {}'.format(uniqueDates))

	## List of "allowable" dates
	with open(inpt.dateList,'r') as Fallowed:
		allowedDates=Fallowed.readlines()
		Fallowed.close()

	allowedDates=[allowedDate.strip('\n') for allowedDate in allowedDates]

	if inpt.verbose is True:
		print('Allowed dates: {}'.format(allowedDates))
		print('Nb ifg dates: {}'.format(len(uniqueDates)))
		print('Nb allowed dates: {}'.format(len(allowedDates)))

	invalidDates=[date for date in uniqueDates if date not in allowedDates]

	if inpt.verbose is True:
		print('Invalid dates: {}'.format(invalidDates))
		print('Nb invalid dates: {}'.format(len(invalidDates)))

	## Save to file
	if inpt.outName:
		outName='{}_invalid_dates.txt'.format(inpt.outName)
		with open(outName,'w') as Fout:
			for date in invalidDates:
				Fout.write('{}\n'.format(date))
			Fout.close()
			print('... Saved to: {}'.format(outName))
