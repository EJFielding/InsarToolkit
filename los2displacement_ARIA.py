#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 
from osgeo import gdal
from InsarFormatting import *
from aria2LOS import *


### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Convert LOS to ground displacement.')
	# Required
	parser.add_argument(dest='netcdfList',type=str,nargs='+',help='List of NETCDF files')
	parser.add_argument('-v','--verbose',dest='verbose',action='store_true',help='Verbose mode')
	return parser 

def cmdParser(inpt_args=None):
	parser = createParser()
	return parser.parse_args(inpt_args)


### --- Processing functions ---
def invertForDisp():
	pass



### --- Main function ---
if __name__=="__main__":
	inpt=cmdParser()

	## Handle NETCDF files
	# How many files?
	N=len(inpt.netcdfList) # total number of files
	if N<2:
		print('ERROR: Only one interferogram detected.')
		exit()

	if inpt.verbose is True:
		print('{} files given:'.format(N))
		for i in range(N):
			print('{}'.format(inpt.netcdfList[i]))

	# How many ascending vs descending?
	#  Inversion will still be performed, but warning will be displayed
	#   if not ascending and descending
	trackDirection=[] # empty list

	# Loop through file names and check
	for file in inpt.netcdfList:
		# Parse ARIA filename
		filename=ARIAname(file)
		trackDirection.append(filename.orient)
	nAsc=trackDirection.count('A') # count nb ascending
	nDsc=trackDirection.count('D') # count nb descending
	if nAsc<1:
		print('WARNING: No ascending interferograms detected.')
	elif nDsc<1:
		print('WARNING: No descending interferograms detected.')

	## What type of data