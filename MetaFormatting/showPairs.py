#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# by Rob Zinke 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from glob import glob
from generalFormatting import listUnique

def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='List the interferogram date pairs given a folder of ARIA products, interferograms, or a MintPy HDF5 file.')
	parser.add_argument('-f','--fldr','--folder', dest='Fldr', type=str, default='.', help='Folder with files.')
	parser.add_argument('-n','--name', dest='fileName', type=str, default=None, help='Single file name')
	parser.add_argument('-e','--ext', dest='extension', type=str, default=None, help='File extension')
	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name')

	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


if __name__=='__main__':
	## Script inputs 
	inpt=cmdParser()

	## Locate files
	searchString=inpt.Fldr
	if inpt.fileName:
		searchString=os.path.join(searchString,inpt.fileName)
	if inpt.extension:
		extension=inpt.extension
		searchString=os.path.join(searchString,inpt.extension)
	else:
		extension=searchString.split('.')[-1]

	if inpt.verbose is True:
		print('Search string: {}'.format(searchString))

	# Return search results
	files=glob(searchString)
	nFiles=len(files)

	if inpt.verbose is True:
		print('Files: {}'.format(files))
		print('Files detected: {}'.format(nFiles))
		print('Extension: {}'.format(extension))

	## Create list of date pairs
	# Handle by extension
	if extension in ['nc']:
		# NETCDF
		print('Cannot yet interpret NETCDF'); exit()
	elif extension in ['h5']:
		# HDF5
		print('Cannot yet interpret HDF5'); exit()
	else:
		# Use basenames for filenames
		files=[fileName.strip('.'+extension) for fileName in files]
		datePairs=[os.path.basename(fileName) for fileName in files]

	# Find unique pairs
	nPairs_all=len(datePairs)
	datePairs=listUnique(datePairs)
	nPairs=len(datePairs)

	if inpt.verbose is True:
		print('Unique date pairs: {}'.format(datePairs))
		print('Date pairs detected: {}'.format(nPairs_all))
		print('Unique date pairs detected: {}'.format(nPairs))

	# Save to file if requested
	if inpt.outName:
		outName='{}.txt'.format(inpt.outName)
		with open(outName,'w') as outFile:
			for pair in datePairs:
				outFile.write('{}\n'.format(pair))
			outFile.close()
		if inpt.verbose is True:
			print('List saved to: {}'.format(outName))
