#!/usr/bin/env python3
import os
import glob
import numpy as np
from datetime import *
from InsarFormatting import *

### --- Parser --- ###
def createParser():
	'''
		List pairs
	'''

	import argparse
	parser = argparse.ArgumentParser(description='Stack interferograms to average them.')
	# Folder with pair list
	parser.add_argument('-t','--type',dest='prodtype',type=str, defaut='exct', help='Type of product to analyze [\'extracted\' (only)]')
	# Folder with dates
	parser.add_argument('-f',dest='filelist', type=str, required=True, help='Folder with products')
	# Date/time criteria
	parser.add_argument('--min-time',dest=minTime, type=float, help='Minimum amount of time (years) between acquisitions in a pair')
	parser.add_argument('--max-time',dest=maxTime, type=float, help='Maximum amount of time (years) between acquisitions in a pair')
	# Outputs
	parser.add_argument('-v','--verbose',dest=verbose, action='store_true', help='Verbose mode')
	parser.add_argument('-o','--outName',dest=outName, type=str, default=None, help='Output name')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### --- Main function --- ###
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## List of available files