#!/usr/bin/env python3
import os
import h5py

### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Replace the x and y step values in a MintPy h5 file')
	parser.add_argument('-fpath','--fpath', dest='filePath', type=str, default='.', help='File path')
	parser.add_argument(dest='fileName', type=str, help='Name of h5 file')
	parser.add_argument('-x','--x-step','--xStep', dest='xStep', type=float, default=0.0025, help='X step size')
	parser.add_argument('-y','--y-step','--yStep', dest='yStep', type=float, default=-0.0025, help='Y step size')

	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### --- Main function ---
if __name__=='__main__':

	# Inputs 
	inpt=cmdParser()

	# Load data
	fname=os.path.join(inpt.filePath,inpt.fileName)

	DS=h5py.File(fname)

	if inpt.verbose is True:
		print(DS.attrs.keys())
		print(DS.attrs['Y_STEP'])
		print(DS.attrs['X_STEP'])

	# Modify attributes
	#DS.attrs.modify('Y_STEP',inpt.yStep)
	DS.attrs['Y_STEP']=inpt.yStep
	DS.attrs['X_STEP']=inpt.xStep
	DS.close()