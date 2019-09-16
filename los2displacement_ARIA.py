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
	parser.add_argument('--plot_inputs','--plot_inputs',dest='plot_inputs',action='store_true',help='Plot inputs')
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
	ascList=[]; dscList=[] # list of ascending and descending

	# Loop through file names and check
	for file in inpt.netcdfList:
		# Parse ARIA filename
		filename=ARIAname(file)
		trackDirection.append(filename.orient)
		if filename.orient=='A':
			ascList.append(file)
		elif filename.orient=='D':
			dscList.append(file)
	nAsc=trackDirection.count('A') # count nb ascending
	nDsc=trackDirection.count('D') # count nb descending
	if nAsc<1:
		print('WARNING: No ascending interferograms detected.')
	elif nDsc<1:
		print('WARNING: No descending interferograms detected.')

	## Load maps
	'''
	'''
	Phase={}
	LookAngle={}
	for file in inpt.netcdfList:
		PHS=gdal.Open('NETCDF:"{}":/science/grids/data/unwrappedPhase'.format(file))
		Phase[file]=PHS.GetRasterBand(1).ReadAsArray()
		Look=gdal.Open('NETCDF:"{}":/science/grids/imagingGeometry/lookAngle'.format(file))
		LookAngle[file]=Look.GetRasterBand(1).ReadAsArray()

	# Plot inputs if desired
	if inpt.plot_inputs is True:
		# Print ascending
		for i in range(nAsc):
			F=plt.figure('AscendingPhase{}'.format(i+1))
			# Plot phase
			ax=F.add_subplot(121)
			phs=Phase[ascList[i]]
			phs=np.ma.array(phs,mask=phs==0)
			cax=ax.imshow(phs,cmap='jet')
			F.colorbar(cax,orientation='horizontal')
			# Plot look angle
			ax=F.add_subplot(122)
			look=LookAngle[ascList[i]]
			cax=ax.imshow(look,cmap='jet')
			F.colorbar(cax,orientation='horizontal')
		# Print descending
		for i in range(nDsc):
			F=plt.figure('DescendingPhase{}'.format(i+1))
			ax=F.add_subplot(121)
			phs=Phase[dscList[i]]
			phs=np.ma.array(phs,mask=phs==0)
			cax=ax.imshow(phs,cmap='jet')
			F.colorbar(cax,orientation='horizontal')


	if inpt.plot_inputs is True:
		plt.show()