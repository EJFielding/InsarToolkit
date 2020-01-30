#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic, general mathematical functions
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules 
import numpy as np 
from dateFormatting import udatesFromPairs, daysBetween

### --- Incidence matrix ---
def incidenceMatrix(timePairs,epochs,masterPos=0,slavePos=1,verbose=False):
	## Setup

	mObs=len(timePairs)
	nEpochs=len(epochs)

	## Incidence values
	A=np.zeros((mObs,nEpochs)) # empty incidence matrix

	# Loop through each interferogram
	for i in range(mObs):
		pair=timePairs[i] 
		master=pair[masterPos]
		slave=pair[slavePos]

		# Master date
		try:
			A[i,epochs==master]=-1
		except:
			pass
		# Slave date
		try:
			A[i,epochs==slave]=+1
		except:
			pass

	return A

### --- SBAS --- 
def SBASrz(dataStack,datePairs,refDate='firstDate',dateFmt="%Y%m%d",verbose=False):
	## Setup parameters
	nIFGs=dataStack.shape[0] # nb interferograms = first dimension of data stack
	M=dataStack.shape[1] # NS map dimension
	N=dataStack.shape[2] # EW map dimension

	# Specify convention
	masterPos=0
	slavePos=1

	# Pairs
	mObs=len(datePairs)
	assert mObs==nIFGs, 'Number of date pairs and interferograms must be same'

	# Find unique dates and place in order
	uniqueDates=udatesFromPairs(datePairs)
	uniqueDates.sort()

	# Determine reference date
	if refDate=='firstDate':
		refDate=uniqueDates[0]
	else:
		print('Changing refDate does not work for now!'); exit()

	# Determine epochs in decimal years for columns of matrix
	if dateFmt in ['decimal']:
		epochs=[uDate-refDate for uDate in uniqueDates]
	else:
		epochs=[daysBetween(uDate,refDate,fmt=dateFmt)/365.25 for uDate in uniqueDates] # time since reference date
	epochs.remove(refDate) # remove reference date from list 
	nEpochs=len(epochs)

	# Convert date pairs into time pairs
	if dateFmt in ['decimal']:
		timePairs=[[pair[0]-refDate,pair[1]-refDate] for pair in datePairs]
	else:
		timePairs=[[daysBetween(pair[0],refDate,fmt=dateFmt)/365.25,daysBetween(pair[1],refDate,fmt=dateFmt)/365.25] \
			for pair in datePairs]

	# Report setup if requested
	if verbose is True:
		print('Nb pairs: {}'.format(mObs))
		print('Nb unique dates (incl ref date): {}'.format(len(uniqueDates)))
		print('Reference date: {}'.format(refDate))
		print('Nb epochs (excl ref date): {}'.format(nEpochs))
		print('Constructing incidence matrix')

	## Solve for time series
	PhiHat=np.zeros((nEpochs,M,N))

	# Work pixel-by-pixel
	for m in range(M):
		for n in range(N):
			if verbose is True:
				print('Pixel (m n):',m,n)
			# Construct incidence matrix
			A=incidenceMatrix(timePairs,epochs,verbose=verbose)

			# Invert incidence matrix
			Ainv=np.linalg.inv(np.dot(A.T,A))

			# Reconstruct displacements
			PhiHat[:,m,n]=Ainv.dot(A.T).dot(dataStack[:,m,n])

	# Outputs
	uniqueDates.remove(refDate) # return list of dates
	return uniqueDates,PhiHat