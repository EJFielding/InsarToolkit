#!/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import mode
import h5py
from dateFormatting import daysBetween
from viewingFunctions import imgBackground
from mathFunctions import fit_periodic_trend


### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Fit trends to pixels in data cube.')
	# Data cube file
	parser.add_argument(dest='fName', type=str, help='HDF5 data cube')

	# Masking
	parser.add_argument('-modes','--modes', dest='nModes', type=int, default=0, help='Number of secondary modes to mask [default = 1]')

	# Outputs
	parser.add_argument('-v','--verbose',dest='verbose', action='store_true', help='Verbose mode')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### --- Main function ---
if __name__=='__main__':
	## Gather arguments
	inpt=cmdParser()


	## Load data cube
	DS=h5py.File(inpt.fName,'r')

	# Parse data sets
	dates=list(DS['date'][:].astype(int))
	U=DS['timeseries'][:,:,:].astype(float)
	D,M,N=U.shape

	if inpt.verbose is True:
		print('Loaded: {}'.format(inpt.fName))
		print(DS.keys())
		print('Dates: {}'.format(dates))
		print('Timeseries array size: {}'.format(U.shape))


	## Calculate time since beginning
	refTime=dates[0] 
	times=np.array([daysBetween(refTime,date,absTime=False)/365.25 for date in dates])


	## Masking
	Utot=U[-1,:,:] # use top slice = total displacement
	mask=np.ones((M,N))

	# Based on background value
	bg=imgBackground(Utot)
	mask[Utot==bg]=0

	# Based on other modes
	while inpt.nModes > 0:
			modeVal,count=mode(Utot[mask==1],axis=None)
			mask[Utot==modeVal]=0
			
			if inpt.verbose is True:
				print('secondary modes: {} ({} pixels)'.format(modeVal,count))

			inpt.nModes-=1

	# Based on percent clip
	vmin,vmax=np.percentile(Utot[mask==1],(2,98))
	mask[Utot<vmin]=0; mask[Utot>vmax]=0

	# Final masked array
	Utot=np.ma.array(Utot,mask=(mask==0))

	Mask=mask.reshape(1,M,N)
	Mask=np.repeat(Mask,D,axis=0)
	U=np.ma.array(U,mask=(Mask==0))


	## Plot topmost layer of data cube
	Fmap=plt.figure()
	axMap=Fmap.add_subplot(111)
	caxMap=axMap.imshow(Utot,
		cmap='jet',vmin=vmin,vmax=vmax,zorder=1)
	Fmap.colorbar(caxMap,orientation='vertical')
	axMap.set_title('Total displacement')

	## Loop through each pixel and calculate the ratio of periodic vs linear
	P=np.zeros((M,N))
	L=np.zeros((M,N))
	R=np.zeros((M,N)) # empty array of ratio values

	for i in range(M):
		for j in range(N):
			u=U[:,i,j]
			trend=fit_periodic_trend(times,u)

			if trend.linear!=0:
				p=trend.periodic # periodic coefficient
				l=np.abs(trend.linear) # linear coefficient

				P[i,j]=p 
				L[i,j]=l 
				R[i,j]=p/l 


	P=np.ma.array(P,mask=(mask==0))
	pmin,pmax=np.percentile(P,(2,98))
	Fperiodic=plt.figure()
	axPeriodic=Fperiodic.add_subplot(111)
	caxPeriodic=axPeriodic.imshow(P,cmap='jet',vmin=pmin,vmax=pmax)
	Fperiodic.colorbar(caxPeriodic,orientation='vertical')
	axPeriodic.set_title('Periodic')


	L=np.ma.array(L,mask=(mask==0))
	lmin,lmax=np.percentile(L,(2,98))
	Flinear=plt.figure()
	axLinear=Flinear.add_subplot(111)
	caxLinear=axLinear.imshow(L,cmap='jet',vmin=lmin,vmax=lmax)
	Flinear.colorbar(caxLinear,orientation='vertical')
	axLinear.set_title('Linear')



	R=np.ma.array(R,mask=(mask==0))
	rmin,rmax=np.percentile(R,(2,95))
	Fratio=plt.figure()
	axRatio=Fratio.add_subplot(111)
	caxRatio=axRatio.imshow(R,cmap='jet',vmin=rmin,vmax=rmax)
	Fratio.colorbar(caxRatio,orientation='vertical')
	axRatio.set_title('Ratio P/L')


	plt.show()