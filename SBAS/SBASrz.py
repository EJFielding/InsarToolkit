#!/usr/bin/env python3
"""
	Provide a list of interferograms to compute the timeseries using
	 the short baseline subset (SBAS) approach, with or without
	 regularization.
"""

### IMPORT MODULES ---
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from osgeo import gdal
from viewingFunctions import mapPlot, imagettes


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compute the SBAS timeseries.')
	# Data sets
	parser.add_argument('-f','--files', dest='files', type=str, required=True, help='Files to be analyzed')
	parser.add_argument('-no-reg','--no-regularization', dest='regularization', action='store_false', help='Regularization ([True]/False)')

	# Reference point
	parser.add_argument('-refLaLo', dest='refLaLo', type=float, nargs=2, help='Reference lat/lon, e.g., 37.5 90.0')
	parser.add_argument('-refYX', dest='refYX', type=int, nargs=2, help='Reference Y/X, e.g., 45 119')
	parser.add_argument('--no-ref', dest='noRef', action='store_true', help='Do not use a reference point. Must be explicit.')

	# Plotting specs
	parser.add_argument('-pctmin', dest='pctmin', type=float, default=0, help='Min percent clip')
	parser.add_argument('-pctmax', dest='pctmax', type=float, default=100, help='Max percent clip')
	parser.add_argument('-bg','--background', dest='background', default=None, help='Background value')

	# Outputs
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('--plot-inputs', dest='plotInputs', action='store_true', help='Plot input interferograms')

	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name, for difference map and analysis plots')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### OBJECTS ---
## Object for passing parameters
class TSparams:
	def __init__(self):
		pass



### LOAD DATA ---
## Load data from geotiff files
def loadARIAdata(inpt):
	from glob import glob
	from geoFormatting import GDALtransform

	# Detect file names
	inpt.fnames=glob(inpt.files)

	# Empty lists
	stack=[] # data cube
	inpt.pairNames=[] # pair names
	inpt.pairs=[] # pairs as lists

	# Loop through files to load data
	for fname in inpt.fnames:
		# Print filename if requested
		if inpt.verbose is True:
			print('Loading: {}'.format(fname))

		# Add pair name to list
		pairName=os.path.basename(fname).split('.')[0]
		pairName=pairName[:17]
		inpt.pairNames.append(pairName)
		inpt.pairs.append(pairName.split('_'))

		# Load gdal data set
		DS=gdal.Open(fname,gdal.GA_ReadOnly)
		img=DS.GetRasterBand(1).ReadAsArray()

		# Add image to stack
		stack.append(img)

	# Grab spatial parameters from final map data set
	inpt.R=DS.RasterYSize; inpt.S=DS.RasterXSize
	inpt.Proj=DS.GetProjection()
	inpt.Tnsf=DS.GetGeoTransform()
	inpt.T=GDALtransform(DS)
	del DS

	# Convert stack to 3D array
	stack=np.array(stack)

	return stack



### REFERENCE POINT ---
## Format and apply reference point
def refPoint(inpt,stack):
	# Check that a refernce point is provided, or no-ref is specified
	if inpt.noRef is True:
		print('No reference point specified')
	else:
		assert inpt.refLaLo is not None or inpt.refYX is not None, 'Reference point must be specified'


		## Find reference
		# If reference point is given in lat lon, find equivalent pixels
		if inpt.refLaLo is not None:
			# Y pixel
			deltaLat=(inpt.refLaLo[0]-inpt.T.ystart)
			yref=int(0+deltaLat/inpt.T.ystep)
			# X pixel
			deltaLon=(inpt.refLaLo[1]-inpt.T.xstart)
			xref=int(0+deltaLon/inpt.T.xstep)
			# Ref La Lo
			inpt.refYX=[yref,xref]

		# If reference point is given in pixels, find equivalent lat lon
		if inpt.refLaLo is None:
			# Lat
			refLat=inpt.T.ystart+inpt.T.ystep*inpt.refYX[0]
			# Lon
			refLon=inpt.T.xstart+inpt.T.xstep*inpt.refYX[1]
			# Ref Y X
			inpt.refLaLo=[refLat,refLon]

		# Remove reference value from each map
		for k in range(stack.shape[0]):
			stack[k,:,:]=stack[k,:,:]-stack[k,inpt.refYX[0],inpt.refYX[1]]

	return stack



### SBAS COMPUTATION ---
## SBAS
class SBAS:
	def __init__(self,inpt,stack):
		# Basic parameters
		self.verbose=inpt.verbose

		# Spatial parameters
		if inpt.noRef is False:
			self.refY=inpt.refYX[0]
			self.refX=inpt.refYX[1]
			self.refLat=inpt.refLaLo[0]
			self.refLon=inpt.refLaLo[1]

		# List of epochs to solve for
		self.epochList(inpt)

		# Design matrix
		self.constructIncidenceMatrix(inpt)
		if inpt.regularization is True: self.regularizeMatrix()

		# Solve for displacements
		self.constructDisplacements(inpt,stack)


	# Epochs
	def epochList(self,inpt):
		# Number of interferograms
		self.M=len(inpt.pairs)

		# List of all dates, including redundant
		allDates=[]
		[allDates.extend(pair) for pair in inpt.pairs]

		# List of unique dates
		self.dates=[]
		[self.dates.append(date) for date in allDates if date not in self.dates]
		self.dates.sort() # sort oldest to youngest
		self.N=len(self.dates)

		# Reference date
		self.referenceDate=self.dates[0]

		# Epochs to solve for
		self.epochs=[datetime.strptime(date,'%Y%m%d') for date in self.dates]

		# Times since reference date
		self.times=[(epoch-self.epochs[0]).days/365.2422 for epoch in self.epochs]

		# Report if requested
		if self.verbose==True:
			print(self.dates)
			print('{} dates'.format(len(self.dates)))
			print('Reference date: {}'.format(self.referenceDate))


	# Incidence matrix
	def constructIncidenceMatrix(self,inpt):
		"""
			The incidence matrix A is an M x (N-1) matrix in which
			 the master date is represented by +1, and the slave
			 date is -1 if it is not the reference date.
		"""

		# Empty matrix of all zeros
		self.A=np.zeros((self.M,self.N-1))

		# Loop through pairs
		for i,pair in enumerate(inpt.pairs):
			masterDate=pair[0]
			slaveDate=pair[1]

			# Master date
			masterNdx=self.dates.index(masterDate) # index within date list
			self.A[i,masterNdx-1]=1 # Index-1 to ignore reference date

			# Slave date
			if slaveDate!=self.referenceDate:
				slaveNdx=self.dates.index(slaveDate) # index within date list
				self.A[i,slaveNdx-1]=-1 # Index-1 to ignore reference date


	# Regularization functions
	def regularizeMatrix(self):
		"""
			Regularization based on the linear model phs - v(tj-t1) - c = 0
		"""

		# Expand A matrix
		self.A=np.concatenate([self.A,np.zeros((self.M,2))],axis=1)
		self.A=np.concatenate([self.A,np.zeros((self.N-1,self.N-1+2))],axis=0)
		for i in range(self.N-1):
			self.A[self.M+i,i]=1
			self.A[self.M+i,-2]=-(self.epochs[i+1]-self.epochs[0]).days/365.2422
			self.A[self.M+i,-1]=-1

		# Report if requested
		if self.verbose==True: print('Enforcing regularization')


	# Solve for displacements
	def constructDisplacements(self,inpt,stack):
		## Setup
		# Invert design matrix
		Ainv=np.linalg.inv(np.dot(self.A.T,self.A)).dot(self.A.T)

		# Empty maps of solution values
		self.PHS=np.zeros((self.N,inpt.R,inpt.S)) # empty phase cube
		self.V=np.zeros((inpt.R,inpt.S))
		self.C=np.zeros((inpt.R,inpt.S))


		## Without regularization, use a linear fit to the data
		if inpt.regularization==False:
			for i in range(inpt.R):
				for j in range(inpt.S):
					# Interferogram values of pixel
					series=stack[:,i,j]
					series=series.reshape(self.M,1)

					# Solve for displacements
					self.PHS[1:,i,j]=Ainv.dot(series).flatten()

					# Solve for linear velocity and constant using polyfit
					fit=np.polyfit(self.times,self.PHS[:,i,j],1)
					self.V[i,j]=fit[0]
					self.C[i,j]=fit[1]


		## Slow method - pixel-by-pixel
		elif inpt.regularization==True:
			# Simulatenously solve for phase and velocity on a
			#  pixel-by-pixel basis
			for i in range(inpt.R):
				for j in range(inpt.S):
					# Interferogram values of pixel
					series=stack[:,i,j]
					series=series.reshape(self.M,1)

					# Add zeros for regularization
					series=np.concatenate([series,np.zeros((self.N-1,1))],axis=0)

					# Solve for dipslacement, velocity, and constant
					sln=Ainv.dot(series)

					# Add results to arrays
					self.PHS[1:,i,j]=sln[:-2].flatten()
					self.V[i,j]=sln[-2]
					self.C[i,j]=sln[-1]


	## Plot results
	def plotResults(self):
		## Plot velocity map
		velFig,velAx=mapPlot(self.V,cmap='jet',pctmin=inpt.pctmin,pctmax=inpt.pctmax,background=inpt.background,
			extent=None,showExtent=False,cbar_orientation='horizontal',title='LOS velocity')
		# Plot reference point
		if inpt.noRef is False: velAx.plot(self.refX,self.refY,'ks')

		return velFig, velAx


### PHASE ANALYSIS ---
def phaseAnalysis(event):
	print('Phase analysis')

	# Location
	px=event.xdata; py=event.ydata
	px=int(round(px)); py=int(round(py))

	# Report position and cumulative values
	print('px {} py {}'.format(px,py)) # report position
	print('long-term velocity {}'.format(S.V[py,px]))

	# Extract phase values
	phsValues=S.PHS[:,py,px]
	print('Phase values: {}'.format(phsValues))
	velocityFit=np.poly1d([S.V[py,px],S.C[py,px]])
	resids=phsValues-velocityFit(S.times)
	print('RMS {}'.format(np.sqrt(np.mean(resids**2))))

	# Plot phase over time
	phsAx.cla()
	phsAx.plot(S.epochs,velocityFit(S.times),'g',label='linear fit')
	phsAx.plot(S.epochs,phsValues,'k.',label='reconstructed phase')
	phsAx.set_xticks(S.epochs)
	labels=[datetime.strftime(epoch,'%Y%m%d') for epoch in S.epochs]
	phsAx.set_xticklabels(labels,rotation=80)
	phsFig.tight_layout()


	# Draw
	phsFig.canvas.draw()



### MAIN ---
if __name__=='__main__':
	# Gather inputs
	inpt=cmdParser()


	## Load data
	stack=loadARIAdata(inpt)

	# Plot data if requested
	if inpt.plotInputs:
		imagettes(stack,3,4,cmap='viridis',pctmin=inpt.pctmin,pctmax=inpt.pctmax,
			colorbarOrientation='horizontal',background=inpt.background,
			extent=inpt.T.extent,showExtent=False,titleList=inpt.pairNames,supTitle='Inputs')


	## Refereence point
	stack=refPoint(inpt,stack)


	## SBAS
	# Compute SBAS
	S=SBAS(inpt,stack)

	# Plot displacement timeseries
	velFig,velAx=S.plotResults()


	## Analyze displacement timeseries
	# Plot phase over time
	phsFig=plt.figure(figsize=(7,6))
	phsAx=phsFig.add_subplot(111)
	phsAx.set_title('Phase over time')
	phsAx.set_xlabel('time'); phsAx.set_ylabel('phase')

	# Interact with velocity figure
	velFig.canvas.mpl_connect('button_press_event',phaseAnalysis)



	plt.show()
