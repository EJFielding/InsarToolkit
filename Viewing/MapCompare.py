#!/usr/bin/env python3
"""
	Plot and regress the second map relative to the first. 


	by Rob Zinke 2019, updated 2020
"""

### IMPORT MODULES ---
import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal 
from geoFormatting import GDALtransform
from viewingFunctions import imgBackground, mapStats


### ARGUMENT PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Plot most types of Insar products, including complex images and multiband images')
	# Data sets
	parser.add_argument(dest='baseName', type=str, help='Map to serve as base (independent) data set')
	parser.add_argument(dest='compName', type=str, help='Map to serve as comparison (dependent) to the base')
	# Map options
	parser.add_argument('-b','--bounds', dest='bounds', type=str, default=None, help='Map bounds (minX, minY, maxX, maxY)')
	parser.add_argument('-ds','--downsample','--dsFactor', dest='dsFactor', type=int, default=0, help='Downsample factor (power of two)')
	# Masking
	parser.add_argument('-bg','--background', dest='background', default=None, help='Background value for both maps')
	parser.add_argument('--bgBase', dest='bgBase', default=None, help='Background value for base image')
	parser.add_argument('--bgComp', dest='bgComp', default=None, help='Background value for comparison image')
	# Map plot options
	parser.add_argument('-p','--plotMaps', dest='plotMaps', action='store_true', help='Plot maps')
	parser.add_argument('-c','--cmap','--colormap', dest='cmap', type=str, default='viridis', help='Colormap')
	parser.add_argument('-pctmin','--pctmin', dest='pctmin', type=float, default=0, help='Minimum percent clip')
	parser.add_argument('-pctmax','--pctmax', dest='pctmax', type=float, default=100, help='Maximum percent clip')
	# Comparison plot options
	parser.add_argument('-t','--plotType', dest='plotType', type=str, default='pts', help='Comparison plot type ([default = points], )')
	parser.add_argument('-s','--skips', dest='skips', type=int, default=1, help='Skip by this value when plotting points')
	parser.add_argument('--aspect','--plotAspect', dest='plotAspect', type=float, default=None, help='Comparison plot aspect ratio')
	# Analysis options
	parser.add_argument('-a','--analysis','--analysisType', dest='analysisType', type=str, default=None, help='Analysis type (polyfit, PCA, kmeans)')
	parser.add_argument('--degree', dest='degree', type=int, default=1, help='Polynomial degree for fit [default = 1]')
	parser.add_argument('--nbins', dest='nbins', type=int, default=10, help='Number of bins for 2D histograms')
	parser.add_argument('-k','--clusters','--kclusters', dest='kclusters', type=int, default=2, help='Number of clusters for k-means cluster analysis')
	parser.add_argument('--max-iterations', dest='maxIterations', type=int, default=20, help='Max number of iterations for k-means cluster analysis')

	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
## Transform to extent
def transform2extent(DS):
	N=DS.RasterXSize; M=DS.RasterYSize
	T=DS.GetGeoTransform()
	left=T[0]; dx=T[1]; right=left+N*dx
	top=T[3]; dy=T[5]; bottom=top+M*dy
	extent=(left, right, bottom, top)

	return extent


## Adjust size and resolution of images
def preFormat(inpt,baseDS,compDS):
	baseTnsf=baseDS.GetGeoTransform()

	## First adjust base image to desired extent/resolution
	# Adjust bounds of base image
	if not inpt.bounds:
		N=baseDS.RasterXSize; M=baseDS.RasterYSize
		minX=baseTnsf[0]; maxY=baseTnsf[3]
		dx=baseTnsf[1]; dy=baseTnsf[5]
		maxX=minX+dx*N; minY=maxY+dy*M
		inpt.bounds=(minX, minY, maxX, maxY) # (minX, minY, maxX, maxY)

	# Adjust resolution of base image
	dsFactor=int(2**inpt.dsFactor)
	dx=dsFactor*baseTnsf[1]; dy=dsFactor*baseTnsf[5]

	# Resample base image
	baseDS=gdal.Warp('',baseDS,options=gdal.WarpOptions(format='MEM',xRes=dx,yRes=dy,outputBounds=inpt.bounds,resampleAlg='bilinear'))

	## Then, resample comparison image to match base image
	compDS=gdal.Warp('',compDS,options=gdal.WarpOptions(format='MEM',xRes=dx,yRes=dy,outputBounds=inpt.bounds,resampleAlg='bilinear'))

	## Check that geotransforms are equal
	baseTnsf=baseDS.GetGeoTransform(); compTnsf=compDS.GetGeoTransform()
	if baseTnsf!=compTnsf:
		print('!!! WARNING WARNING WARNING!!!\n\t Geo-transforms are not equal!')

	return baseDS, compDS 


## Masking
def createMask(inpt,baseDS,compDS):
	commonMask=np.ones((baseDS.RasterYSize,baseDS.RasterXSize))

	# Determine all background values
	if inpt.background=='auto':
		baseImg=baseDS.GetRasterBand(1).ReadAsArray()
		bgBase=imgBackground(baseImg)
		# Apply mask
		commonMask[baseImg==bgBase]=0

		compImg=compDS.GetRasterBand(1).ReadAsArray()
		bgComp=imgBackground(compImg)
		# Apply mask
		commonMask[compImg==bgComp]=0

	# Mask base image
	if inpt.bgBase is not None:
		if inpt.bgBase=='auto':
			baseImg=baseDS.GetRasterBand(1).ReadAsArray()
		# Apply mask
		commonMask[baseImg==inpt.bgBase]=0

	# Mask comparison image
	if inpt.bgComp is not None:
		if inpt.bgComp=='auto':
			compImg=compDS.GetRasterBand(1).ReadAsArray()
		# Apply mask
		commonMask[compImg==inpt.bgComp]=0

	return commonMask


## Plot image data set
def plotDatasets(inpt,baseDS,compDS):
	# Geographic formatting
	extent=transform2extent(baseDS)

	# Masking
	baseImg=baseDS.GetRasterBand(1).ReadAsArray()
	baseImg=np.ma.array(baseImg,mask=(inpt.commonMask==0))

	compImg=compDS.GetRasterBand(1).ReadAsArray()
	compImg=np.ma.array(compImg,mask=(inpt.commonMask==0))

	# Min/max values
	baseStats=mapStats(baseImg,pctmin=inpt.pctmin,pctmax=inpt.pctmax)
	compStats=mapStats(compImg,pctmin=inpt.pctmin,pctmax=inpt.pctmax)

	# Plot maps
	Fig=plt.figure()

	axBase=Fig.add_subplot(121) # base figure
	caxBase=axBase.imshow(baseImg,
		vmin=baseStats.vmin,vmax=baseStats.vmax,extent=extent)
	Fig.colorbar(caxBase,orientation='horizontal')
	axBase.set_title('Base')

	axComp=Fig.add_subplot(122) # comp figure
	caxComp=axComp.imshow(compImg,
		vmin=compStats.vmin,vmax=compStats.vmax,extent=extent)
	axComp.set_yticks([])
	Fig.colorbar(caxComp,orientation='horizontal')
	axComp.set_title('Comparison')


## Plot image
def plotImg(inpt,img,title=None,extent=None):
	# Format image
	img=np.ma.array(img,mask=(inpt.commonMask==0))
	stats=mapStats(img,pctmin=inpt.pctmin,pctmax=inpt.pctmax)

	# Plot
	Fig=plt.figure()
	ax=Fig.add_subplot(111)
	cax=ax.imshow(img,cmap='jet',vmin=stats.vmin,vmax=stats.vmax,extent=extent)

	# Format
	ax.set_title(title)
	Fig.colorbar(cax,orientation='horizontal')


def computeKmeans(data,k,max_iterations):
		# Setup
		dataMin=np.min(data,axis=0)
		dataMax=np.max(data,axis=0)
		dataShape=data.shape
		np.random.seed(0)
		Dists=np.zeros((dataShape[0],k))
		# Pick initial centroids
		Centroids=np.linspace(dataMin,dataMax,k)
		Centroids_new=np.zeros((k,dataShape[1]))
		# Find closest centroids
		def find_closest_centroid(data,k):
			for i in range(k):
				# Subract each centroid from the data
				#  and compute Euclidean distance
				Dists[:,i]=np.linalg.norm(data-Centroids[i,:],
					ord=2,axis=1)
			return Dists
		# Loop through iterations
		while(max_iterations): # for each iteration...
		# 1. Calculate distance to points
			Dists=find_closest_centroid(data,k)
			Centroid_ndx=np.argmin(Dists,axis=1)
		# 2. Assign data points to centroids
			for j in range(k):
				cluster_mean=np.mean(data[Centroid_ndx==j],axis=0)
				Centroids_new[j,:]=cluster_mean
			if not np.sum(Centroids_new-Centroids):
				break
			Centroids=Centroids_new
			max_iterations-=1
		# Plot
		Dists=find_closest_centroid(data,k) # final calculation
		Centroid_ndx=np.argmin(Dists,axis=1) # final indices
		return Centroids



### Compare maps ---
class mapCompare:
	## Format data
	def __init__(self,baseDS,compDS,mask=None,verbose=False):
		self.verbose=verbose # True/False
		self.aProps=analysisProperties # dictionary of analysis properties

		# Load image bands and transforms
		baseImg=baseDS.GetRasterBand(1).ReadAsArray()
		compImg=compDS.GetRasterBand(1).ReadAsArray()

		## Simple difference map
		# Compute diff
		diff=compImg-baseImg

		# Plot diff
		extent=transform2extent(baseDS)
		plotImg(inpt,diff,title='Difference',extent=extent)


		## Compare 1D arrays
		# Mask values
		baseImg=np.ma.array(baseImg,mask=(mask==0))
		compImg=np.ma.array(compImg,mask=(mask==0))

		# Flatten to 1D arrays
		baseImg=baseImg.flatten()
		compImg=compImg.flatten()

		# Compress to ignore mask values
		self.base=baseImg.compressed(); del baseImg
		self.comp=compImg.compressed(); del compImg

		self.baseMean=self.base.mean()
		self.compMean=self.comp.mean()

		self.baseStd=self.base.std()
		self.compStd=self.comp.std()

		# Report if requested
		if self.verbose is True:
			print('Base mean {:.4f}; std {:.4f}'.format(self.baseMean,self.baseStd))
			print('Comparison mean {:.4f}; std {:.4f}'.format(self.compMean,self.compStd))


	## Plot comparison
	def plotComparison(self,plotProperties,analysisProperties):
		# Parse properties
		plotType=plotProperties['plotType']
		analysisType=analysisProperties['analysisType']

		if self.verbose is True:
			print('Plotting comparison type: {}'.format(plotType.upper()))


		# Establish figure
		self.Fig=plt.figure()
		self.ax=self.Fig.add_subplot(111)

		# Plot data
		plotType=plotType.lower()

		if plotType in ['pts','points']:
			self.plotPts(plotProperties)
		elif plotType in ['hex','hexbin']:
			self.plotHex(plotProperties)
		elif plotType in ['hist','hist2d','histogram']:
			self.plotHist(plotProperties)
		elif plotType in ['kde','contour','contourf']:
			self.plotKDE(plotProperties)
		else:
			print('Choose an appropriate plot type'); exit()


		# Conduct and plot analyses
		if analysisType: analysisType=analysisType.lower()

		if analysisType in ['poly','polyfit','polynomial']:
			self.polyFit(analysisProperties)
		elif analysisType in ['pca']:
			self.PCA(analysisProperties)
		elif analysisType in ['kmeans','cluster','clusters','kcluster','kclusters']:
			self.clusterAnalysis(analysisProperties)


		# Finalize figure formatting
		if plotProperties['plotAspect']: self.ax.set_aspect(plotProperties['plotAspect'])
		self.ax.set_xlabel('base data')
		self.ax.set_ylabel('comparison data')
		self.Fig.tight_layout()


	## Data plotting --
	# Points comparison
	def plotPts(self,plotProperties):
		# Parse plotProperties
		skips=plotProperties['skips']
		plotAspect=plotProperties['plotAspect']

		## Plot data points
		self.ax.plot(self.base[::skips],self.comp[::skips],marker='.',color=(0.5,0.5,0.5),linewidth=0,zorder=1)


	# Hexbin comparison
	def plotHex(self,plotProperties):
		# Parse plotProperties
		cmap=plotProperties['cmap']

		## Plot hexbin
		cax=self.ax.hexbin(self.base,self.comp,cmap=cmap)
		self.Fig.colorbar(cax,orientation='horizontal')


	# Histogram comparison
	def plotHist(self,plotProperties):
		# Parse plotProperties
		cmap=plotProperties['cmap']
		nbins=plotProperties['nbins']

		## Construct histogram
		H,xedges,yedges=np.histogram2d(self.base,self.comp,bins=nbins)
		H=H.T
		X,Y=np.meshgrid(xedges,yedges)

		## Plot histogram
		cax=self.ax.pcolormesh(X,Y,H,cmap=cmap)

		self.Fig.colorbar(cax,orientation='horizontal')


	# KDE comparison
	def plotKDE(self,plotProperties):
		# Parse plotProperties
		plotType=plotProperties['plotType'] # plot type for contour contourf
		cmap=plotProperties['cmap']
		nbins=plotProperties['nbins']

		## Compute kernel density estimate
		from scipy.stats import gaussian_kde

		x=np.linspace(self.base.min(),self.base.max(),nbins)
		y=np.linspace(self.comp.min(),self.comp.max(),nbins)

		X,Y=np.meshgrid(x,y)
		positions=np.vstack([X.flatten(),Y.flatten()])
		values=np.vstack([self.base.flatten(),self.comp.flatten()])
		kernel=gaussian_kde(values)
		H=kernel(positions)
		H=np.reshape(H,X.shape)

		## Plot KDE
		if plotType in ['kde']:
			cax=self.ax.pcolormesh(X,Y,H,cmap=cmap)
		elif plotType in ['contour']:
			cax=self.ax.contour(X,Y,H,cmap=cmap)
		elif plotType in ['contourf']:
			cax=self.ax.contourf(X,Y,H,cmap=cmap)


	## Analyses --
	# Polynomial fit
	def polyFit(self,analysisProperties):
		## Fit polynomial
		# Determine degree of polynomial
		degree=analysisProperties['degree']

		# Fit polynomial
		fit=np.polyfit(self.base,self.comp,deg=degree)

		## Plot polynomial
		# Evaluate polynomial at x-coordinates
		P=np.poly1d(fit)
		self.comp_hat=P(self.base) # predicted values

		# Compute statistics
		n=len(self.base)
		m=degree+1

		SStot=np.sum((self.comp-self.comp.mean())**2)
		SSres=np.sum((self.comp-self.comp_hat)**2)

		Rsquared=1-SSres/SStot

		MSE=np.sum((self.comp-self.comp_hat)**2)/(n-m)

		# Report if requested
		if self.verbose is True:
			print('Fitting degree: {}'.format(degree))
			[print('x^{}: {}'.format(degree-n,coeff)) for n,coeff in enumerate(fit)]
			print('Rsquared: {}'.format(Rsquared))
			print('MSE: {}'.format(MSE))

		## Plot polynomial
		# Plot polynomial fit
		ndx=np.argsort(self.base)
		self.ax.plot(self.base[ndx],self.comp_hat[ndx],'k--')

		# Text box
		textstr='Linear regression analysis'
		textstr+='\nRelationship:'
		for n,coeff in enumerate(fit):
			textstr+=' {:.3f} $x^{}$'.format(coeff,degree-n)
		textstr+='\n$R^2$: {:.4f}'.format(Rsquared)
		self.ax.text(0.05,0.85,textstr,transform=self.ax.transAxes)


	# PCA
	def PCA(self,analysisProperties):
		## PCA analysis
		# Note -- no analysisProperties accepted for now

		# Reformat in data matrix
		A=np.hstack([self.base.reshape(-1,1),self.comp.reshape(-1,1)])

		# Remove centroid
		A[:,0]=A[:,0]-self.baseMean
		A[:,1]=A[:,1]-self.compMean

		# Covariance matrix
		Cov=np.dot(A.T,A)/(A.shape[0]-1)

		# Eigen decomp
		eigvals,eigvecs=np.linalg.eig(Cov)
		sort_ndx=np.argsort(eigvals)[::-1] # reorder
		eigvals=eigvals[sort_ndx]
		eigvecs=eigvecs[:,sort_ndx]

		# Standard deviations
		eigstds=np.sqrt(eigvals)

		# Construct principal components
		PC1=eigstds[0]*eigvecs[:,0]
		PC2=eigstds[1]*eigvecs[:,1]

		# Percent variance
		totalVar=np.sum(eigvals) # sum of eigenvalues
		pctVar=100*eigvals/totalVar # percent variance explained

		# Relationship between base and comp
		eigslope=eigvecs[1,0]/eigvecs[0,0]

		# Report if requested
		if self.verbose is True:
			print('Eigenvalues: {}'.format(eigvals))
			print('Eigenvectors:\n{}'.format(eigvecs))


		## Plot PCA
		# Plot data centroid
		self.ax.plot(self.baseMean,self.compMean,'ko',zorder=2)

		# Plot eigenbasis
		scale=3 # standard deviations
		self.ax.plot([-scale*PC1[0]+self.baseMean,scale*PC1[0]+self.baseMean],\
			[-scale*PC1[1]+self.compMean,scale*PC1[1]+self.compMean],'k--')
		self.ax.plot([-scale*PC2[0]+self.baseMean,scale*PC2[0]+self.baseMean],\
			[-scale*PC2[1]+self.compMean,scale*PC2[1]+self.compMean],'k--')

		# Text box
		textstr='Principal component analysis'
		textstr+='\nRelationship: {:.3f}\nPC1: {:.3f}\nPC2: {:.3f}'.format(eigslope,pctVar[0],pctVar[1])
		self.ax.text(0.05,0.85,textstr,transform=self.ax.transAxes)


	# K-means cluster analysis
	def clusterAnalysis(self,analysisProperties):
		## K-clusters
		# Parse analysis properties
		k=analysisProperties['kclusters']
		max_iterations=analysisProperties['maxIterations']

		# Format data
		data=np.hstack([self.base.reshape(-1,1),self.comp.reshape(-1,1)])

		# Compute kmeans
		centroids=computeKmeans(data,k,max_iterations)

		# Report if requested
		if self.verbose is True:
			print('{} clusters computed'.format(k))
			print('Centroids:\n')
			[print('\t{}'.format(centroid)) for centroid in centroids]


		## Plot cluster centers
		for centroid in centroids:
			self.ax.plot(centroid[0],centroid[1],'bo')



### MAIN FUNCTION ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load maps
	# Load base data set
	baseDS=gdal.Open(inpt.baseName,gdal.GA_ReadOnly)

	# Load comparison data set
	compDS=gdal.Open(inpt.compName,gdal.GA_ReadOnly)


	## Pre-format data sets - sample to same map bounds and resolution
	baseDS,compDS=preFormat(inpt,baseDS,compDS)


	## Mask background and other values
	inpt.commonMask=createMask(inpt,baseDS,compDS)


	## Plot input images
	if inpt.plotMaps is True:
		plotDatasets(inpt,baseDS,compDS)


	## Compare two data sets
	# Format plot properties
	plotProperties=dict(plotType=inpt.plotType,skips=inpt.skips,plotAspect=inpt.plotAspect,cmap=inpt.cmap,nbins=inpt.nbins)

	# Format analysis properties
	analysisProperties=dict(analysisType=inpt.analysisType,degree=inpt.degree,kclusters=inpt.kclusters,maxIterations=inpt.maxIterations)

	# Conduct comparison
	comparison=mapCompare(baseDS,compDS,mask=inpt.commonMask,verbose=inpt.verbose)
	comparison.plotComparison(plotProperties=plotProperties,\
		analysisProperties=analysisProperties)


	plt.show()